import numpy as np
from scipy import signal

class RingBuffer:
    """Simple ring buffer for storing recent baseband samples."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=complex)
        self.write_pos = 0
        self.size = 0
    
    def push(self, data):
        """Add samples to the ring buffer."""
        n = len(data)
        if n >= self.capacity:
            # If data is larger than buffer, just keep the most recent samples
            self.buffer[:] = data[-self.capacity:]
            self.write_pos = 0
            self.size = self.capacity
        else:
            # Calculate how much fits before wrap
            space_before_wrap = self.capacity - self.write_pos
            
            if n <= space_before_wrap:
                # All data fits without wrapping
                self.buffer[self.write_pos:self.write_pos + n] = data
                self.write_pos = (self.write_pos + n) % self.capacity
            else:
                # Need to wrap around
                self.buffer[self.write_pos:] = data[:space_before_wrap]
                self.buffer[:n - space_before_wrap] = data[space_before_wrap:]
                self.write_pos = n - space_before_wrap
            
            self.size = min(self.size + n, self.capacity)
    
    def get_recent(self, n):
        """Get the n most recent samples."""
        if n > self.size:
            raise ValueError(f"Requested {n} samples but only {self.size} available")
        
        if self.size < self.capacity:
            # Buffer not full yet, data is contiguous
            return self.buffer[self.size - n:self.size].copy()
        else:
            # Buffer is full, may need to handle wrap-around
            start_pos = (self.write_pos - n) % self.capacity
            if start_pos < self.write_pos:
                # No wrap needed
                return self.buffer[start_pos:self.write_pos].copy()
            else:
                # Need to concatenate wrapped data
                return np.concatenate([
                    self.buffer[start_pos:],
                    self.buffer[:self.write_pos]
                ])
    
    def get_all(self):
        """Get all available samples in chronological order."""
        return self.get_recent(self.size)


class StreamingPreprocessor:
    """
    Streaming preprocessor that converts raw RF samples to complex baseband
    using continuous filtering and phase-coherent decimation.
    """
    
    def __init__(self, fs_orig, f1, f2, margin_cents=50, atten_dB=60, 
                 passband_ratio=0.8):
        """
        Initialize the streaming preprocessor with automatic parameter calculation.
        
        Parameters:
        -----------
        fs_orig : float
            Original sampling rate (Hz)
        f1 : float
            First tone frequency (Hz)
        f2 : float
            Second tone frequency (Hz)
        margin_cents : float
            Margin in cents around the tone separation (default: 50)
        atten_dB : float
            Stopband attenuation in dB (default: 60)
        passband_ratio : float
            Passband edge as fraction of Nyquist (default: 0.8)
        """
        self.fs_orig = fs_orig
        self.f1 = f1
        self.f2 = f2
        
        # Compute decimation parameters
        self.f_center = (f1 + f2) / 2
        cent_factor = 2 ** (margin_cents / 1200)
        window_margin = self.f_center * (cent_factor - 1)
        bandwidth = abs(f2 - f1) + 2 * window_margin
        fs_bb_min = 2.5 * bandwidth
        
        self.Q = max(1, int(np.floor(fs_orig / fs_bb_min /2))) 
        self.fs_out = fs_orig / self.Q
        
        # Design anti-alias filter
        nyquist = fs_orig / 2
        fp = min(bandwidth, passband_ratio * self.fs_out / 2) / nyquist
        fs_stop = (self.fs_out / 2) / nyquist
        
        if fp >= fs_stop:
            fs_stop = min(0.99, fp * 1.2)
        
        # Kaiser's formula for filter order
        delta_f = fs_stop - fp
        delta_p = 10 ** (-atten_dB / 20)
        A = -20 * np.log10(delta_p)
        D = (A - 7.95) / 14.36 if A > 21 else 0.9222
        N = int(np.ceil(D / delta_f))
        N = N + (N % 2)  # Make even
        
        # Parks-McClellan design
        bands = [0, fp, fs_stop, 0.5]
        desired = [1, 0]
        weights = [1, 10]
        self.b_lpf = signal.remez(N + 1, bands, desired, weight=weights)
        self.a_lpf = np.array([1.0])
        
        # Filter length
        self.M = len(self.b_lpf)
        
        # Phase accumulator for continuous LO
        self.phase_acc = 0.0
        self.omega = 2 * np.pi * self.f_center / fs_orig
        
        # Complex filter state - initialize for continuous filtering
        self.zi_complex = signal.lfilter_zi(self.b_lpf, self.a_lpf).astype(complex) * 0
        
        # Startup management
        self.startup_buffer = np.array([])
        self.startup_complete = False
        
        # Decimation tracking
        self.total_filtered_kept = 0  # Total filter outputs after dropping initial M-1
        
    def process(self, raw_samples):
        """
        Process a block of raw RF samples using continuous filtering.
        """
        # Handle startup buffering
        if not self.startup_complete:
            # Accumulate samples until we have at least M-1
            self.startup_buffer = np.concatenate([self.startup_buffer, raw_samples])
            
            if len(self.startup_buffer) < self.M - 1:
                # Not enough samples yet, return empty
                return np.array([], dtype=complex)
            
            # We have enough samples to start processing
            samples_to_process = self.startup_buffer
            self.startup_buffer = np.array([])  # Clear buffer
            self.startup_complete = True
        else:
            # Normal processing - process immediately
            samples_to_process = raw_samples
        
        N = len(samples_to_process)
        
        # Complex heterodyning with continuous phase
        phases = self.phase_acc + self.omega * np.arange(N)
        complex_lo = np.exp(-1j * phases)
        baseband = samples_to_process * complex_lo
        
        # Update phase accumulator
        self.phase_acc = (self.phase_acc + self.omega * N) % (2 * np.pi)
        
        # Continuous filtering with state preservation
        baseband_filt, self.zi_complex = signal.lfilter(
            self.b_lpf, self.a_lpf, baseband, zi=self.zi_complex
        )
        
        # Handle startup transient removal (only once)
        if not self.startup_complete:
            # This should not happen as we set startup_complete=True above,
            # but keeping for clarity
            pass
        elif self.total_filtered_kept == 0 and len(baseband_filt) >= self.M - 1:
            # First time processing after startup - drop first M-1 samples
            baseband_filt = baseband_filt[self.M - 1:]
        
        # Phase-coherent decimation
        decimated_samples = []
        
        # Find indices in the current filtered block that align with decimation grid
        for i in range(len(baseband_filt)):
            global_filtered_index = self.total_filtered_kept + i
            if global_filtered_index % self.Q == 0:
                decimated_samples.append(baseband_filt[i])
        
        # Update total filtered samples counter
        self.total_filtered_kept += len(baseband_filt)
        
        # Return decimated output
        if decimated_samples:
            return np.array(decimated_samples, dtype=complex)
        else:
            return np.array([], dtype=complex)
