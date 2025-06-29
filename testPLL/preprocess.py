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


class StreamingPreprocessor:
    """
    Streaming preprocessor that converts raw RF samples to complex baseband
    using proper overlap-save method for filtering and decimation.
    """
    
    def __init__(self, fs_in, fs_out, f_center, b_lpf, a_lpf):
        """
        Initialize the streaming preprocessor.
        
        Parameters:
        -----------
        fs_in : float
            Input sample rate (Hz)
        fs_out : float
            Output sample rate after decimation (Hz)
        f_center : float
            Center frequency for heterodyning (Hz)
        b_lpf : array
            Numerator coefficients of anti-alias filter
        a_lpf : array
            Denominator coefficients of anti-alias filter (must be 1 for FIR)
        """
        self.fs_in = fs_in
        self.fs_out = fs_out
        self.f_center = f_center
        self.b_lpf = b_lpf
        self.a_lpf = a_lpf
        
        # Check decimation factor is integer
        Q_exact = fs_in / fs_out
        if not Q_exact.is_integer():
            raise ValueError(f"Decimation factor must be integer, got {Q_exact}")
        self.Q = int(Q_exact)
        
        # Filter length
        self.M = len(b_lpf)
        
        # Transient length in decimated samples
        self.T_dec = int(np.ceil((self.M - 1) / self.Q))
        
        # Minimum samples needed to produce output
        # First block: need at least (2*T_dec + 1) decimated samples
        # This means we need at least (2*T_dec + 1) * Q input samples
        self.min_first_block = (2 * self.T_dec + 1) * self.Q
        # Subsequent blocks: need at least (T_dec + 1) decimated samples
        # With M-1 overlap, need at least (T_dec + 1) * Q - (M - 1) new samples
        self.min_subsequent_block = max(1, (self.T_dec + 1) * self.Q - (self.M - 1))
        
        # Phase accumulator for continuous LO
        self.phase_acc = 0.0
        self.omega = 2 * np.pi * f_center / fs_in
        
        # Filter states (for IIR filters)
        self.zi_i = signal.lfilter_zi(b_lpf, a_lpf) * 0
        self.zi_q = signal.lfilter_zi(b_lpf, a_lpf) * 0
        
        # Input buffer for accumulating samples
        self.input_buffer = np.array([])
        
        # Overlap buffer (M-1 samples)
        self.overlap_buffer = np.zeros(self.M - 1)
        
        # First block flag
        self.is_first_block = True
        
    def process(self, raw_samples):
        """
        Process a block of raw RF samples using overlap-save method.
        
        Parameters:
        -----------
        raw_samples : array
            Block of real-valued RF samples
            
        Returns:
        --------
        complex_baseband : array
            Decimated complex baseband samples, free of edge artifacts
        """
        # Accumulate input samples
        self.input_buffer = np.concatenate([self.input_buffer, raw_samples])
        
        # Check if we have enough samples to process
        if self.is_first_block:
            min_samples = self.min_first_block
        else:
            min_samples = self.min_subsequent_block
            
        if len(self.input_buffer) < min_samples:
            # Not enough samples yet, return empty array
            return np.array([], dtype=complex)
        
        # Process accumulated samples
        samples_to_process = self.input_buffer
        self.input_buffer = np.array([])  # Clear buffer
        
        N = len(samples_to_process)
        
        # Step 1: Prepare input with overlap
        if self.is_first_block:
            # First block: no overlap, just process samples
            extended_block = samples_to_process
            extended_N = N
        else:
            # Subsequent blocks: prepend M-1 samples from previous block
            extended_block = np.concatenate([self.overlap_buffer, samples_to_process])
            extended_N = N + self.M - 1
            
        # Step 2: Save overlap for next block
        if N >= self.M - 1:
            self.overlap_buffer = samples_to_process[-(self.M - 1):]
        else:
            # If block is smaller than M-1, pad with zeros
            self.overlap_buffer[-N:] = samples_to_process
            self.overlap_buffer[:-N] = 0
            
        # Step 3: Heterodyne with continuous phase
        phases = self.phase_acc + self.omega * np.arange(extended_N)
        
        i_het = extended_block * np.cos(phases)
        q_het = extended_block * np.sin(phases)
        
        # Update phase accumulator
        self.phase_acc = (self.phase_acc + self.omega * extended_N) % (2 * np.pi)
        
        # Step 4: Filter
        i_filt, self.zi_i = signal.lfilter(self.b_lpf, self.a_lpf, i_het, zi=self.zi_i)
        q_filt, self.zi_q = signal.lfilter(self.b_lpf, self.a_lpf, q_het, zi=self.zi_q)
        
        # Step 5: Decimate
        i_dec = i_filt[::self.Q]
        q_dec = q_filt[::self.Q]
        
        # Step 6: Discard transients according to overlap-save
        if self.is_first_block:
            # First block: discard T_dec samples from both ends
            if len(i_dec) > 2 * self.T_dec:
                i_clean = i_dec[self.T_dec:-self.T_dec]
                q_clean = q_dec[self.T_dec:-self.T_dec]
            else:
                # This shouldn't happen due to minimum sample check
                i_clean = np.array([])
                q_clean = np.array([])
            self.is_first_block = False
        else:
            # Subsequent blocks: discard only first T_dec samples
            if len(i_dec) > self.T_dec:
                i_clean = i_dec[self.T_dec:]
                q_clean = q_dec[self.T_dec:]
            else:
                # This shouldn't happen due to minimum sample check
                i_clean = np.array([])
                q_clean = np.array([])
                
        # Step 7: Return complex baseband
        return i_clean + 1j * q_clean
