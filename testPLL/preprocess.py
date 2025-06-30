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
    using proper overlap-save method for filtering and decimation.
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
        
        self.Q = max(1, int(np.floor(fs_orig / fs_bb_min)))
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
        
        # Parks-McClellan design (corrected band edges)
        bands = [0, fp, fs_stop, 0.5]
        desired = [1, 0]
        weights = [1, 10]
        self.b_lpf = signal.remez(N + 1, bands, desired, weight=weights)
        self.a_lpf = np.array([1.0])
        
        # Filter length
        self.M = len(self.b_lpf)
        
        # Transient length in decimated samples
        self.T_dec = int(np.ceil((self.M - 1) / self.Q))
        
        # Minimum samples needed
        self.min_first_block = (2 * self.T_dec + 1) * self.Q
        self.min_subsequent_block = max(1, (self.T_dec + 1) * self.Q - (self.M - 1))
        
        # Phase accumulator for continuous LO
        self.phase_acc = 0.0
        self.omega = 2 * np.pi * self.f_center / fs_orig
        
        # Complex filter state
        self.zi_complex = signal.lfilter_zi(self.b_lpf, self.a_lpf).astype(complex) * 0
        
        # Buffers
        self.input_buffer = np.array([])
        self.overlap_buffer = np.zeros(self.M - 1)
        
        # First block flag
        self.first_block = True
        
    def process(self, raw_samples):
        """
        Process a block of raw RF samples using overlap-save method.
        """
        # Accumulate input samples
        self.input_buffer = np.concatenate([self.input_buffer, raw_samples])
        
        # Process in fixed-size blocks
        output_blocks = []
        
        while True:
            # Determine block size to process
            if self.first_block:
                block_size = self.min_first_block
            else:
                block_size = self.min_subsequent_block
                
            # Check if we have enough samples
            if len(self.input_buffer) < block_size:
                break
                
            # Extract exactly block_size samples
            samples_to_process = self.input_buffer[:block_size]
            self.input_buffer = self.input_buffer[block_size:]  # Keep remaining samples
            
            N = len(samples_to_process)
            
            # Prepare input with overlap
            if self.first_block:
                extended_block = samples_to_process
            else:
                extended_block = np.concatenate([self.overlap_buffer, samples_to_process])
                
            # IMPORTANT: Use actual length of extended_block
            extended_N = len(extended_block)
            
            # Save overlap for next block
            if N >= self.M - 1:
                self.overlap_buffer = samples_to_process[-(self.M - 1):]
            else:
                # Handle case where block is smaller than filter length
                # Shift existing overlap buffer and add new samples
                self.overlap_buffer = np.concatenate([
                    self.overlap_buffer[N:],
                    samples_to_process
                ])
            
            # Complex heterodyning
            phases = self.phase_acc + self.omega * np.arange(extended_N)
            complex_lo = np.exp(-1j * phases)
            baseband = extended_block * complex_lo
            
            # Update phase accumulator
            self.phase_acc = (self.phase_acc + self.omega * extended_N) % (2 * np.pi)
            
            # Causal filtering with state preservation
            baseband_filt, self.zi_complex = signal.lfilter(
                self.b_lpf, self.a_lpf, baseband, zi=self.zi_complex
            )
            
            # Decimate
            baseband_dec = baseband_filt[::self.Q]
            
            # Remove transients
            if self.first_block:
                if len(baseband_dec) > 2 * self.T_dec:
                    clean = baseband_dec[self.T_dec:-self.T_dec]
                else:
                    clean = np.array([], dtype=complex)
                self.first_block = False
            else:
                if len(baseband_dec) > self.T_dec:
                    clean = baseband_dec[self.T_dec:]
                else:
                    clean = np.array([], dtype=complex)
                    
            output_blocks.append(clean)
        
        # Concatenate all output blocks
        if output_blocks:
            return np.concatenate(output_blocks)
        else:
            return np.array([], dtype=complex)
    

# Application code
def run_tracker_streaming(signal_rf, fs_orig, f1_rf, f2_rf, margin_cents, atten_dB,
                         M, T_mem, label, f1_init=None, f2_init=None):
    """Run tracker with streaming preprocessor and ring buffer."""
    
    # Initialize preprocessor with automatic parameter calculation
    preprocessor = StreamingPreprocessor(fs_orig, f1_rf, f2_rf, 
                                       margin_cents=margin_cents, 
                                       atten_dB=atten_dB,
                                       passband_ratio=0.8)
    
    # Get derived parameters from preprocessor
    fs_bb = preprocessor.fs_out
    f_center = preprocessor.f_center
    Q = preprocessor.Q
    
    print(f"  Preprocessor params: Q={Q}, fs_bb={fs_bb:.1f} Hz, f_center={f_center:.3f} Hz")
    
    # Initialize ring buffer for 2.5 seconds of baseband samples
    ring_buffer_size = int(2.5 * fs_bb)
    ring_buffer = RingBuffer(ring_buffer_size)
    
    # Initialize tracker
    tracker = ToneRLS(M, fs_bb, T_mem)
    
    # Calculate baseband frequencies
    f1_bb = f1_rf - f_center
    f2_bb = f2_rf - f_center
    
    # Override initial frequency estimates if provided
    if f1_init is not None and f2_init is not None:
        tracker.theta[2*M] = f1_init
        tracker.theta[2*M + 1] = f2_init
    
    # Calculate chunk size for 0.1 seconds of baseband samples
    bb_samples_per_block = int(0.1 * fs_bb)
    # Calculate corresponding RF samples needed (with some margin for filter transients)
    rf_samples_per_block = bb_samples_per_block * Q + preprocessor.M
    
    # Storage for history
    history = {
        'freq1': [],
        'freq2': [],
        'A1': [],
        'A2': [],
        'separation': [],
        'phase1': [],
        'phase2': []
    }
    
    # Accumulator for complete baseband signal
    baseband_accumulator = []
    
    # Process signal in chunks
    total_rf_samples = len(signal_rf)
    rf_position = 0
    
    while rf_position < total_rf_samples:
        # Get next chunk of RF samples
        chunk_end = min(rf_position + rf_samples_per_block, total_rf_samples)
        rf_chunk = signal_rf[rf_position:chunk_end]
        
        # Process through streaming preprocessor
        bb_chunk = preprocessor.process(rf_chunk)
        
        # Add to ring buffer and accumulator
        if len(bb_chunk) > 0:
            ring_buffer.push(bb_chunk)
            baseband_accumulator.append(bb_chunk)
            
            # Process new samples through RLS tracker sample-by-sample
            for sample in bb_chunk:
                tracker.update(np.real(sample), np.imag(sample))
                
                # Store history periodically
                if tracker.n % 10 == 0:  # Every 10 samples
                    state = tracker.get_state()
                    history['freq1'].append(state['freqs'][0])
                    history['freq2'].append(state['freqs'][1])
                    history['A1'].append(state['amplitudes'][0])
                    history['A2'].append(state['amplitudes'][1])
                    history['separation'].append(abs(state['freqs'][1] - state['freqs'][0]))
                    history['phase1'].append(state['phases'][0])
                    history['phase2'].append(state['phases'][1])
        
        rf_position = chunk_end
    
    # Get final state
    final_state = tracker.get_state()
    
    # Concatenate all baseband chunks
    data_full = np.concatenate(baseband_accumulator) if baseband_accumulator else np.array([], dtype=complex)
    
    # Get last 2.5 seconds from ring buffer
    data_last_frame = ring_buffer.get_all()
    
    result = {
        'label': label,
        'f1': final_state['freqs'][0],
        'f2': final_state['freqs'][1],
        'beat': final_state['freqs'][1] - final_state['freqs'][0],
        'f1_init': f1_init if f1_init is not None else tracker.theta[2*M],
        'f2_init': f2_init if f2_init is not None else tracker.theta[2*M + 1],
        'history': {k: np.array(v) for k, v in history.items()},
        'data_full': data_full,
        'data_last_frame': data_last_frame,
        'f1_bb': f1_bb,
        'f2_bb': f2_bb,
        'fs_bb': fs_bb
    }
    
    return result