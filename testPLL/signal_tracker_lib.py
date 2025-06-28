"""Signal processing library for two-tone tracking with EDA-LMS."""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Any, Optional


def compute_decimation_params(fs_orig: float, f1: float, f2: float, margin_cents: float) -> Tuple[int, float, float]:
    """
    Compute decimation parameters for two-tone tracking.
    
    Args:
        fs_orig: Original sampling rate (Hz)
        f1: First tone frequency (Hz)
        f2: Second tone frequency (Hz)
        margin_cents: Margin in cents around the tone separation
        
    Returns:
        decimation_factor: Integer decimation factor Q
        window_margin: Frequency margin in Hz
        fs_bb: Resulting baseband sample rate
    """
    # Convert margin from cents to Hz
    # 100 cents = factor of 2^(1/12) ≈ 1.0595
    f_center = (f1 + f2) / 2
    cent_factor = 2 ** (margin_cents / 1200)  # 1200 cents per octave
    window_margin = f_center * (cent_factor - 1)
    
    # Total bandwidth needed (tone separation + margins)
    bandwidth = abs(f2 - f1) + 2 * window_margin
    
    # Nyquist requires fs_bb > 2 * bandwidth, add some headroom
    fs_bb_min = 2.5 * bandwidth
    
    # Find integer decimation factor
    decimation_factor = int(np.floor(fs_orig / fs_bb_min))
    decimation_factor = max(1, decimation_factor)  # Ensure at least 1
    
    fs_bb = fs_orig / decimation_factor
    
    return decimation_factor, window_margin, fs_bb


def design_anti_alias_filter(fs_orig: float, fs_bb: float, passband: float, atten_dB: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design FIR lowpass filter for decimation.
    
    Args:
        fs_orig: Original sample rate (Hz)
        fs_bb: Target baseband rate (Hz)
        passband: Passband edge frequency (Hz)
        atten_dB: Stopband attenuation (dB)
        
    Returns:
        b: FIR filter numerator coefficients
        a: FIR filter denominator (always [1.0] for FIR)
    """
    # Normalize frequencies to Nyquist
    nyquist = fs_orig / 2
    
    # Passband edge at 80% of final Nyquist frequency
    fp = min(passband, 0.8 * fs_bb / 2) / nyquist
    
    # Stopband starts at final Nyquist frequency
    fs_stop = (fs_bb / 2) / nyquist
    
    # Ensure valid band edges  
    if fp >= fs_stop:
        fs_stop = min(0.99, fp * 1.2)
    
    # Estimate filter order using Kaiser's formula
    delta_f = fs_stop - fp
    delta_p = 10 ** (-atten_dB / 20)  # Passband ripple same as stopband
    
    # Kaiser's approximation
    A = -20 * np.log10(delta_p)
    if A > 21:
        D = (A - 7.95) / 14.36
    else:
        D = 0.9222
    
    N = int(np.ceil(D / delta_f))
    N = N + (N % 2)  # Make even
    
    # Design using Parks-McClellan
    bands = [0, fp/2, fs_stop/2, 0.5]
    desired = [1, 0]
    weights = [1, 10]  # Weight stopband more heavily
    
    b = signal.remez(N + 1, bands, desired, weight=weights)
    a = np.array([1.0])
    
    return b, a


def heterodyne(x: np.ndarray, fs: float, lo_freq: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mix signal down to baseband using complex exponential.
    
    Args:
        x: Input signal
        fs: Sample rate (Hz)
        lo_freq: Local oscillator frequency (Hz)
        
    Returns:
        i: In-phase component
        q: Quadrature component
    """
    t = np.arange(len(x)) / fs
    lo_complex = np.exp(-2j * np.pi * lo_freq * t)
    
    # Mix down
    baseband = x * lo_complex
    
    # Extract I and Q
    i = np.real(baseband)
    q = np.imag(baseband)
    
    return i, q


def decimate(x: np.ndarray, decimation_factor: int, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Filter and downsample signal.
    
    Args:
        x: Input signal
        decimation_factor: Integer downsampling factor
        b: Filter numerator coefficients
        a: Filter denominator coefficients
        
    Returns:
        y: Decimated signal
    """
    # Apply filter
    filtered = signal.filtfilt(b, a, x)
    
    # Downsample
    decimated = filtered[::decimation_factor]
    
    return decimated


class ToneRLS:
    """Recursive Least Squares tracker for multiple tones with frequency estimation."""
    
    def __init__(self, M: int, fs: float, T_mem: float):
        """
        Initialize RLS tracker.
        
        Args:
            M: Number of sinusoids to track
            fs: Baseband sample rate (Hz)
            T_mem: Memory time constant (seconds)
        """
        self.M = M
        self.fs = fs
        self.T_mem = T_mem
        self.dt = 1.0 / fs
        
        # Forgetting factor
        self.lambda_forget = np.exp(-self.dt / T_mem)
        
        # State: [Re(A1), Im(A1), ..., Re(AM), Im(AM), f1, f2, ..., fM]
        # Using real representation for easier Jacobian computation
        self.theta = np.zeros(3 * M)
        
        # Initialize with random values
        for k in range(M):
            # Complex amplitude A_k = Re + j*Im
            amplitude = 0.1
            phase = np.random.uniform(0, 2*np.pi)
            self.theta[2*k] = amplitude * np.cos(phase)      # Real part
            self.theta[2*k + 1] = amplitude * np.sin(phase)  # Imag part
            # Frequency
            self.theta[2*M + k] = np.random.uniform(-10, 10)
        
        # Extended RLS covariance matrix
        self.P = np.eye(3 * M) * 1000
        
        # Sample counter
        self.n = 0
        
        # Regularization for frequency separation
        self.lambda_reg = 1e-5
        self.beta_sep = 5000
        
        # Error variance
        self.sigma2_est = 1.0
        
    def init_from_bulk(self, x: np.ndarray):
        """Optional: bootstrap from batch."""
        pass
    
    def update(self, i: float, q: float) -> None:
        """
        Extended RLS update for amplitude, phase, and frequency.
        
        Args:
            i: In-phase sample
            q: Quadrature sample
        """
        M = self.M
        x_real = i
        x_imag = q
        t = self.n * self.dt
        
        # Compute prediction and Jacobian
        y_pred_real = 0
        y_pred_imag = 0
        
        # Jacobian matrix (derivatives w.r.t. parameters)
        H = np.zeros((2, 3 * self.M))  # 2 rows for real/imag parts
        
        for k in range(self.M):
            # Extract parameters for tone k
            a_real = self.theta[2*k]
            a_imag = self.theta[2*k + 1]
            freq = self.theta[2*M + k]
            
            # Basis functions
            phase = 2 * np.pi * freq * t
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            
            # Prediction contribution
            y_pred_real += a_real * cos_phase - a_imag * sin_phase
            y_pred_imag += a_real * sin_phase + a_imag * cos_phase
            
            # Jacobian w.r.t. complex amplitude (linear part)
            H[0, 2*k] = cos_phase      # d(y_real)/d(a_real)
            H[0, 2*k + 1] = -sin_phase  # d(y_real)/d(a_imag)
            H[1, 2*k] = sin_phase       # d(y_imag)/d(a_real)
            H[1, 2*k + 1] = cos_phase   # d(y_imag)/d(a_imag)
            
            # Jacobian w.r.t. frequency (nonlinear part)
            d_phase = 2 * np.pi * t
            H[0, 2*M + k] = d_phase * (-a_real * sin_phase - a_imag * cos_phase)
            H[1, 2*M + k] = d_phase * (a_real * cos_phase - a_imag * sin_phase)
        
        # Innovation (prediction error)
        e = np.array([x_real - y_pred_real, x_imag - y_pred_imag])
        
        # Extended RLS gain
        S = H @ self.P @ H.T + self.lambda_forget * np.eye(2)
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update parameters
        self.theta += K @ e
        
        # Add frequency separation regularization
        for k in range(self.M):
            reg_grad = 0
            for ell in range(self.M):
                if ell != k:
                    f_diff = self.theta[2*M + k] - self.theta[2*M + ell]
                    reg_grad += self.lambda_reg * self.beta_sep * np.sign(f_diff) * np.exp(-self.beta_sep * abs(f_diff))
            self.theta[2*M + k] -= 0.001 * reg_grad  # Small step for regularization
        
        # Update covariance matrix
        self.P = (self.P - K @ H @ self.P) / self.lambda_forget
        Q = np.zeros((3*self.M, 3*self.M))
        
        
        df = 1  # degrees of freedom (lower = heavier tails)
        chi2 = np.random.chisquare(df)
        q_freq = 0.1 * df / chi2
        
        Q[2*self.M:, 2*self.M:] = np.eye(self.M) * q_freq
        self.P += Q
        
        # Update error variance estimate
        self.sigma2_est = self.lambda_forget * self.sigma2_est + (1 - self.lambda_forget) * np.dot(e, e)
        
        self.n += 1
    
    def get_state(self) -> Dict[str, Any]:
        """
        Return amplitudes, phases, frequencies, and error covariances.
        
        Returns:
            Dictionary with:
                - amplitudes: Array of tone amplitudes
                - phases: Array of tone phases (radians)
                - freqs: Array of tone frequencies (Hz)
                - cov: Covariance matrix diagonal
        """
        amplitudes = np.zeros(self.M)
        phases = np.zeros(self.M)
        freqs = np.zeros(self.M)
        
        for k in range(self.M):
            a_real = self.theta[2*k]
            a_imag = self.theta[2*k + 1]
            amplitudes[k] = np.sqrt(a_real**2 + a_imag**2)
            phases[k] = np.arctan2(a_imag, a_real)
            freqs[k] = self.theta[2*self.M + k]
        
        # Extract relevant diagonal elements from covariance
        cov = np.diag(self.P) * self.sigma2_est
        
        return {
            'amplitudes': amplitudes,
            'phases': phases,
            'freqs': freqs,
            'cov': cov
        }