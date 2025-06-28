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
    fs = (fs_bb / 2) / nyquist
    
    # Ensure valid band edges
    if fp >= fs:
        fs = min(0.99, fp * 1.2)
    
    # Estimate filter order using Kaiser's formula
    delta_f = fs - fp
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
    bands = [0, fp, fs, 1]
    desired = [1, 0]
    weights = [1, 10]  # Weight stopband more heavily
    
    b = signal.remez(N + 1, bands, desired, weights, Hz=2)
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
    """Recursive Least Squares tracker for multiple tones using EDA-LMS."""
    
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
        
        # EDA-LMS parameters
        self.alpha = 0.95#np.exp(-self.dt / T_mem)
        
        # State: complex amplitudes and frequencies for each tone
        # Parameters: [A1*exp(j*phi1), A2*exp(j*phi2), ..., omega1, omega2, ...]
        self.theta = np.zeros(2 * M, dtype=complex)
        
        # Initialize with random frequencies and small amplitudes
        for k in range(M):
            self.theta[k] = 0.1 * np.exp(1j * np.random.uniform(0, 2*np.pi))  # Complex amplitude
            self.theta[M + k] = np.random.uniform(-10, 10)  # Frequency offset in Hz
        
        # Sample counter
        self.n = 0
        
        # Regularization parameters
        self.lambda_reg = 1e-5  # Regularization weight
        self.beta_sep = 5000    # Separation sharpness
        
        # For error covariance estimation
        self.N_eff = int(T_mem * fs)
        self.recent_phi = []
        self.recent_weights = []
        self.sigma2_est = 1.0
        
    def init_from_bulk(self, x: np.ndarray):
        """
        Optional: bootstrap theta, P from a batch (ESPRIT/NLS).
        
        Args:
            x: Complex baseband signal array
        """
        # Placeholder - would implement ESPRIT or similar here
        # For now, keep random initialization
        pass
    
    def update(self, i: float, q: float) -> None:
        """
        One-sample RLS update given I/Q sample pair.
        
        Args:
            i: In-phase sample
            q: Quadrature sample
        """
        # Form complex sample
        x = i + 1j * q
        
        # Build regressor vector
        phi = np.zeros(self.M, dtype=complex)
        t = self.n * self.dt
        
        for k in range(self.M):
            omega_k = self.theta[self.M + k].real  # Frequency in Hz
            phi[k] = np.exp(1j * 2 * np.pi * omega_k * t)
        
        # Prediction using current parameters
        y_pred = np.sum(self.theta[:self.M] * phi)
        
        # Prediction error
        e = x - y_pred
        
        # Adaptive step size (approximating RLS)
        mu = (1 - self.alpha) / (np.sum(np.abs(phi)**2) + 1e-10)
        
        # Update complex amplitudes
        self.theta[:self.M] += mu * phi.conj() * e
        
        # Update frequencies with gradient descent
        for k in range(self.M):
            # Gradient of prediction w.r.t. omega_k
            dphi_domega = 1j * 2 * np.pi * t * phi[k]
            grad = -2 * np.real(e.conj() * self.theta[k] * dphi_domega)
            
            # Add regularization gradient to enforce separation
            for ell in range(self.M):
                if ell != k:
                    f_diff = self.theta[self.M + k].real - self.theta[self.M + ell].real
                    reg_grad = self.lambda_reg * self.beta_sep * np.sign(f_diff) * np.exp(-self.beta_sep * abs(f_diff))
                    grad -= mu * reg_grad
            
            # Update frequency
            self.theta[self.M + k] -= mu * grad * 0.1  # Scale down frequency updates for stability
        
        # Store regressor and weight for covariance estimation
        self.recent_phi.append(phi)
        self.recent_weights.append(self.alpha ** (self.n))
        
        # Keep only recent history
        if len(self.recent_phi) > self.N_eff:
            self.recent_phi.pop(0)
            self.recent_weights.pop(0)
        
        self.n += 1
    
    def get_state(self) -> Dict[str, Any]:
        """
        Return amplitudes, phases, frequencies, and error covariances.
        
        Returns:
            Dictionary with:
                - amplitudes: Array of tone amplitudes
                - phases: Array of tone phases (radians)
                - freqs: Array of tone frequencies (Hz)
                - cov: Placeholder for covariance matrix diagonal
        """
        amplitudes = np.abs(self.theta[:self.M])
        phases = np.angle(self.theta[:self.M])
        freqs = self.theta[self.M:].real
        
        # Placeholder for covariance - would compute from Fisher Information
        # For now, return small fixed values
        cov = np.ones(2 * self.M) * 0.01
        
        return {
            'amplitudes': amplitudes,
            'phases': phases,
            'freqs': freqs,
            'cov': cov
        }
