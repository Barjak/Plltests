"""Signal processing library for two-tone tracking with EDA-LMS."""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Any, Optional
from scipy.linalg import eigh
from numpy.lib.stride_tricks import sliding_window_view

class ToneRLS:
    """Recursive Least Squares tracker for multiple tones with frequency estimation and velocity tracking."""
    
    def __init__(self, M: int, fs: float, T_mem: float, 
                 beta_smooth: float = 9.71e-02,
                 lambda_reg: float = 312e+03,
                 beta_sep: float = 3.04e+03,
                 sigma2_init: float = 3.51e+08,
                 q_freq_vel: float = 6.39e+02):
        """
        Initialize RLS tracker.
        [1.25e+03, 9.71e-02, 3.12e+06, 3.04e+03, 3.51e+08, 6.39e+02]
        [5.45e+03, 1.79e-01, 8.60e+04, 9.24e-03, 2.83e+03, 3.49e+03]
        Args:
            M: Number of sinusoids to track
            fs: Baseband sample rate (Hz)
            T_mem: Memory time constant (seconds)
            beta_smooth: Smoothing factor for frequency updates (0-1, default: 0.5)
            lambda_reg: Regularization strength for frequency separation (default: 1e-5)
            beta_sep: Separation penalty steepness (default: 5000)
            sigma2_init: Initial error variance estimate (default: 1e0)
            q_freq_vel: Process noise for frequency velocity (default: 1e2)
        """
        self.M = M
        self.fs = fs
        self.T_mem = T_mem
        self.dt = 1.0 / fs
        
        # Forgetting factor
        self.lambda_forget = np.exp(-self.dt / T_mem)
        
        # State: [Re(A1), Im(A1), ..., Re(AM), Im(AM), f1, ..., fM, f1_dot, ..., fM_dot]
        # Size: 2*M + M + M = 4*M
        self.theta = np.zeros(4 * M)
        
        self.beta_smooth = beta_smooth
        self.deltatheta_smooth = np.zeros_like(self.theta)

        # Initialize with random values
        for k in range(M):
            # Complex amplitude A_k = Re + j*Im
            amplitude = 0.5
            phase = np.random.uniform(0, 2*np.pi)
            self.theta[2*k] = amplitude * np.cos(phase)      # Real part
            self.theta[2*k + 1] = amplitude * np.sin(phase)  # Imag part
            # Frequency
            self.theta[2*M + k] = np.random.uniform(-10, 10)
            # Frequency derivative (initialized to 0)
            self.theta[3*M + k] = 0.0
            
        # Extended RLS covariance matrix
        self.P = np.eye(4 * M) * 1000
        
        # Sample counter
        self.n = 0
        
        # Regularization for frequency separation
        self.lambda_reg = lambda_reg
        self.beta_sep = beta_sep
        
        # Error variance
        self.sigma2_est = sigma2_init
        
        # Process noise parameters
        self.q_freq_vel = q_freq_vel
        
    def predict_state(self):
        """Predict next state using dynamics model."""
        # State transition: frequencies evolve according to their velocities
        for k in range(self.M):
            # f(t+1) = f(t) + f_dot(t) * dt
            self.theta[2*self.M + k] += self.theta[3*self.M + k] * self.dt
            # f_dot(t+1) = f_dot(t) (constant velocity model)
            # No change to amplitude states
            
    def update(self, i: float, q: float) -> None:
        """
        Extended RLS update for amplitude, phase, frequency, and frequency velocity.
        
        Args:
            i: In-phase sample
            q: Quadrature sample
        """
        M = self.M
        x_real = i
        x_imag = q
        t = self.n * self.dt
        
        # State prediction step
        self.predict_state()
        
        # State transition matrix F (4M x 4M)
        F = np.eye(4 * M)
        for k in range(M):
            # f(t+1) = f(t) + f_dot(t) * dt
            F[2*M + k, 3*M + k] = self.dt
        
        # Predict covariance
        self.P = F @ self.P @ F.T
        
        # Compute measurement prediction and Jacobian
        y_pred_real = 0
        y_pred_imag = 0
        
        # Jacobian matrix (derivatives w.r.t. parameters)
        H = np.zeros((2, 4 * self.M))  # 2 rows for real/imag measurements
        
        for k in range(self.M):
            # Extract parameters for tone k
            a_real = self.theta[2*k]
            a_imag = self.theta[2*k + 1]
            freq = self.theta[2*M + k]
            # freq_dot = self.theta[3*M + k]  # Not directly used in measurement
            
            # Basis functions
            phase = 2 * np.pi * freq * t
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            
            # Prediction contribution
            y_pred_real += a_real * cos_phase - a_imag * sin_phase
            y_pred_imag += a_real * sin_phase + a_imag * cos_phase
            
            # Jacobian w.r.t. complex amplitude
            H[0, 2*k] = cos_phase      # d(y_real)/d(a_real)
            H[0, 2*k + 1] = -sin_phase  # d(y_real)/d(a_imag)
            H[1, 2*k] = sin_phase       # d(y_imag)/d(a_real)
            H[1, 2*k + 1] = cos_phase   # d(y_imag)/d(a_imag)
            
            # Jacobian w.r.t. frequency
            d_phase = 2 * np.pi * t
            H[0, 2*M + k] = d_phase * (-a_real * sin_phase - a_imag * cos_phase)
            H[1, 2*M + k] = d_phase * (a_real * cos_phase - a_imag * sin_phase)
            
            # Jacobian w.r.t. frequency derivative is 0 (no direct measurement dependence)
            H[0, 3*M + k] = 0
            H[1, 3*M + k] = 0
        
        # Innovation (prediction error)
        e = np.array([x_real - y_pred_real, x_imag - y_pred_imag])
        
        # Extended RLS gain
        S = H @ self.P @ H.T + self.lambda_forget * np.eye(2)
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update parameters
        dtheta = K @ e
        
        # Update amplitudes directly
        self.theta[:2*M] += dtheta[:2*M]
        
        # Update frequencies with smoothing
        f_idx = slice(2*M, 3*M)
        old_f = self.theta[f_idx].copy()
        self.theta[f_idx] += (1 - self.beta_smooth) * dtheta[f_idx]
        
        # Update frequency velocities
        f_dot_idx = slice(3*M, 4*M)
        self.theta[f_dot_idx] += dtheta[f_dot_idx]
        
        # Add frequency separation regularization
        for k in range(M):
            reg_grad = 0
            for ell in range(M):
                if ell != k:
                    f_diff = self.theta[2*M + k] - self.theta[2*M + ell]
                    reg_grad += self.lambda_reg * self.beta_sep * np.sign(f_diff) * np.exp(-self.beta_sep * abs(f_diff))
            self.theta[2*M + k] -= 0.001 * reg_grad
        
        # Update covariance matrix
        self.P = (self.P - K @ H @ self.P) / self.lambda_forget
        
        # Process noise matrix Q
        Q = np.zeros((4*M, 4*M))
        # Add process noise to frequency velocities
        Q[3*M:, 3*M:] = np.eye(M) * self.q_freq_vel
        self.P += Q
        
        # Update error variance estimate
        self.sigma2_est = self.lambda_forget * self.sigma2_est + (1 - self.lambda_forget) * np.dot(e, e)
        
        self.n += 1
    
    def get_state(self) -> Dict[str, Any]:
        """
        Return amplitudes, phases, frequencies, frequency velocities, and error covariances.
        Results are sorted by frequency in ascending order.
        
        Returns:
            Dictionary with:
                - amplitudes: Array of tone amplitudes (sorted by frequency)
                - phases: Array of tone phases in radians (sorted by frequency)
                - freqs: Array of tone frequencies in Hz (sorted)
                - freq_dots: Array of frequency derivatives in Hz/s (sorted)
                - cov: Covariance matrix diagonal (reordered to match sorting)
        """
        amplitudes = np.zeros(self.M)
        phases = np.zeros(self.M)
        freqs = np.zeros(self.M)
        freq_dots = np.zeros(self.M)
        
        # First extract all parameters
        for k in range(self.M):
            a_real = self.theta[2*k]
            a_imag = self.theta[2*k + 1]
            amplitudes[k] = np.sqrt(a_real**2 + a_imag**2)
            phases[k] = np.arctan2(a_imag, a_real)
            freqs[k] = self.theta[2*self.M + k]
            freq_dots[k] = self.theta[3*self.M + k]
        
        # Get sorting indices based on frequency
        sort_idx = np.argsort(freqs)
        
        # Extract and reorder covariance diagonal
        cov_full = np.diag(self.P) * self.sigma2_est
        cov_reordered = np.zeros(4 * self.M)
        
        # Reorder covariance to match sorted frequencies
        for new_idx, old_idx in enumerate(sort_idx):
            # Amplitude covariances (real and imaginary parts)
            cov_reordered[2*new_idx] = cov_full[2*old_idx]
            cov_reordered[2*new_idx + 1] = cov_full[2*old_idx + 1]
            # Frequency covariances
            cov_reordered[2*self.M + new_idx] = cov_full[2*self.M + old_idx]
            # Frequency derivative covariances
            cov_reordered[3*self.M + new_idx] = cov_full[3*self.M + old_idx]
        
        return {
            'amplitudes': amplitudes[sort_idx],
            'phases': phases[sort_idx],
            'freqs': freqs[sort_idx],
            'freq_dots': freq_dots[sort_idx],
            'cov': cov_reordered
        }
class ToneEKF:
    """Extended Kalman Filter for dual-tone tracking with minimum separation enforcement."""
    
    def __init__(self, M: int, fs: float, T_mem: float, 
                 min_separation_hz: float = 0.008,
                 separation_buffer_hz: float = 0.001,
                 separation_weight: float = 10.1):
        """
        Initialize dual-tone EKF tracker (backwards compatible with RLS interface).
        
        Args:
            M: Number of sinusoids (must be 2)
            fs: Baseband sample rate (Hz)
            T_mem: Memory time constant (seconds) - for RLS compatibility
            min_separation_hz: Minimum allowed frequency separation (default 6 mHz)
            separation_buffer_hz: Buffer above minimum separation (default 1 mHz)
            separation_weight: Weight for pseudo-measurement (default 0.01)
        """
        assert M == 2, "DualToneEKF only supports M=2"
        self.M = M
        self.fs = fs
        self.T_mem = T_mem
        self.dt = 1.0 / fs
        
        # Pseudo-measurement parameters
        self.min_separation_hz = min_separation_hz
        self.separation_buffer_hz = separation_buffer_hz
        self.separation_weight = separation_weight
        self.target_separation = min_separation_hz + separation_buffer_hz
        
        # State: [phi1, phi2, f1, f2, A1, A2]
        self.x = np.array([0.0, 0.0, -5.0, 5.0, 1.0, 1.0])  # Default frequencies at ±5 Hz
        
        # State transition matrix
        self.F = np.array([[1, 0, 2*np.pi*self.dt, 0, 0, 0],
                          [0, 1, 0, 2*np.pi*self.dt, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])
        
        # Process noise covariance
        sigma_phi = 1e-7
        sigma_f = 1e-4
        sigma_A = 1e-6
        self.Q = np.diag([sigma_phi**2, sigma_phi**2, 
                         sigma_f**2, sigma_f**2,
                         sigma_A**2, sigma_A**2])
        
        # RLS-style memory-based Q scaling (commented out)
        # lambda_forget = np.exp(-self.dt / T_mem)
        # scale_factor = (1 - lambda_forget) / lambda_forget
        # self.Q *= scale_factor
        
        # State covariance
        self.P = np.diag([0.1, 0.1, 0.05, 0.05, 0.01, 0.01])
        
        # Measurem
#        ent noise covariance
        self.R = 0.01
        
        # Sample counter
        self.n = 0
    
    def init_from_bulk(self, x: np.ndarray):
        """Optional: bootstrap from batch (for compatibility)."""
        pass
    
    def update(self, i: float, q: float) -> None:
        """
        EKF update for dual-tone tracking.
        
        Args:
            i: In-phase sample
            q: Quadrature sample
        """
        # Complex measurement
        y = i + 1j * q
        
        # ---- Predict ----
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Wrap phases to [-pi, pi]
        self.x[0] = np.angle(np.exp(1j * self.x[0]))
        self.x[1] = np.angle(np.exp(1j * self.x[1]))
        
        # ---- Measurement Update ----
        phi1, phi2, f1, f2 = self.x[0], self.x[1], self.x[2], self.x[3]
        A1, A2 = self.x[4], self.x[5]
        
        # Complex measurement prediction
        y_hat = A1 * np.exp(1j * phi1) + A2 * np.exp(1j * phi2)
        
        # Jacobian H
        H = np.zeros((2, 6))
        
        # Derivatives w.r.t phi1
        H[0, 0] = -A1 * np.sin(phi1)  # dRe/dphi1
        H[1, 0] = A1 * np.cos(phi1)   # dIm/dphi1
        
        # Derivatives w.r.t phi2
        H[0, 1] = -A2 * np.sin(phi2)  # dRe/dphi2
        H[1, 1] = A2 * np.cos(phi2)   # dIm/dphi2
        
        # Derivatives w.r.t A1
        H[0, 4] = np.cos(phi1)  # dRe/dA1
        H[1, 4] = np.sin(phi1)  # dIm/dA1
        
        # Derivatives w.r.t A2
        H[0, 5] = np.cos(phi2)  # dRe/dA2
        H[1, 5] = np.sin(phi2)  # dIm/dA2
        
        # Innovation
        z = np.array([np.real(y), np.imag(y)])
        z_hat = np.array([np.real(y_hat), np.imag(y_hat)])
        innov = z - z_hat
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R * np.eye(2)
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update
        self.x = self.x + K @ innov
        self.P = (np.eye(6) - K @ H) @ self.P
        
        # ---- Pseudo-measurement for minimum separation ----
        current_sep = self.x[3] - self.x[2]
        
        if current_sep < self.target_separation:
            # Pseudo-measurement update on the separation f2 - f1
            H_pseudo = np.zeros((1, 6))
            H_pseudo[0, 2] = -1  # df1
            H_pseudo[0, 3] = 1   # df2
            
            z_pseudo = np.array([self.target_separation])
            z_hat_pseudo = np.array([current_sep])
            
            R_pseudo = 1.0 / self.separation_weight
            
            S_pseudo = H_pseudo @ self.P @ H_pseudo.T + R_pseudo
            K_pseudo = self.P @ H_pseudo.T / S_pseudo
            
            self.x = self.x + K_pseudo.flatten() * (z_pseudo - z_hat_pseudo)
            self.P = (np.eye(6) - np.outer(K_pseudo, H_pseudo)) @ self.P
        
        # Ensure ordering is maintained (f1 < f2)
        if self.x[2] > self.x[3]:
            self.x[2], self.x[3] = self.x[3], self.x[2]
            self.x[0], self.x[1] = self.x[1], self.x[0]
            self.x[4], self.x[5] = self.x[5], self.x[4]
        
        # Ensure positive amplitudes
        self.x[4] = max(0.1, self.x[4])
        self.x[5] = max(0.1, self.x[5])
        
        self.n += 1
    
    def get_state(self) -> Dict[str, Any]:
        """
        Return state in RLS-compatible format.
        Results are sorted by frequency in ascending order.
        
        Returns:
            Dictionary with:
                - amplitudes: Array of tone amplitudes
                - phases: Array of tone phases in radians  
                - freqs: Array of tone frequencies in Hz
                - cov: Covariance matrix diagonal (RLS format)
        """
        # Extract states (already ordered as f1 < f2)
        phi1, phi2 = self.x[0], self.x[1]
        f1, f2 = self.x[2], self.x[3]
        A1, A2 = self.x[4], self.x[5]
        
        # Convert to RLS format
        amplitudes = np.array([A1, A2])
        phases = np.array([phi1, phi2])
        freqs = np.array([f1, f2])
        
        # Convert covariance to RLS format
        # RLS format: [Re(A1), Im(A1), Re(A2), Im(A2), f1, f2]
        # We need to transform from EKF format: [phi1, phi2, f1, f2, A1, A2]
        
        # Jacobian for the transformation
        J = np.zeros((6, 6))
        
        # Re(A1) = A1 * cos(phi1)
        J[0, 0] = -A1 * np.sin(phi1)  # d/dphi1
        J[0, 4] = np.cos(phi1)         # d/dA1
        
        # Im(A1) = A1 * sin(phi1)
        J[1, 0] = A1 * np.cos(phi1)   # d/dphi1
        J[1, 4] = np.sin(phi1)         # d/dA1
        
        # Re(A2) = A2 * cos(phi2)
        J[2, 1] = -A2 * np.sin(phi2)  # d/dphi2
        J[2, 5] = np.cos(phi2)         # d/dA2
        
        # Im(A2) = A2 * sin(phi2)
        J[3, 1] = A2 * np.cos(phi2)   # d/dphi2
        J[3, 5] = np.sin(phi2)         # d/dA2
        
        # f1, f2 are direct
        J[4, 2] = 1.0
        J[5, 3] = 1.0
        
        # Transform covariance
        P_rls = J @ self.P @ J.T
        cov_diag = np.diag(P_rls)
        
        return {
            'amplitudes': amplitudes,
            'phases': phases,
            'freqs': freqs,
            'cov': cov_diag
        }
    
    def get_state_ekf(self) -> Dict[str, Any]:
        """
        Return state in natural EKF format.
        
        Returns:
            Dictionary with:
                - phi1, phi2: Phases in radians
                - f1, f2: Frequencies in Hz (f1 < f2)
                - A1, A2: Amplitudes
                - P: Full 6x6 covariance matrix
                - separation: Current frequency separation
        """
        return {
            'phi1': self.x[0],
            'phi2': self.x[1],
            'f1': self.x[2],
            'f2': self.x[3],
            'A1': self.x[4],
            'A2': self.x[5],
            'P': self.P.copy(),
            'separation': self.x[3] - self.x[2]
        }


class ToneLM:
    """Block-based Levenberg-Marquardt tracker for multiple tones."""
    
    def __init__(self, M: int, fs: float, T_window: float = 2.5, T_update: float = 0.1,
                 target_freq: Optional[float] = None):
        """
        Initialize LM tracker.
        [6.5, 5.3e-1, 3.0e-2, 1e6, 4e4, 5.9e5]
        Args:
            M: Number of sinusoids to track
            fs: Baseband sample rate (Hz)
            T_window: Window size in seconds
            T_update: Update interval in seconds
            target_freq: Optional target frequency for initialization (Hz)
        """
        self.M = M
        self.fs = fs
        self.dt = 1.0 / fs
        self.T_window = T_window
        self.T_update = T_update
        self.target_freq = target_freq if target_freq is not None else 0.0
        
        # Buffer management
        self.max_samples = int(T_window * fs)
        self.update_samples = int(T_update * fs)
        self.buffer_i = np.zeros(self.max_samples)
        self.buffer_q = np.zeros(self.max_samples)
        self.buffer_pos = 0
        self.buffer_filled = False
        self.samples_since_update = 0
        
        # Parameter state: [Re(A1), Im(A1), ..., Re(AM), Im(AM), f1, ..., fM]
        self.theta = np.zeros(3 * M)
        self.theta_initialized = False
        
        # LM parameters
        self.max_iter = 100
        self.tol = 1e-8
        self.mu_factor = 10.0
        
        # Regularization
        self.lambda_reg = 1e-5
        self.beta_sep = 5000
        
        # Covariance estimate
        self.cov_theta = np.eye(3 * M) * 1000
        
    def update(self, i: float, q: float) -> None:
        """Add a sample and run optimization if needed."""
        # Add to circular buffer
        self.buffer_i[self.buffer_pos] = i
        self.buffer_q[self.buffer_pos] = q
        self.buffer_pos = (self.buffer_pos + 1) % self.max_samples
        
        if self.buffer_pos == 0:
            self.buffer_filled = True
            
        self.samples_since_update += 1
        
        # Check if time to optimize
        if self.samples_since_update >= self.update_samples:
            self._run_optimization()
            self.samples_since_update = 0
    
    def init_from_bulk(self, x: np.ndarray) -> None:
        """Initialize from bulk complex samples."""
        n_samples = min(len(x), self.max_samples)
        self.buffer_i[:n_samples] = x[:n_samples].real
        self.buffer_q[:n_samples] = x[:n_samples].imag
        self.buffer_pos = n_samples % self.max_samples
        if n_samples >= self.max_samples:
            self.buffer_filled = True
        self._run_optimization()
        
    def _get_buffer_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract current buffer data in correct order."""
        if self.buffer_filled:
            n_samples = self.max_samples
            # Reorder circular buffer
            idx = np.arange(n_samples)
            idx = (idx + self.buffer_pos) % n_samples
            i_data = self.buffer_i[idx]
            q_data = self.buffer_q[idx]
        else:
            n_samples = self.buffer_pos
            i_data = self.buffer_i[:n_samples]
            q_data = self.buffer_q[:n_samples]
            
        t = np.arange(n_samples) * self.dt
        return i_data, q_data, t
    
    def _initialize_params(self, i_data: np.ndarray, q_data: np.ndarray) -> None:
        """Initialize parameters using target frequency."""
        # Initialize frequencies around target frequency
        if self.M == 1:
            self.theta[2*self.M] = self.target_freq
        else:
            # Spread frequencies around target
            freq_spacing = 2.0  # Hz spacing between initial frequencies
            freq_offset = freq_spacing * (self.M - 1) / 2
            for k in range(self.M):
                self.theta[2*self.M + k] = self.target_freq - freq_offset + k * freq_spacing
        
        # Initialize amplitudes
        for k in range(self.M):
            # Small initial amplitude
            amplitude = 0.1
            phase = np.random.uniform(0, 2*np.pi)
            self.theta[2*k] = amplitude * np.cos(phase)      # Real part
            self.theta[2*k + 1] = amplitude * np.sin(phase)  # Imag part
            
        self.theta_initialized = True
    
    def _compute_residual_and_jacobian(self, theta: np.ndarray, i_data: np.ndarray, 
                                     q_data: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute residual and Jacobian for current parameters."""
        n_samples = len(t)
        M = self.M
        
        # Residual vector [real parts; imag parts]
        r = np.zeros(2 * n_samples)
        # Jacobian matrix
        J = np.zeros((2 * n_samples, 3 * M))
        
        # Compute predicted signal and derivatives
        for n in range(n_samples):
            y_pred_real = 0
            y_pred_imag = 0
            
            for k in range(M):
                a_real = theta[2*k]
                a_imag = theta[2*k + 1]
                freq = theta[2*M + k]
                
                phase = 2 * np.pi * freq * t[n]
                cos_phase = np.cos(phase)
                sin_phase = np.sin(phase)
                
                # Prediction
                y_pred_real += a_real * cos_phase - a_imag * sin_phase
                y_pred_imag += a_real * sin_phase + a_imag * cos_phase
                
                # Jacobian entries
                J[n, 2*k] = cos_phase              # d(y_real)/d(a_real)
                J[n, 2*k + 1] = -sin_phase         # d(y_real)/d(a_imag)
                J[n + n_samples, 2*k] = sin_phase  # d(y_imag)/d(a_real)
                J[n + n_samples, 2*k + 1] = cos_phase  # d(y_imag)/d(a_imag)
                
                # Frequency derivatives
                d_phase_dt = 2 * np.pi * t[n]
                J[n, 2*M + k] = d_phase_dt * (-a_real * sin_phase - a_imag * cos_phase)
                J[n + n_samples, 2*M + k] = d_phase_dt * (a_real * cos_phase - a_imag * sin_phase)
            
            # Residuals
            r[n] = i_data[n] - y_pred_real
            r[n + n_samples] = q_data[n] - y_pred_imag
            
        return r, J
    
    def _compute_regularization_with_hessian(self, theta: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute frequency separation regularization term, gradient, and Hessian."""
        M = self.M
        reg_cost = 0
        reg_grad = np.zeros(3 * M)
        reg_hess = np.zeros((3 * M, 3 * M))
        
        for k in range(M):
            for ell in range(k + 1, M):
                f_diff = theta[2*M + k] - theta[2*M + ell]
                exp_term = np.exp(-self.beta_sep * abs(f_diff))
                reg_cost += self.lambda_reg * exp_term
                
                # Gradient
                sign_diff = np.sign(f_diff) if f_diff != 0 else 0
                grad_term = -self.lambda_reg * self.beta_sep * sign_diff * exp_term
                reg_grad[2*M + k] += grad_term
                reg_grad[2*M + ell] -= grad_term
                
                # Hessian
                hess_term = self.lambda_reg * self.beta_sep**2 * exp_term
                # Diagonal terms
                reg_hess[2*M + k, 2*M + k] += hess_term
                reg_hess[2*M + ell, 2*M + ell] += hess_term
                # Off-diagonal terms
                reg_hess[2*M + k, 2*M + ell] -= hess_term
                reg_hess[2*M + ell, 2*M + k] -= hess_term
                
        return reg_cost, reg_grad, reg_hess
    
    def _levenberg_marquardt(self, i_data: np.ndarray, q_data: np.ndarray, 
                           t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Levenberg-Marquardt optimization."""
        theta = self.theta.copy()
        
        # Initial cost
        r, J = self._compute_residual_and_jacobian(theta, i_data, q_data, t)
        reg_cost, reg_grad, reg_hess = self._compute_regularization_with_hessian(theta)
        cost = 0.5 * np.dot(r, r) + reg_cost
        
        # Initialize mu based on maximum diagonal of J.T @ J
        H = J.T @ J
        diag_H = np.diag(H)
        mu = 1e-3 * np.max(diag_H)
        
        for iteration in range(self.max_iter):
            # Compute gradient and full Hessian (including regularization)
            g = J.T @ r + reg_grad
            H_full = H + reg_hess
            
            # LM update with diagonal damping
            H_lm = H_full + mu * np.diag(diag_H)
            try:
                delta = np.linalg.solve(H_lm, -g)
            except np.linalg.LinAlgError:
                # Singular matrix, increase damping
                mu *= self.mu_factor
                continue
                
            # Test new parameters
            theta_new = theta + delta
            r_new, J_new = self._compute_residual_and_jacobian(theta_new, i_data, q_data, t)
            reg_cost_new, reg_grad_new, reg_hess_new = self._compute_regularization_with_hessian(theta_new)
            cost_new = 0.5 * np.dot(r_new, r_new) + reg_cost_new
            
            # Accept or reject update
            if cost_new < cost:
                # Accept
                theta = theta_new
                r = r_new
                J = J_new
                reg_grad = reg_grad_new
                reg_hess = reg_hess_new
                cost = cost_new
                H = J.T @ J
                diag_H = np.diag(H)
                mu /= self.mu_factor
                
                # Check convergence
                if np.linalg.norm(delta) / (np.linalg.norm(theta) + 1e-10) < self.tol:
                    break
            else:
                # Reject, increase damping
                mu *= self.mu_factor
                
        # Estimate covariance from final Hessian
        try:
            H_final = J.T @ J + reg_hess
            sigma2 = np.dot(r, r) / (len(r) - 3 * self.M)
            self.cov_theta = sigma2 * np.linalg.inv(H_final + 1e-10 * np.eye(3 * self.M))
        except:
            # Keep previous covariance if inversion fails
            pass
            
        return theta, self.cov_theta
    
    def _run_optimization(self) -> None:
        """Run LM optimization on current buffer."""
        i_data, q_data, t = self._get_buffer_data()
        
        if len(i_data) < 10 * self.M:  # Need enough samples
            return
            
        # Initialize if needed
        if not self.theta_initialized:
            self._initialize_params(i_data, q_data)
            
        # Run optimization
        self.theta, self.cov_theta = self._levenberg_marquardt(i_data, q_data, t)
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state with parameters sorted by frequency."""
        amplitudes = np.zeros(self.M)
        phases = np.zeros(self.M)
        freqs = np.zeros(self.M)
        
        # Extract parameters
        for k in range(self.M):
            a_real = self.theta[2*k]
            a_imag = self.theta[2*k + 1]
            amplitudes[k] = np.sqrt(a_real**2 + a_imag**2)
            phases[k] = np.arctan2(a_imag, a_real)
            freqs[k] = self.theta[2*self.M + k]
        
        # Sort by frequency
        sort_idx = np.argsort(freqs)
        
        # Reorder covariance diagonal
        cov_diag = np.diag(self.cov_theta)
        cov_reordered = np.zeros(3 * self.M)
        
        for new_idx, old_idx in enumerate(sort_idx):
            cov_reordered[2*new_idx] = cov_diag[2*old_idx]
            cov_reordered[2*new_idx + 1] = cov_diag[2*old_idx + 1]
            cov_reordered[2*self.M + new_idx] = cov_diag[2*self.M + old_idx]
        
        return {
            'amplitudes': amplitudes[sort_idx],
            'phases': phases[sort_idx],
            'freqs': freqs[sort_idx],
            'cov': cov_reordered
        }




import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy.linalg import eigh, toeplitz
from numpy.polynomial import polynomial as P

class ToneMUSIC:
    """Block-based root-MUSIC tracker for multiple tones."""
    
    def __init__(self, M: int, fs: float, T_window: float = 2.5, T_update: float = 0.1,
                 target_freq: Optional[float] = None, subarray_factor: float = 0.5):
        """
        Initialize root-MUSIC tracker.
        
        Args:
            M: Number of sinusoids to track
            fs: Baseband sample rate (Hz)
            T_window: Window size in seconds
            T_update: Update interval in seconds
            target_freq: Optional target frequency for initialization (Hz)
            subarray_factor: Fraction of window size to use for subarray length
        """
        self.M = M
        self.fs = fs
        self.dt = 1.0 / fs
        self.T_window = T_window
        self.T_update = T_update
        self.target_freq = target_freq if target_freq is not None else 0.0
        self.subarray_factor = subarray_factor
        
        # Buffer management
        self.max_samples = int(T_window * fs)
        self.update_samples = int(T_update * fs)
        self.buffer_i = np.zeros(self.max_samples)
        self.buffer_q = np.zeros(self.max_samples)
        self.buffer_pos = 0
        self.buffer_filled = False
        self.samples_since_update = 0
        
        # MUSIC parameters
        self.subarray_length = max(2 * M + 1, int(self.max_samples * subarray_factor))
        self.snapshot_count = self.max_samples - self.subarray_length + 1
        
        # State storage
        self.freqs = np.zeros(M)
        self.amplitudes = np.zeros(M)
        self.phases = np.zeros(M)
        self.initialized = False
        
        # Covariance estimate (simplified - frequency uncertainties only)
        self.freq_vars = np.ones(M) * 10.0  # Hz^2
        
    def update(self, i: float, q: float) -> None:
        """Add a sample and run root-MUSIC if needed."""
        # Add to circular buffer
        self.buffer_i[self.buffer_pos] = i
        self.buffer_q[self.buffer_pos] = q
        self.buffer_pos = (self.buffer_pos + 1) % self.max_samples
        
        if self.buffer_pos == 0:
            self.buffer_filled = True
            
        self.samples_since_update += 1
        
        # Check if time to run MUSIC
        if self.samples_since_update >= self.update_samples:
            self._run_music()
            self.samples_since_update = 0
    
    def init_from_bulk(self, x: np.ndarray) -> None:
        """Initialize from bulk complex samples."""
        n_samples = min(len(x), self.max_samples)
        self.buffer_i[:n_samples] = x[:n_samples].real
        self.buffer_q[:n_samples] = x[:n_samples].imag
        self.buffer_pos = n_samples % self.max_samples
        if n_samples >= self.max_samples:
            self.buffer_filled = True
        self._run_music()
        
    def _get_buffer_data(self) -> np.ndarray:
        """Extract current buffer data as complex array in correct order."""
        if self.buffer_filled:
            n_samples = self.max_samples
            # Reorder circular buffer
            idx = np.arange(n_samples)
            idx = (idx + self.buffer_pos) % n_samples
            data = self.buffer_i[idx] + 1j * self.buffer_q[idx]
        else:
            n_samples = self.buffer_pos
            data = self.buffer_i[:n_samples] + 1j * self.buffer_q[:n_samples]
            
        return data
    
    def _form_snapshot_matrix(self, data: np.ndarray) -> np.ndarray:
        """Vectorized: form forward/backward snapshots via sliding windows."""


        L = self.subarray_length
        N = len(data) - L + 1
        if N <= 0:
            return np.zeros((L, 1), dtype=complex)

        # sliding_window_view returns shape (N, L)
        windows = sliding_window_view(data, window_shape=L)
        # Forward: transpose to (L, N)
        X_fwd = windows.T
        # Backward: reverse each window in the last axis, then transpose
        X_bwd = np.conj(windows[:, ::-1].T)

        # Stack them once (shape becomes (L, 2N))
        return np.concatenate((X_fwd, X_bwd), axis=1)
    
    def _root_music(self, R: np.ndarray) -> np.ndarray:
        """
        Perform root-MUSIC to find frequencies.
        
        Args:
            R: Covariance matrix
            
        Returns:
            Array of estimated frequencies in Hz
        """
        L = R.shape[0]
        
        # Eigendecomposition
        eigvals, eigvecs = eigh(R)
        
        # Sort eigenvalues/vectors in descending order
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Noise subspace (smallest eigenvalues)
        noise_dim = L - self.M
        G = eigvecs[:, self.M:]  # Noise eigenvectors
        
        # Form polynomial coefficients from noise subspace
        # P(z) = sum_i |v_i^H * a(z)|^2 where v_i are noise eigenvectors
        # and a(z) = [1, z^-1, ..., z^-(L-1)]^T
        
        # Compute C = G * G^H
        C = G @ G.conj().T
        
        # Extract polynomial coefficients from first row/column of C
        # Since C is Hermitian Toeplitz, we can form the polynomial
        poly_coeffs = np.zeros(2*L-1, dtype=complex)
        for k in range(L):
            poly_coeffs[L-1-k] = C[0, k]  # c_{-k}
            if k > 0:
                poly_coeffs[L-1+k] = np.conj(C[0, k])  # c_k = conj(c_{-k})
        
        # Find roots of the polynomial
        roots = np.roots(poly_coeffs[::-1])  # numpy wants highest degree first
        
        # Select roots close to unit circle (signal roots)
        unit_circle_roots = roots[np.abs(np.abs(roots) - 1.0) < 0.1]
        
        # Sort by angle and take M roots closest to unit circle
        if len(unit_circle_roots) >= self.M:
            distances = np.abs(np.abs(unit_circle_roots) - 1.0)
            idx = distances.argsort()[:self.M]
            signal_roots = unit_circle_roots[idx]
        else:
            # Fallback: take M roots closest to unit circle
            distances = np.abs(np.abs(roots) - 1.0)
            idx = distances.argsort()[:self.M]
            signal_roots = roots[idx]
        
        # Convert roots to frequencies
        angles = np.angle(signal_roots)
        freqs_hz = angles * self.fs / (2 * np.pi)
        
        # Ensure positive frequencies
        freqs_hz = np.abs(freqs_hz)
        
        return np.sort(freqs_hz)
    
    def _estimate_amplitudes_phases(self, data: np.ndarray, freqs_hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate amplitudes and phases given frequencies using least squares."""
        N = len(data)
        t = np.arange(N) * self.dt
        
        # Build design matrix
        A = np.zeros((N, 2*self.M))
        for k in range(self.M):
            phase = 2 * np.pi * freqs_hz[k] * t
            A[:, 2*k] = np.cos(phase)
            A[:, 2*k+1] = np.sin(phase)
        
        # Least squares fit
        # data = A @ [a1*cos, a1*sin, a2*cos, a2*sin, ...]
        x = np.linalg.lstsq(A, data.real, rcond=None)[0]
        y = np.linalg.lstsq(A, data.imag, rcond=None)[0]
        
        # Extract amplitudes and phases
        amplitudes = np.zeros(self.M)
        phases = np.zeros(self.M)
        
        for k in range(self.M):
            a_real = x[2*k] + 1j * y[2*k]
            a_imag = x[2*k+1] + 1j * y[2*k+1]
            # Complex amplitude is a_real * exp(j*0) + a_imag * exp(j*pi/2)
            complex_amp = a_real + 1j * a_imag
            amplitudes[k] = np.abs(complex_amp)
            phases[k] = np.angle(complex_amp)
            
        return amplitudes, phases
    
    def _estimate_frequency_variance(self, R: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        """Estimate frequency estimation variance using perturbation analysis."""
        L = R.shape[0]
        
        # Simple estimate based on SNR and effective snapshot count
        # This is a simplified version - full MUSIC CRB is more complex
        N_eff = 2 * (len(self._get_buffer_data()) - L + 1)  # Forward-backward
        
        # Estimate noise power from smallest eigenvalue
        eigvals = eigh(R, eigvals_only=True)
        noise_power = np.mean(eigvals[:L-self.M])
        
        # Frequency variance approximation (simplified)
        # Actual CRB depends on array manifold derivatives
        freq_vars = np.ones(self.M) * (noise_power * self.fs**2) / (N_eff * L)
        
        # Scale by amplitude (stronger signals have lower variance)
        if hasattr(self, 'amplitudes') and np.any(self.amplitudes > 0):
            snr_factors = self.amplitudes**2 / (noise_power + 1e-10)
            freq_vars /= (snr_factors + 1)
            
        return freq_vars
    
    def _run_music(self) -> None:
        """Run root-MUSIC on current buffer."""
        data = self._get_buffer_data()
        
        if len(data) < self.subarray_length:
            return
        
        # Form snapshot matrix with forward-backward averaging
        X = self._form_snapshot_matrix(data)
        
        # Compute covariance matrix
        R = (X @ X.conj().T) / X.shape[1]
        
        # Apply diagonal loading for numerical stability
        R += np.eye(R.shape[0]) * 1e-10 * np.trace(R)
        
        # Root-MUSIC frequency estimation
        self.freqs = self._root_music(R)
        
        # Estimate amplitudes and phases
        self.amplitudes, self.phases = self._estimate_amplitudes_phases(data, self.freqs)
        
        # Estimate frequency variances
        self.freq_vars = self._estimate_frequency_variance(R, self.freqs)
        
        self.initialized = True
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state with parameters sorted by frequency."""
        if not self.initialized:
            return {
                'amplitudes': np.zeros(self.M),
                'phases': np.zeros(self.M),
                'freqs': np.zeros(self.M),
                'cov': np.ones(3 * self.M) * 1000.0
            }
        
        # Already sorted by frequency from root-MUSIC
        # Build covariance diagonal in same format as ToneLM
        cov_diag = np.zeros(3 * self.M)
        for k in range(self.M):
            # Amplitude variance (simplified - proportional to amplitude)
            cov_diag[2*k] = (0.1 * self.amplitudes[k])**2  # Real part variance
            cov_diag[2*k + 1] = (0.1 * self.amplitudes[k])**2  # Imag part variance
            # Frequency variance
            cov_diag[2*self.M + k] = self.freq_vars[k]
        
        return {
            'amplitudes': self.amplitudes.copy(),
            'phases': self.phases.copy(),
            'freqs': self.freqs.copy(),
            'cov': cov_diag
        }
