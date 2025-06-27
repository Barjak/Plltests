import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import matplotlib.cm as cm




class DualEKFAnalyzer:
    def __init__(self, fs_baseband=960.0):
        self.fs = fs_baseband
        self.dt = 1.0 / fs_baseband

    def dual_ekf_tracking(self, signal_data, f1_init, f2_init, 
                          Q=None, R=None, P0=None,
                          track_amplitude=True,
                          min_separation_hz=0.006,  # 6 mHz minimum
                          separation_buffer_hz=0.001,  # 1 mHz buffer above minimum
                          separation_weight=0.01):  # Weight for pseudo-measurement
        """
        Dual-tone tracking using Extended Kalman Filter with direct frequency parameterization
        and minimum separation enforcement
        
        State vector: x = [phi1, phi2, f1, f2, A1, A2]
        where:
        - phi1, phi2 = phases
        - f1, f2 = frequencies in Hz (with f1 < f2 enforced)
        - A1, A2 = amplitudes
        """
        n_samples = len(signal_data)
        dt = self.dt
        
        # Ensure initial ordering
        if f1_init > f2_init:
            f1_init, f2_init = f2_init, f1_init
        
        # State transition matrix
        F = np.array([[1, 0, 2*np.pi*dt, 0, 0, 0],  # phi1 += 2*pi*f1*dt
                      [0, 1, 0, 2*np.pi*dt, 0, 0],  # phi2 += 2*pi*f2*dt
                      [0, 0, 1, 0, 0, 0],            # f1
                      [0, 0, 0, 1, 0, 0],            # f2
                      [0, 0, 0, 0, 1, 0],            # A1
                      [0, 0, 0, 0, 0, 1]])           # A2
        
        # Initialize state
        x = np.array([0.0,       # phi1
                      0.0,       # phi2
                      f1_init,   # f1
                      f2_init,   # f2
                      1.0,       # A1
                      1.0])      # A2
        
        # Process noise covariance
        if Q is None:
            sigma_phi = 1e-7      # Phase noise (very small)
            sigma_f = 1e-4        # Frequency drift
            sigma_A = 1e-6        # Amplitude drift
            Q = np.diag([sigma_phi**2, sigma_phi**2, 
                        sigma_f**2, sigma_f**2,
                        sigma_A**2, sigma_A**2])
        
        # Initial state covariance
        if P0 is None:
            P0 = np.diag([0.1,    # phi1 uncertainty
                          0.1,    # phi2 uncertainty
                          0.05,   # f1 uncertainty (Hz)
                          0.05,   # f2 uncertainty (Hz)
                          0.01,   # A1 uncertainty
                          0.01])  # A2 uncertainty
        
        P = P0.copy()
        
        # Measurement noise covariance
        if R is None:
            R = 3.0
        
        # Storage for analysis
        history = {
            'x': [x.copy()],
            'P': [P.copy()],
            'y_pred': [],
            'innov': [],
            'freq1': [f1_init],
            'freq2': [f2_init],
            'A1': [x[4]],
            'A2': [x[5]],
            'phase1': [x[0]],
            'phase2': [x[1]],
            'separation': [f2_init - f1_init],
            'error': [],
            'K': [],
            'swaps': [],
            'pseudo_applied': []  # Track when pseudo-measurement is applied
        }
        
        # Main EKF loop
        for k, y in enumerate(signal_data):
            # ---- Predict ----
            x = F @ x
            P = F @ P @ F.T + Q
            
            # Wrap phases to [-pi, pi]
            x[0] = np.angle(np.exp(1j * x[0]))
            x[1] = np.angle(np.exp(1j * x[1]))
            
            # ---- Measurement prediction ----
            phi1, phi2, f1, f2 = x[0], x[1], x[2], x[3]
            A1, A2 = x[4], x[5]
            
            # Complex measurement prediction
            y_hat = A1 * np.exp(1j * phi1) + A2 * np.exp(1j * phi2)
            
            # ---- Compute Jacobian H ----
            H = np.zeros((2, 6))
            
            # Derivatives w.r.t phi1
            H[0, 0] = -A1 * np.sin(phi1)  # dRe/dphi1
            H[1, 0] = A1 * np.cos(phi1)   # dIm/dphi1
            
            # Derivatives w.r.t phi2
            H[0, 1] = -A2 * np.sin(phi2)  # dRe/dphi2
            H[1, 1] = A2 * np.cos(phi2)   # dIm/dphi2
            
            # Derivatives w.r.t f1, f2 are zero for instantaneous measurement
            
            if track_amplitude:
                # Derivatives w.r.t A1
                H[0, 4] = np.cos(phi1)  # dRe/dA1
                H[1, 4] = np.sin(phi1)  # dIm/dA1
                
                # Derivatives w.r.t A2
                H[0, 5] = np.cos(phi2)  # dRe/dA2
                H[1, 5] = np.sin(phi2)  # dIm/dA2
            
            # ---- Innovation ----
            z = np.array([np.real(y), np.imag(y)])
            z_hat = np.array([np.real(y_hat), np.imag(y_hat)])
            innov = z - z_hat
            
            # ---- Kalman gain ----
            S = H @ P @ H.T + R * np.eye(2)
            K = P @ H.T @ np.linalg.inv(S)
            
            # ---- Update ----
            x = x + K @ innov
            P = (np.eye(6) - K @ H) @ P
            
            # ---- Pseudo-measurement for minimum separation ----
            current_sep = x[3] - x[2]
            
            # Apply pseudo-measurement if below or close to minimum
            if current_sep < min_separation_hz + separation_buffer_hz:
                # Target separation
                target_sep = min_separation_hz + separation_buffer_hz
                
                # Pseudo-measurement update on the separation f2 - f1
                # H_pseudo = [0, 0, -1, 1, 0, 0] to observe f2 - f1
                H_pseudo = np.zeros((1, 6))
                H_pseudo[0, 2] = -1  # df1
                H_pseudo[0, 3] = 1   # df2
                
                z_pseudo = np.array([target_sep])
                z_hat_pseudo = np.array([current_sep])
                
                R_pseudo = 1.0 / separation_weight
                
                S_pseudo = H_pseudo @ P @ H_pseudo.T + R_pseudo
                K_pseudo = P @ H_pseudo.T / S_pseudo
                
                x = x + K_pseudo.flatten() * (z_pseudo - z_hat_pseudo)
                P = (np.eye(6) - np.outer(K_pseudo, H_pseudo)) @ P
                
                history['pseudo_applied'].append(True)
            else:
                history['pseudo_applied'].append(False)
            
            # Ensure ordering is maintained (f1 < f2)
            if x[2] > x[3]:
                x[2], x[3] = x[3], x[2]
                x[0], x[1] = x[1], x[0]
                x[4], x[5] = x[5], x[4]
                history['swaps'].append(True)
            else:
                history['swaps'].append(False)
            
            # Ensure positive amplitudes
            if track_amplitude:
                x[4] = max(0.1, x[4])
                x[5] = max(0.1, x[5])
            
            # ---- Store results ----
            history['x'].append(x.copy())
            history['P'].append(P.copy())
            history['y_pred'].append(y_hat)
            history['innov'].append(innov)
            history['freq1'].append(x[2])
            history['freq2'].append(x[3])
            history['A1'].append(x[4])
            history['A2'].append(x[5])
            history['phase1'].append(x[0])
            history['phase2'].append(x[1])
            history['separation'].append(x[3] - x[2])
            history['error'].append(np.abs(y - y_hat))
            history['K'].append(K.copy())
        
        # Convert lists to arrays
        for key in history:
            if key not in ['x', 'P', 'K']:
                history[key] = np.array(history[key])
        
        # Compute final estimates (average over last quarter of samples)
        converged_f1 = np.mean(history['freq1'][-n_samples//4:])
        converged_f2 = np.mean(history['freq2'][-n_samples//4:])
        converged_A1 = np.mean(history['A1'][-n_samples//4:])
        converged_A2 = np.mean(history['A2'][-n_samples//4:])
        
        return {
            'f1': converged_f1,
            'f2': converged_f2,
            'beat': converged_f2 - converged_f1,
            'A1': converged_A1,
            'A2': converged_A2,
            'history': history,
            'total_swaps': np.sum(history['swaps'])
        }
        
