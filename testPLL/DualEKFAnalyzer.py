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
                          min_separation_hz=0.003,  # 3 mHz minimum
                          separation_weight=0.004,
                          enforce_ordering=True):  # Keep for compatibility
        """
        Dual-tone tracking using Extended Kalman Filter with reparameterized state
        
        State vector: x = [phi1, phi2, w_sum, w_diff, A1, A2]
        where:
        - phi1, phi2 = phases
        - w_sum = (w1 + w2) / 2 (average angular frequency)
        - w_diff = (w2 - w1) / 2 (half frequency difference)
        - A1, A2 = amplitudes
        
        This gives: w1 = w_sum - w_diff, w2 = w_sum + w_diff
        """
        n_samples = len(signal_data)
        dt = self.dt
        min_separation_rad = 2 * np.pi * min_separation_hz
        
        # Ensure initial ordering
        if f1_init > f2_init:
            f1_init, f2_init = f2_init, f1_init
        
        # Convert to angular frequencies
        w1_init = 2 * np.pi * f1_init
        w2_init = 2 * np.pi * f2_init
        w_sum_init = (w1_init + w2_init) / 2
        w_diff_init = (w2_init - w1_init) / 2
        
        # State transition matrix for reparameterized state
        # phi1(k+1) = phi1(k) + w1*dt = phi1(k) + (w_sum - w_diff)*dt
        # phi2(k+1) = phi2(k) + w2*dt = phi2(k) + (w_sum + w_diff)*dt
        F = np.array([[1, 0, dt, -dt, 0, 0],  # phi1
                      [0, 1, dt,  dt, 0, 0],  # phi2
                      [0, 0,  1,   0, 0, 0],  # w_sum
                      [0, 0,  0,   1, 0, 0],  # w_diff
                      [0, 0,  0,   0, 1, 0],  # A1
                      [0, 0,  0,   0, 0, 1]]) # A2
        
        # Initialize state
        x = np.array([0.0,           # phi1
                      0.0,           # phi2
                      w_sum_init,    # w_sum
                      w_diff_init,   # w_diff
                      1.0,           # A1
                      0.7])          # A2
        
        # Process noise covariance - diagonal with different noise for sum/diff
        if Q is None:
            sigma_phi = 1e-7      # Phase noise (very small)
            sigma_w_sum = 1e-4    # Average frequency drift (allow more)
            sigma_w_diff = 1e-5   # Frequency difference drift (restrict more)
            sigma_A = 1e-6        # Amplitude drift
            Q = np.diag([sigma_phi**2, sigma_phi**2, 
                         sigma_w_sum**2, sigma_w_diff**2,
                         sigma_A**2, sigma_A**2])
        
        # Measurement noise covariance
        if R is None:
            R = 3.0
        
        # Initial state covariance
        if P0 is None:
            P0 = np.diag([0.1,    # phi1 uncertainty
                          0.1,    # phi2 uncertainty
                          0.1,    # w_sum uncertainty (rad/s)
                          0.05,   # w_diff uncertainty (rad/s) - smaller
                          0.01,   # A1 uncertainty
                          0.01])  # A2 uncertainty
        
        P = P0.copy()
        
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
            'separation': [abs(f2_init - f1_init)],
            'error': [],
            'K': [],
            'swaps': []  # Keep for compatibility, always False now
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
            phi1, phi2, w_sum, w_diff, A1, A2 = x
            
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
            
            # Derivatives w.r.t w_sum and w_diff (zero for instantaneous measurement)
            H[0, 2] = 0
            H[1, 2] = 0
            H[0, 3] = 0
            H[1, 3] = 0
            
            if track_amplitude:
                # Derivatives w.r.t A1
                H[0, 4] = np.cos(phi1)  # dRe/dA1
                H[1, 4] = np.sin(phi1)  # dIm/dA1
                
                # Derivatives w.r.t A2
                H[0, 5] = np.cos(phi2)  # dRe/dA2
                H[1, 5] = np.sin(phi2)  # dIm/dA2
            else:
                # Don't update amplitudes
                H[0, 4] = 0
                H[1, 4] = 0
                H[0, 5] = 0
                H[1, 5] = 0
            
            # ---- Innovation ----
            z = np.array([np.real(y), np.imag(y)])
            z_hat = np.array([np.real(y_hat), np.imag(y_hat)])
            innov = z - z_hat
            
            # ---- Kalman gain ----
            S = H @ P @ H.T + R * np.eye(2)
            K = P @ H.T @ np.linalg.inv(S)
            
            # ---- Update ----
            x = x + K @ innov
            P = (np.eye(len(x)) - K @ H) @ P
            
            # ---- Pseudo-measurement update for minimum separation ----
            # We want to enforce w_diff >= min_separation_rad / 2
            min_w_diff = min_separation_rad / 2
            
            if x[3] < min_w_diff:
                # Create pseudo-measurement that observes w_diff
                H_pseudo = np.zeros((1, 6))
                H_pseudo[0, 3] = 1  # Observe w_diff directly
                
                # Pseudo-measurement value (push w_diff to minimum)
                z_pseudo = np.array([min_w_diff])
                z_hat_pseudo = np.array([x[3]])
                
                # Pseudo-measurement noise (controls strength of constraint)
                R_pseudo = 1.0 / separation_weight  # Smaller R_pseudo = stronger constraint
                
                # Kalman gain for pseudo-measurement
                S_pseudo = H_pseudo @ P @ H_pseudo.T + R_pseudo
                K_pseudo = P @ H_pseudo.T / S_pseudo
                
                # Update state and covariance
                x = x + K_pseudo.flatten() * (z_pseudo - z_hat_pseudo)
                P = (np.eye(len(x)) - np.outer(K_pseudo, H_pseudo)) @ P
            
            # Ensure positive amplitudes
            if track_amplitude:
                x[4] = max(0.1, x[4])
                x[5] = max(0.1, x[5])
            
            # ---- Convert back to individual frequencies for storage ----
            w1 = x[2] - x[3]  # w_sum - w_diff
            w2 = x[2] + x[3]  # w_sum + w_diff
            
            # ---- Store results ----
            history['x'].append(x.copy())
            history['P'].append(P.copy())
            history['y_pred'].append(y_hat)
            history['innov'].append(innov)
            history['freq1'].append(w1 / (2 * np.pi))  # Convert to Hz
            history['freq2'].append(w2 / (2 * np.pi))  # Convert to Hz
            history['A1'].append(x[4])
            history['A2'].append(x[5])
            history['phase1'].append(x[0])
            history['phase2'].append(x[1])
            history['separation'].append(2 * x[3] / (2 * np.pi))  # 2*w_diff = w2-w1
            history['error'].append(np.abs(y - y_hat))
            history['K'].append(K.copy())
            history['swaps'].append(False)  # No swaps with this parameterization
        
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
            'total_swaps': 0  # Always 0 with this parameterization
        }
        
    def dual_pll_with_tracking(self, signal_data, f1_init, f2_init, 
                              Q=None, R=None, P0=None,
                              track_amplitude=True,
                              min_separation_hz=0.003,  # 3 mHz minimum
                              separation_weight=0.004,
                              enforce_ordering=True):  # New parameter
        """
        Dual PLL with detailed tracking of all parameters
        Backwards compatible with dual_ekf_tracking signature
        
        Parameters match EKF method for compatibility:
        - Q, R, P0: Not used in PLL but accepted for compatibility
        - min_separation_hz: Minimum frequency separation in Hz
        - separation_weight: Maps to frequency regularization strength
        - enforce_ordering: If True, enforces f1 < f2 constraint
        """
        n_samples = len(signal_data)
        
        # Ensure initial ordering if enforce_ordering is True
        if enforce_ordering and f1_init > f2_init:
            f1_init, f2_init = f2_init, f1_init
        
        # Initialize
        phase1, phase2 = 0.0, 0.0
        freq1, freq2 = f1_init, f2_init
        
        # Amplitude tracking
        if track_amplitude:
            A1, A2 = 1.0, 0.7  # Initial guess matching EKF
        else:
            A1, A2 = 1.0, 0.7  # Fixed
        
        # PLL parameters
        loop_bw = 0.5  # Hz
        damping = 1.0
        theta = 2 * np.pi * loop_bw / self.fs
        d = 1 + 2 * damping * theta + theta**2
        g1 = 4 * damping * theta / d
        g2 = 4 * theta**2 / d
        
        # Map EKF parameters to PLL equivalents
        min_separation = min_separation_hz  # Use same name convention
        freq_regularization = separation_weight * 25  # Scale factor for similar behavior
        amplitude_regularization = 0.1
        
        # Storage for analysis - match EKF history structure
        history = {
            'x': [],  # Will store state vector for compatibility
            'P': [],  # Dummy for compatibility
            'y_pred': [],
            'innov': [],
            'freq1': [freq1],
            'freq2': [freq2],
            'A1': [A1],
            'A2': [A2],
            'phase1': [phase1],
            'phase2': [phase2],
            'separation': [abs(freq2 - freq1)],
            'error': [],
            'K': [],  # Dummy for compatibility
            'swaps': []  # Track when swaps occur
        }
        
        # Additional PLL-specific tracking
        history['phase_error1'] = []
        history['phase_error2'] = []
        history['reg_force'] = []
        
        # Loop filter integrals
        phase_error1_integral = 0
        phase_error2_integral = 0
        
        # Store initial state vector for compatibility
        x = np.array([phase1, 2*np.pi*freq1, phase2, 2*np.pi*freq2, A1, A2])
        history['x'].append(x.copy())
        history['P'].append(np.eye(6))  # Dummy
        history['K'].append(np.zeros((6, 2)))  # Dummy
        
        for i, sample in enumerate(signal_data):
            # Generate NCOs
            nco1 = np.exp(1j * phase1)
            nco2 = np.exp(1j * phase2)
            
            # Current signal estimate
            if track_amplitude:
                signal_est = A1 * nco1 + A2 * nco2
            else:
                # Simple correlation for fixed amplitudes
                a1 = sample * np.conj(nco1)
                a2 = sample * np.conj(nco2)
                signal_est = a1 * nco1 + a2 * nco2
                # Use correlation magnitudes as effective amplitudes
                A1_eff = np.abs(a1)
                A2_eff = np.abs(a2)
            
            # Store prediction for EKF compatibility
            history['y_pred'].append(signal_est)
            
            # Innovation (error)
            error = sample - signal_est
            innov = np.array([np.real(error), np.imag(error)])
            history['innov'].append(innov)
            history['error'].append(np.abs(error))
            
            # Phase errors
            if track_amplitude:
                phase_error1 = np.real(np.conj(error) * 1j * A1 * nco1)
                phase_error2 = np.real(np.conj(error) * 1j * A2 * nco2)
            else:
                phase_error1 = np.real(np.conj(error) * 1j * A1_eff * nco1)
                phase_error2 = np.real(np.conj(error) * 1j * A2_eff * nco2)
            
            history['phase_error1'].append(phase_error1)
            history['phase_error2'].append(phase_error2)
            
            # Regularization force for minimum separation
            separation = abs(freq2 - freq1)
            reg_force = 0
            if separation < min_separation:
                reg_force = freq_regularization * (min_separation - separation) / min_separation
                if enforce_ordering:
                    # Always push f1 down and f2 up to maintain ordering
                    phase_error1 -= reg_force
                    phase_error2 += reg_force
                else:
                    # Original symmetric push
                    if freq2 > freq1:
                        phase_error2 += reg_force
                        phase_error1 -= reg_force
                    else:
                        phase_error2 -= reg_force
                        phase_error1 += reg_force
            
            history['reg_force'].append(reg_force)
            
            # Update frequencies with loop filter
            phase_error1_integral += phase_error1
            phase_error2_integral += phase_error2
            
            freq1_new = f1_init + g1 * phase_error1 + g2 * phase_error1_integral
            freq2_new = f2_init + g1 * phase_error2 + g2 * phase_error2_integral
            
            # Check for ordering violation and handle swaps
            swap_occurred = False
            if enforce_ordering and freq1_new > freq2_new:
                # Swap frequencies
                freq1_new, freq2_new = freq2_new, freq1_new
                # Swap phases
                phase1, phase2 = phase2, phase1
                # Swap amplitudes
                if track_amplitude:
                    A1, A2 = A2, A1
                # Swap integral states
                phase_error1_integral, phase_error2_integral = phase_error2_integral, phase_error1_integral
                swap_occurred = True
            
            freq1, freq2 = freq1_new, freq2_new
            history['swaps'].append(swap_occurred)
            
            # Update phases
            phase1 += 2 * np.pi * freq1 / self.fs
            phase2 += 2 * np.pi * freq2 / self.fs
            phase1 = np.angle(np.exp(1j * phase1))
            phase2 = np.angle(np.exp(1j * phase2))
            
            # Update amplitudes if tracking
            if track_amplitude:
                # Gradient descent on amplitudes with regularization
                learning_rate = 0.01
                
                dA1 = -2 * np.real(np.conj(error) * nco1)
                dA2 = -2 * np.real(np.conj(error) * nco2)
                
                # Add regularization gradient
                dA1 += amplitude_regularization * (A1 - 1.0)
                dA2 += amplitude_regularization * (A2 - 1.0)
                
                A1 -= learning_rate * dA1
                A2 -= learning_rate * dA2
                
                # Constrain to positive
                A1 = max(0.1, A1)
                A2 = max(0.1, A2)
            
            # Store history
            history['freq1'].append(freq1)
            history['freq2'].append(freq2)
            history['A1'].append(A1)
            history['A2'].append(A2)
            history['phase1'].append(phase1)
            history['phase2'].append(phase2)
            history['separation'].append(abs(freq2 - freq1))
            
            # Store state vector for EKF compatibility
            x = np.array([phase1, 2*np.pi*freq1, phase2, 2*np.pi*freq2, A1, A2])
            history['x'].append(x.copy())
            history['P'].append(np.eye(6))  # Dummy
            history['K'].append(np.zeros((6, 2)))  # Dummy
        
        # Convert lists to arrays (except x, P, K as per EKF)
        for key in history:
            if key not in ['x', 'P', 'K']:
                history[key] = np.array(history[key])
        
        # Compute final estimates (average over last quarter of samples)
        converged_f1 = np.mean(history['freq1'][-n_samples//4:])
        converged_f2 = np.mean(history['freq2'][-n_samples//4:])
        converged_A1 = np.mean(history['A1'][-n_samples//4:])
        converged_A2 = np.mean(history['A2'][-n_samples//4:])
        
        # Return structure matching EKF output exactly
        return {
            'f1': converged_f1,
            'f2': converged_f2,
            'beat': converged_f2 - converged_f1,
            'A1': converged_A1,
            'A2': converged_A2,
            'history': history,
            'total_swaps': np.sum(history['swaps'])
        }
    
    def run_from_multiple_initializations(self, signal_data, f1_true, f2_true, 
                                    Q=None, R=None, enforce_ordering=True):
        """Test EKF from various starting points"""
        
        # Define test cases
        test_cases = [
            # (f1_init, f2_init, label)
            (f1_true, f2_true, "Truth"),
            (f1_true - 0.010, f2_true + 0.010, "Far from truth"),
            (f2_true, f1_true, "Swapped"),
            (f1_true, f1_true + 0.001, "Very close"),
            (f1_true, f1_true + 0.020, "Far apart"),
            (f1_true - 0.005, f2_true + 0.005, "Symmetric offset"),
            (f1_true + 0.002, f2_true - 0.002, "Inward offset"),
#             (5.5, 5.7, "Random far"),
        ]
        
        results = []
        for f1_init, f2_init, label in test_cases:
            result = self.dual_ekf_tracking(signal_data, f1_init, f2_init, Q=Q, R=R)
            result['label'] = label
            result['f1_init'] = f1_init
            result['f2_init'] = f2_init
            results.append(result)
        
        return results
    
    def compute_error_landscape(self, signal_data, f1_range, f2_range, true_f1, true_f2):
        """Compute the error landscape for visualization"""
        n_points = 50
        f1_grid = np.linspace(f1_range[0], f1_range[1], n_points)
        f2_grid = np.linspace(f2_range[0], f2_range[1], n_points)
        
        error_landscape = np.zeros((n_points, n_points))
        
        # Pre-compute true signal for efficiency
        t = np.arange(len(signal_data)) / self.fs
        
        for i, f1 in enumerate(f1_grid):
            for j, f2 in enumerate(f2_grid):
                # Compute error for this (f1, f2) pair
                s1 = np.exp(1j * 2 * np.pi * f1 * t)
                s2 = 0.7 * np.exp(1j * 2 * np.pi * f2 * t)
                estimate = s1 + s2
                
                error = np.mean(np.abs(signal_data - estimate)**2)
                error_landscape[j, i] = error  # Note: j, i for proper orientation
        
        return f1_grid, f2_grid, error_landscape
    def visualize_ekf_analysis(self, results, signal_data, f1_true, f2_true):
        """Comprehensive visualization of EKF behavior with reparameterized state"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3, 
                              width_ratios=[2, 1, 1])
        
        # 1. 2D Phase Space Trajectories (large, left side)
        ax1 = fig.add_subplot(gs[:, 0])
        
        # Compute and plot error landscape
        f1_range = (f1_true - 0.015, f1_true + 0.015)
        f2_range = (f2_true - 0.015, f2_true + 0.015)
        f1_grid, f2_grid, error_landscape = self.compute_error_landscape(
            signal_data, f1_range, f2_range, f1_true, f2_true
        )
        
        # Plot error landscape with more contour lines
        contour = ax1.contourf(f1_grid, f2_grid, np.log10(error_landscape + 1e-10), 
                               levels=40, cmap='viridis', alpha=0.6)
        
        # Plot trajectories
        colors = cm.rainbow(np.linspace(0, 1, len(results)))
        for result, color in zip(results, colors):
            history = result['history']
            ax1.plot(history['freq1'], history['freq2'], '-', color=color, 
                    linewidth=1.5, alpha=0.8, label=result['label'])
            # Start point
            ax1.plot(result['f1_init'], result['f2_init'], 'o', 
                    color=color, markersize=12, markeredgecolor='black', 
                    markeredgewidth=1.5)
            # End point
            ax1.plot(history['freq1'][-1], history['freq2'][-1], 's', 
                    color=color, markersize=12, markeredgecolor='black',
                    markeredgewidth=1.5)
        
        # True values
        ax1.plot(f1_true, f2_true, '*', color='red', markersize=25, 
                markeredgecolor='black', markeredgewidth=2, label='True')
        
        # Diagonal line (f1 = f2)
        diag_line = np.array([min(f1_range[0], f2_range[0]), 
                             max(f1_range[1], f2_range[1])])
        ax1.plot(diag_line, diag_line, 'k--', alpha=0.5, linewidth=2, label='f1=f2')
        
        # Add minimum separation line (3 mHz below diagonal)
        min_sep_hz = 0.003
        ax1.plot(diag_line, diag_line + min_sep_hz, 'k:', alpha=0.5, linewidth=1.5, 
                label=f'min sep ({min_sep_hz*1000:.0f} mHz)')
        
        ax1.set_xlabel('f1 (Hz)', fontsize=12)
        ax1.set_ylabel('f2 (Hz)', fontsize=12)
        ax1.set_title('2D Frequency Space Trajectories (Reparameterized EKF)', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Add colorbar for error landscape
        cbar = plt.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('log10(MSE)', fontsize=10)
        
        # 2. Separation Evolution (top right)
        ax2 = fig.add_subplot(gs[0, 1:])
        
        for result, color in zip(results, colors):
            history = result['history']
            samples = np.arange(len(history['separation']))
            ax2.plot(samples, history['separation'] * 1000, '-', color=color, 
                    linewidth=2, alpha=0.8)
        
        ax2.axhline(6.0, color='red', linestyle='--', linewidth=2, label='True separation')
        ax2.axhline(3.0, color='black', linestyle=':', linewidth=1.5, label='Min separation')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('|f2 - f1| (mHz)')
        ax2.set_title('Frequency Separation Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, max(20, ax2.get_ylim()[1])])
        ax2.legend()
        
        # 3. Uncertainty Evolution (second from top right)
        ax3 = fig.add_subplot(gs[1, 1:])
        
        # Plot frequency uncertainty for the "Truth" case
        truth_result = results[0]
        P_history = truth_result['history']['P']
        
        # Extract standard deviations for frequencies from reparameterized state
        # State: [phi1, phi2, w_sum, w_diff, A1, A2]
        # w1 = w_sum - w_diff, w2 = w_sum + w_diff
        std_w_sum = []
        std_w_diff = []
        std_f1 = []
        std_f2 = []
        
        for P in P_history:
            # Extract variances
            var_w_sum = P[2, 2]
            var_w_diff = P[3, 3]
            cov_sum_diff = P[2, 3]
            
            # Compute frequency variances using error propagation
            # var(w1) = var(w_sum - w_diff) = var(w_sum) + var(w_diff) - 2*cov(w_sum, w_diff)
            # var(w2) = var(w_sum + w_diff) = var(w_sum) + var(w_diff) + 2*cov(w_sum, w_diff)
            var_w1 = var_w_sum + var_w_diff - 2 * cov_sum_diff
            var_w2 = var_w_sum + var_w_diff + 2 * cov_sum_diff
            
            # Convert to frequency standard deviations in mHz
            std_f1.append(np.sqrt(max(0, var_w1)) / (2 * np.pi) * 1000)
            std_f2.append(np.sqrt(max(0, var_w2)) / (2 * np.pi) * 1000)
            std_w_sum.append(np.sqrt(var_w_sum) / (2 * np.pi) * 1000)
            std_w_diff.append(np.sqrt(var_w_diff) / (2 * np.pi) * 1000)
        
        samples = np.arange(len(std_f1))
        ax3.plot(samples, std_f1, 'b-', linewidth=2, label='f1 uncertainty')
        ax3.plot(samples, std_f2, 'r-', linewidth=2, label='f2 uncertainty')
        ax3.plot(samples, std_w_sum, 'g--', linewidth=1.5, alpha=0.7, label='f_avg uncertainty')
        ax3.plot(samples, std_w_diff, 'm--', linewidth=1.5, alpha=0.7, label='f_diff/2 uncertainty')
        
        ax3.set_xlabel('Sample')
        ax3.set_ylabel('Frequency Uncertainty (mHz)')
        ax3.set_title('EKF Uncertainty Evolution (Reparameterized State)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Convergence Comparison (third from top right)
        ax4 = fig.add_subplot(gs[2, 1:])
        
        # Bar chart of final errors
        labels = [r['label'] for r in results]
        f1_errors = [(r['f1'] - f1_true) * 1000 for r in results]
        f2_errors = [(r['f2'] - f2_true) * 1000 for r in results]
        beat_errors = [(r['beat'] - (f2_true - f1_true)) * 1000 for r in results]
        
        x = np.arange(len(labels))
        width = 0.25
        
        ax4.bar(x - width, f1_errors, width, label='f1 error', alpha=0.8)
        ax4.bar(x, f2_errors, width, label='f2 error', alpha=0.8)
        ax4.bar(x + width, beat_errors, width, label='beat error', alpha=0.8)
        
        ax4.set_xlabel('Initialization')
        ax4.set_ylabel('Error (mHz)')
        ax4.set_title('Final Estimation Errors')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(0, color='black', linewidth=0.6)
        
        # 5. Signal Reconstruction (bottom right)
        ax5 = fig.add_subplot(gs[3, 1:])
        
        # Use the "Truth" initialization case
        history = truth_result['history']
        
        # Reconstruct signals at a few time points
        t = np.arange(len(signal_data)) / self.fs
        time_points = [0, len(signal_data)//4, len(signal_data)//2, -1]
        
        for i, tp in enumerate(time_points):
            alpha = 0.3 + 0.7 * i / len(time_points)
            if tp == -1:
                label = 'Final'
            else:
                label = f'Sample {tp}'
                
            # Reconstruct
            s1 = history['A1'][tp] * np.exp(1j * 2 * np.pi * history['freq1'][tp] * t)
            s2 = history['A2'][tp] * np.exp(1j * 2 * np.pi * history['freq2'][tp] * t)
            reconstruction = s1 + s2
            
            if tp == -1:
                ax5.plot(t[:100], np.real(reconstruction[:100]), 'k-', 
                        linewidth=1.5, alpha=alpha, label=label)
            else:
                ax5.plot(t[:100], np.real(reconstruction[:100]), '-', 
                        linewidth=1, alpha=alpha, label=label)
        
        ax5.plot(t[:100], np.real(signal_data[:100]), 'r--', 
                linewidth=1.5, alpha=0.8, label='True signal')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Real part')
        ax5.set_title('Signal Reconstruction Evolution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Extended Kalman Filter Analysis (Reparameterized State)', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return fig
