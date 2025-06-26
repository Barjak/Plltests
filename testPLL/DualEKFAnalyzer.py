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
                      enforce_ordering=True):  # New parameter
        """
        Dual-tone tracking using Extended Kalman Filter
        
        State vector: x = [phi1, w1, phi2, w2, A1, A2]
        where phi_i = phase, w_i = angular frequency, A_i = amplitude
        
        enforce_ordering: If True, enforces w1 < w2 constraint
        """
        n_samples = len(signal_data)
        dt = self.dt
        min_separation_rad = 2 * np.pi * min_separation_hz
        
        # Ensure initial ordering if enforce_ordering is True
        if enforce_ordering and f1_init > f2_init:
            f1_init, f2_init = f2_init, f1_init
        
        # State transition matrix
        F = np.array([[1, dt, 0,  0, 0, 0],
                      [0,  1, 0,  0, 0, 0],
                      [0,  0, 1, dt, 0, 0],
                      [0,  0, 0,  1, 0, 0],
                      [0,  0, 0,  0, 1, 0],
                      [0,  0, 0,  0, 0, 1]])
        
        # Initialize state
        x = np.array([0.0,                    # phi1
                      2*np.pi*f1_init,        # w1
                      0.0,                    # phi2
                      2*np.pi*f2_init,        # w2
                      1.0,                    # A1
                      0.7])                   # A2
        
        # Process noise covariance
        if Q is None:
            sigma_phi = 1e-7      # Phase noise (very small)
            sigma_w = 1e-4        # Frequency drift
            sigma_A = 1e-6        # Amplitude drift
            Q = np.diag([sigma_phi**2, sigma_w**2, 
                         sigma_phi**2, sigma_w**2,
                         sigma_A**2, sigma_A**2])
        
        # Measurement noise covariance
        if R is None:
            R = 3.0
        
        # Initial state covariance
        if P0 is None:
            P0 = np.diag([0.1,    # phi1 uncertainty
                          0.1,    # w1 uncertainty (rad/s)
                          0.1,    # phi2 uncertainty
                          0.1,    # w2 uncertainty (rad/s)
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
            'phase2': [x[2]],
            'separation': [abs(f2_init - f1_init)],
            'error': [],
            'K': [],  # Kalman gain
            'swaps': []  # Track when swaps occur
        }
        
        # Main EKF loop
        for k, y in enumerate(signal_data):
            # ---- Predict ----
            x = F @ x
            P = F @ P @ F.T + Q
            
            # Wrap phases to [-pi, pi]
            x[0] = np.angle(np.exp(1j * x[0]))
            x[2] = np.angle(np.exp(1j * x[2]))
            
            # ---- Measurement prediction ----
            phi1, w1, phi2, w2, A1, A2 = x
            
            # Complex measurement prediction
            y_hat = A1 * np.exp(1j * phi1) + A2 * np.exp(1j * phi2)
            
            # ---- Compute Jacobian H ----
            H = np.zeros((2, 6))
            
            # Derivatives w.r.t phi1
            H[0, 0] = -A1 * np.sin(phi1)  # dRe/dphi1
            H[1, 0] = A1 * np.cos(phi1)   # dIm/dphi1
            
            # Derivatives w.r.t w1 (zero for instantaneous measurement)
            H[0, 1] = 0
            H[1, 1] = 0
            
            # Derivatives w.r.t phi2
            H[0, 2] = -A2 * np.sin(phi2)  # dRe/dphi2
            H[1, 2] = A2 * np.cos(phi2)   # dIm/dphi2
            
            # Derivatives w.r.t w2 (zero for instantaneous measurement)
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
            x_before = x.copy()
            x = x + K @ innov
            
            # ---- Enforce ordering constraint if enabled ----
            swap_occurred = False
            if enforce_ordering and x[1] > x[3]:  # If w1 > w2
                # Method 1: Simple swap
                # Swap frequencies
                x[1], x[3] = x[3], x[1]
                # Swap phases
                x[0], x[2] = x[2], x[0]
                # Swap amplitudes
                if track_amplitude:
                    x[4], x[5] = x[5], x[4]
                
                # Also need to swap the corresponding rows/columns in P
                # Create permutation matrix
                perm = np.array([[0, 2, 1, 3, 4, 5],  # Swap indices 1 and 2
                                [2, 0, 3, 1, 5, 4],   # Swap indices 0 and 1, 4 and 5
                                [1, 3, 0, 2, 4, 5],
                                [3, 1, 2, 0, 5, 4],
                                [4, 5, 4, 5, 0, 1],
                                [5, 4, 5, 4, 1, 0]])
                
                # Actually, let's do this more carefully
                # Create identity permutation matrix
                perm_matrix = np.eye(6)
                # Swap rows/columns for phases (0,2)
                perm_matrix[[0, 2]] = perm_matrix[[2, 0]]
                # Swap rows/columns for frequencies (1,3)
                perm_matrix[[1, 3]] = perm_matrix[[3, 1]]
                # Swap rows/columns for amplitudes (4,5)
                if track_amplitude:
                    perm_matrix[[4, 5]] = perm_matrix[[5, 4]]
                
                # Apply permutation to covariance
                P = perm_matrix @ P @ perm_matrix.T
                swap_occurred = True
            
            # ---- Apply minimum separation constraint ----
            w1_new, w2_new = x[1], x[3]
            separation = abs(w2_new - w1_new)
                    
            if separation < min_separation_rad:
                # Compute regularization force
                force = separation_weight * (min_separation_rad - separation) / min_separation_rad
                
                # For ordered constraint, we know w1 < w2, so we can be more specific
                if enforce_ordering:
                    # Push w1 down and w2 up to maintain ordering
                    x[1] -= force * min_separation_rad / 2
                    x[3] += force * min_separation_rad / 2
                else:
                    # Original symmetric push
                    sign = np.sign(w2_new - w1_new)
                    if sign == 0:
                        sign = np.sign(f2_init - f1_init)
                    x[1] -= force * min_separation_rad * sign / 2
                    x[3] += force * min_separation_rad * sign / 2
                
                # Increase uncertainty in frequency estimates when regularization is active
                P[1, 1] *= (1 + force)
                P[3, 3] *= (1 + force)
            
            P = (np.eye(len(x)) - K @ H) @ P
            
            # Ensure positive amplitudes
            if track_amplitude:
                x[4] = max(0.1, x[4])
                x[5] = max(0.1, x[5])
            
            # ---- Store results ----
            history['x'].append(x.copy())
            history['P'].append(P.copy())
            history['y_pred'].append(y_hat)
            history['innov'].append(innov)
            history['freq1'].append(x[1] / (2 * np.pi))  # Convert to Hz
            history['freq2'].append(x[3] / (2 * np.pi))  # Convert to Hz
            history['A1'].append(x[4])
            history['A2'].append(x[5])
            history['phase1'].append(x[0])
            history['phase2'].append(x[2])
            history['separation'].append(abs(x[3] - x[1]) / (2 * np.pi))
            history['error'].append(np.abs(y - y_hat))
            history['K'].append(K.copy())
            history['swaps'].append(swap_occurred)
        
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
        """Comprehensive visualization of EKF behavior"""
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 2D Phase Space Trajectories
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        # Compute and plot error landscape
        f1_range = (f1_true - 0.015, f1_true + 0.015)
        f2_range = (f2_true - 0.015, f2_true + 0.015)
        f1_grid, f2_grid, error_landscape = self.compute_error_landscape(
            signal_data, f1_range, f2_range, f1_true, f2_true
        )
        
        # Plot error landscape as contours
        contour = ax1.contourf(f1_grid, f2_grid, np.log10(error_landscape + 1e-10), 
                               levels=20, cmap='viridis', alpha=0.6)
        
        # Plot trajectories
        colors = cm.rainbow(np.linspace(0, 1, len(results)))
        for result, color in zip(results, colors):
            history = result['history']
            ax1.plot(history['freq1'], history['freq2'], '-', color=color, 
                    linewidth=0.6, alpha=0.8, label=result['label'])
            # Start point
            ax1.plot(result['f1_init'], result['f2_init'], 'o', 
                    color=color, markersize=10, markeredgecolor='black')
            # End point
            ax1.plot(history['freq1'][-1], history['freq2'][-1], 's', 
                    color=color, markersize=10, markeredgecolor='black')
        
        # True values
        ax1.plot(f1_true, f2_true, '*', color='red', markersize=20, 
                markeredgecolor='black', markeredgewidth=2, label='True')
        
        # Diagonal line (f1 = f2)
        diag_line = np.array([min(f1_range[0], f2_range[0]), 
                             max(f1_range[1], f2_range[1])])
        ax1.plot(diag_line, diag_line, 'k--', alpha=0.5, label='f1=f2')
        
        ax1.set_xlabel('f1 (Hz)')
        ax1.set_ylabel('f2 (Hz)')
        ax1.set_title('2D Frequency Space Trajectories (EKF)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. Separation Evolution
        ax2 = fig.add_subplot(gs[0, 2:])
        
        for result, color in zip(results, colors):
            history = result['history']
            samples = np.arange(len(history['separation']))
            ax2.plot(samples, history['separation'] * 1000, '-', color=color, 
                    linewidth=2, label=result['label'])
        
        ax2.axhline(6.0, color='red', linestyle='--', linewidth=2, label='True separation')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('|f2 - f1| (mHz)')
        ax2.set_title('Frequency Separation Evolution (EKF)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, max(20, ax2.get_ylim()[1])])
        
        # 3. Uncertainty Evolution (from covariance)
        ax3 = fig.add_subplot(gs[1, 2:])
        
        # Plot frequency uncertainty for the "Truth" case
        truth_result = results[0]
        P_history = truth_result['history']['P']
        
        # Extract standard deviations for frequencies
        std_f1 = np.array([np.sqrt(P[1,1]) / (2*np.pi) * 1000 for P in P_history])  # mHz
        std_f2 = np.array([np.sqrt(P[3,3]) / (2*np.pi) * 1000 for P in P_history])  # mHz
        
        samples = np.arange(len(std_f1))
        ax3.plot(samples, std_f1, 'b-', linewidth=2, label='f1 uncertainty')
        ax3.plot(samples, std_f2, 'r-', linewidth=2, label='f2 uncertainty')
        
        ax3.set_xlabel('Sample')
        ax3.set_ylabel('Frequency Uncertainty (mHz)')
        ax3.set_title('EKF Uncertainty Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Convergence Comparison
        ax4 = fig.add_subplot(gs[2, :2])
        
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
        ax4.set_title('Final Estimation Errors (EKF)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(0, color='black', linewidth=0.6)
        
        # 5. Signal Reconstruction
        ax5 = fig.add_subplot(gs[2, 2:])
        
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
                        linewidth=0.6, alpha=alpha, label=label)
            else:
                ax5.plot(t[:100], np.real(reconstruction[:100]), '-', 
                        linewidth=0.6, alpha=alpha, label=label)
        
        ax5.plot(t[:100], np.real(signal_data[:100]), 'r--', 
                linewidth=0.6, alpha=0.8, label='True signal')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Real part')
        ax5.set_title('Signal Reconstruction Evolution (EKF)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Innovation Sequence
        ax6 = fig.add_subplot(gs[3, :2])
        
        # Plot innovation for the "Truth" case
        innov_history = np.array(truth_result['history']['innov'])
        samples = np.arange(len(innov_history))
        
        ax6.plot(samples[::10], innov_history[::10, 0], 'b-', 
                alpha=0.7, label='Real innovation')
        ax6.plot(samples[::10], innov_history[::10, 1], 'r-', 
                alpha=0.7, label='Imag innovation')
        
        ax6.set_xlabel('Sample')
        ax6.set_ylabel('Innovation')
        ax6.set_title('EKF Innovation Sequence')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([-0.1, 0.1])
        
        # 7. Summary Statistics
        ax7 = fig.add_subplot(gs[3, 2:])
        ax7.axis('off')
        
        summary_text = "Extended Kalman Filter Analysis Summary\n" + "="*45 + "\n\n"
        summary_text += f"True frequencies: f1={f1_true:.6f} Hz, f2={f2_true:.6f} Hz\n"
        summary_text += f"True beat: {(f2_true-f1_true)*1000:.3f} mHz\n"
        summary_text += f"True amplitudes: A1=1.0, A2=0.7\n\n"
        
        # Find best and worst cases
        beat_errors_abs = [abs(e) for e in beat_errors]
        best_idx = np.argmin(beat_errors_abs)
        worst_idx = np.argmax(beat_errors_abs)
        
        summary_text += f"Best case: {results[best_idx]['label']}\n"
        summary_text += f"  Beat error: {beat_errors[best_idx]:+.3f} mHz\n"
        summary_text += f"  A1={results[best_idx]['A1']:.3f}, A2={results[best_idx]['A2']:.3f}\n\n"
        
        summary_text += f"Worst case: {results[worst_idx]['label']}\n"
        summary_text += f"  Beat error: {beat_errors[worst_idx]:+.3f} mHz\n\n"
        
        # State covariance information
        final_P = truth_result['history']['P'][-1]
        summary_text += "Final uncertainties (Truth case):\n"
        summary_text += f"  σ(f1) = {np.sqrt(final_P[1,1])/(2*np.pi)*1000:.3f} mHz\n"
        summary_text += f"  σ(f2) = {np.sqrt(final_P[3,3])/(2*np.pi)*1000:.3f} mHz\n"
        summary_text += f"  σ(A1) = {np.sqrt(final_P[4,4]):.3f}\n"
        summary_text += f"  σ(A2) = {np.sqrt(final_P[5,5]):.3f}\n"
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
                fontsize=11, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Extended Kalman Filter Comprehensive Analysis', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return fig
