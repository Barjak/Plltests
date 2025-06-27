import numpy as np
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

def run_from_multiple_initializations(analyzer, signal_data, f1_true, f2_true, 
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
    ]
    
    results = []
    for f1_init, f2_init, label in test_cases:
        result = analyzer.dual_ekf_tracking(signal_data, f1_init, f2_init, Q=Q, R=R)
        result['label'] = label
        result['f1_init'] = f1_init
        result['f2_init'] = f2_init
        results.append(result)
    
    return results

def compute_error_landscape(signal_data, fs, f1_range, f2_range, true_f1, true_f2):
    """Compute the error landscape for visualization"""
    n_points = 100
    f1_grid = np.linspace(f1_range[0], f1_range[1], n_points)
    f2_grid = np.linspace(f2_range[0], f2_range[1], n_points)
    
    error_landscape = np.zeros((n_points, n_points))
    
    # Pre-compute true signal for efficiency
    t = np.arange(len(signal_data)) / fs
    
    for i, f1 in enumerate(f1_grid):
        for j, f2 in enumerate(f2_grid):
            # Skip if f2 < f1 (invalid ordering)
            if f2 < f1:
                error_landscape[j, i] = np.nan
                continue
            
            # Convert to sum/diff for internal clarity
            f_sum = (f1 + f2) / 2
            f_diff = (f2 - f1) / 2
            
            # Convert back (just to show the parameterization is consistent)
            f1_check = f_sum - f_diff
            f2_check = f_sum + f_diff
            
            assert np.abs(f1_check - f1) < 1e-10
            assert np.abs(f2_check - f2) < 1e-10
            
            # Compute error for this (f1, f2) pair
            s1 = np.exp(1j * 2 * np.pi * f1 * t)
            s2 = 0.7 * np.exp(1j * 2 * np.pi * f2 * t)
            estimate = s1 + s2
            
            error = np.mean(np.abs(signal_data - estimate)**2)
            error_landscape[j, i] = error  # Note: j, i for proper orientation
    
    return f1_grid, f2_grid, error_landscape

def visualize_ekf_analysis(results, signal_data, fs, f1_true, f2_true, 
                          f1_high=None, f2_high=None, f_lo=None):
    """Comprehensive visualization of EKF behavior with reparameterized state"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.3, 
                          width_ratios=[2, 1, 1], height_ratios=[1, 1, 1, 1, 1])
    
    # 1. 2D Phase Space Trajectories (large, left side)
    ax1 = fig.add_subplot(gs[:, 0])
    
    # Compute and plot error landscape
    f1_range = (f1_true - 0.015, f1_true + 0.015)
    f2_range = (f2_true - 0.015, f2_true + 0.015)
    f1_grid, f2_grid, error_landscape = compute_error_landscape(
        signal_data, fs, f1_range, f2_range, f1_true, f2_true
    )
    
    # Plot error landscape with more contour lines
    contour = ax1.contourf(f1_grid, f2_grid, np.log10(error_landscape + 1e-10), 
                           levels=50, cmap='viridis', alpha=0.6)
    
    # Plot trajectories
    colors = cm.rainbow(np.linspace(0, 1, len(results)))
    for result, color in zip(results, colors):
        history = result['history']
        ax1.plot(history['freq1'], history['freq2'], '-', color=color, 
                linewidth=1.5, alpha=0.8)
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
            markeredgecolor='black', markeredgewidth=2)
    
    # Diagonal line (f1 = f2)
    diag_line = np.array([min(f1_range[0], f2_range[0]), 
                         max(f1_range[1], f2_range[1])])
    ax1.plot(diag_line, diag_line, 'k--', alpha=0.5, linewidth=2)
    
    # Add minimum separation line (3 mHz below diagonal)
    min_sep_hz = 0.003
    ax1.plot(diag_line, diag_line + min_sep_hz, 'k:', alpha=0.5, linewidth=1.5)
    
    ax1.set_xlabel('f1 (Hz)', fontsize=12)
    ax1.set_ylabel('f2 (Hz)', fontsize=12)
    
    # Update title to show heterodyne info if provided
    if f1_high and f2_high and f_lo:
        title = (f'2D Frequency Space Trajectories (Heterodyned)\n'
                f'Original: {f1_high:.3f} & {f2_high:.3f} Hz, LO: {f_lo:.3f} Hz')
    else:
        title = '2D Frequency Space Trajectories (Reparameterized EKF)'
    ax1.set_title(title, fontsize=14)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add colorbar for error landscape
    cbar = plt.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('log10(MSE)', fontsize=10)
    
    # 2. Power Spectrum (top right)
    ax2 = fig.add_subplot(gs[0, 1:])
    
    # Compute power spectrum
    nperseg = min(1024, len(signal_data) // 4)
    freqs, psd = scipy_signal.welch(signal_data, fs=fs, nperseg=nperseg, 
                                   return_onesided=False, scaling='density')
    
    # Sort frequencies for proper plotting
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    psd = psd[sort_idx]
    
    # Convert to dB
    psd_db = 10 * np.log10(psd + 1e-12)
    
    ax2.plot(freqs, psd_db, 'b-', linewidth=1.5)
    ax2.axvline(f1_true, color='red', linestyle='--', alpha=0.7, label=f'f1={f1_true:.3f} Hz')
    ax2.axvline(f2_true, color='red', linestyle='--', alpha=0.7, label=f'f2={f2_true:.3f} Hz')
    
    # Zoom in on region of interest
    ax2.set_xlim(-30, 30)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density (dB)')
    ax2.set_title('Power Spectrum of Heterodyned Signal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Separation Evolution (second from top right)
    ax3 = fig.add_subplot(gs[1, 1:])
    
    for result, color in zip(results, colors):
        history = result['history']
        samples = np.arange(len(history['separation']))
        ax3.plot(samples, history['separation'] * 1000, '-', color=color, 
                linewidth=2, alpha=0.8)
    
    ax3.axhline(6.0, color='red', linestyle='--', linewidth=2, label='True separation')
    ax3.axhline(3.0, color='black', linestyle=':', linewidth=1.5, label='Min separation')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('|f2 - f1| (mHz)')
    ax3.set_title('Frequency Separation Evolution')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, max(20, ax3.get_ylim()[1])])
    ax3.legend()
    
    # 4. Uncertainty Evolution (third from top right)
    ax4 = fig.add_subplot(gs[2, 1:])
    
    # Plot frequency uncertainty for the "Truth" case
    truth_result = results[0]
    P_history = truth_result['history']['P']
    
    # Extract standard deviations for frequencies from reparameterized state
    std_f1 = []
    std_f2 = []
    std_w_sum = []
    std_w_diff = []
    
    for P in P_history:
        # Extract variances
        var_w_sum = P[2, 2]
        var_w_diff = P[3, 3]
        cov_sum_diff = P[2, 3]
        
        # Compute frequency variances using error propagation
        var_w1 = var_w_sum + var_w_diff - 2 * cov_sum_diff
        var_w2 = var_w_sum + var_w_diff + 2 * cov_sum_diff
        
        # Convert to frequency standard deviations in mHz
        std_f1.append(np.sqrt(max(0, var_w1)) / (2 * np.pi) * 1000)
        std_f2.append(np.sqrt(max(0, var_w2)) / (2 * np.pi) * 1000)
        std_w_sum.append(np.sqrt(var_w_sum) / (2 * np.pi) * 1000)
        std_w_diff.append(np.sqrt(var_w_diff) / (2 * np.pi) * 1000)
    
    samples = np.arange(len(std_f1))
    ax4.plot(samples, std_f1, 'b-', linewidth=2, label='f1 uncertainty')
    ax4.plot(samples, std_f2, 'r-', linewidth=2, label='f2 uncertainty')
    ax4.plot(samples, std_w_sum, 'g--', linewidth=1.5, alpha=0.7, label='f_avg uncertainty')
    ax4.plot(samples, std_w_diff, 'm--', linewidth=1.5, alpha=0.7, label='f_diff/2 uncertainty')
    
    ax4.set_xlabel('Sample')
    ax4.set_ylabel('Frequency Uncertainty (mHz)')
    ax4.set_title('EKF Uncertainty Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. Convergence Comparison (fourth from top right)
    ax5 = fig.add_subplot(gs[3, 1:])
    
    # Bar chart of final errors
    labels = [r['label'] for r in results]
    f1_errors = [(r['f1'] - f1_true) * 1000 for r in results]
    f2_errors = [(r['f2'] - f2_true) * 1000 for r in results]
    beat_errors = [(r['beat'] - (f2_true - f1_true)) * 1000 for r in results]
    
    x = np.arange(len(labels))
    width = 0.25
    
    ax5.bar(x - width, f1_errors, width, label='f1 error', alpha=0.8)
    ax5.bar(x, f2_errors, width, label='f2 error', alpha=0.8)
    ax5.bar(x + width, beat_errors, width, label='beat error', alpha=0.8)
    
    ax5.set_xlabel('Initialization')
    ax5.set_ylabel('Error (mHz)')
    ax5.set_title('Final Estimation Errors')
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axhline(0, color='black', linewidth=0.6)
    
    # 6. Signal Reconstruction (bottom right)
    ax6 = fig.add_subplot(gs[4, 1:])
    
    # Use the "Truth" initialization case
    history = truth_result['history']
    
    # Reconstruct signals at a few time points
    t = np.arange(len(signal_data)) / fs
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
            ax6.plot(t[:100], np.real(reconstruction[:100]), 'k-', 
                    linewidth=1.5, alpha=alpha, label=label)
        else:
            ax6.plot(t[:100], np.real(reconstruction[:100]), '-', 
                    linewidth=1, alpha=alpha, label=label)
    
    ax6.plot(t[:100], np.real(signal_data[:100]), 'r--', 
            linewidth=1.5, alpha=0.8, label='True signal')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Real part')
    ax6.set_title('Signal Reconstruction Evolution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Update main title to indicate heterodyne if info provided
    if f1_high and f2_high and f_lo:
        main_title = 'Extended Kalman Filter Analysis (Heterodyned Signal)'
    else:
        main_title = 'Extended Kalman Filter Analysis (Reparameterized State)'
    plt.suptitle(main_title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return fig

def test_ekf_analyzer_with_proper_heterodyne():
    """Test the EKF analyzer following the exact heterodyne specification"""
    
    import numpy as np
    from scipy import signal as scipy_signal
    
    # 1. Compute the "perfect" LO for integer down-sampling
    # 1.1 RF tones
    f1_rf, f2_rf = 1000.0, 1000.006  # Hz
    fs_orig = 48_000.0  # input sample rate
    window_width = f2_rf - f1_rf  # 6e-3 Hz separation
    # plus a little margin: say ±50 cents ≃ ±29.3 Hz ⇒ total ≃ 58.6 Hz
    window_margin = (f1_rf * (2**(50/1200) - 1)) * 2
    
    # 1.2 RF center & desired BB Nyquist
    f_center = 0.5 * (f1_rf + f2_rf)  # 1000.003 Hz
    # We need fs_bb/2 ≥ window_margin/2 ⇒ fs_bb ≥ window_margin
    # And we want Q = fs_orig/fs_bb integer, so:
    max_Q = int(1.0 * fs_orig / window_margin)
    # pick the largest integer Q that still satisfies Nyquist
    decim_factor = max_Q  # Changed from Q to avoid confusion
    fs_bb = fs_orig / decim_factor
    
    # 1.3 LO sits right at the RF center
    lo_freq = f_center
    
    print(f"Computed parameters:")
    print(f"  Window margin: {window_margin:.3f} Hz")
    print(f"  Decimation factor: {decim_factor}")
    print(f"  Baseband sample rate: {fs_bb:.3f} Hz")
    print(f"  LO frequency: {lo_freq:.6f} Hz")
    print(f"  Baseband f1: {f1_rf - lo_freq:.6f} Hz")
    print(f"  Baseband f2: {f2_rf - lo_freq:.6f} Hz")
    
    # Generate test signal
    frame_duration = 0.3  # seconds per frame
    n_frames = 10  # Number of frames
    total_duration = frame_duration * n_frames
    n_samples = int(fs_orig * total_duration)
    t = np.arange(n_samples) / fs_orig
    
    # Generate RF signal (real-valued, as would come from ADC)
    A1, A2 = 1.0, 0.7
    phi1, phi2 = 0.0, np.pi/4  # Initial phases
    signal_rf = (A1 * np.cos(2 * np.pi * f1_rf * t + phi1) + 
                 A2 * np.cos(2 * np.pi * f2_rf * t + phi2))
    
    # Add noise
    noise_level = 0.51
    signal_rf += noise_level * np.random.randn(n_samples)
    
    # 2. I/Q mixer (without calibration first)
    # For real signal, we assume imag_rf = 0
    real_rf = signal_rf
    imag_rf = np.zeros_like(signal_rf)
    
    # 3.1 Complex demod
    cosl = np.cos(2 * np.pi * lo_freq * t)
    sinl = -np.sin(2 * np.pi * lo_freq * t)
    I = real_rf * cosl - imag_rf * sinl
    Q = real_rf * sinl + imag_rf * cosl  # This is the quadrature component
    
    # 2. I/Q calibration (simplified - normally done with calibration tone)
    # For this test, we'll assume perfect I/Q (no imbalance)
    gain_corr = 1.0
    phase_corr = 1.0
    correction = gain_corr * phase_corr
    
    y_bb = (I + 1j * Q) * correction  # calibrated I/Q
    
    # 3.2 Anti-alias & downsample
    # Use FIR decimation with specified decimation factor
    y_dec = scipy_signal.decimate(y_bb, decim_factor, ftype='fir', zero_phase=True)
    
    # Time array for decimated signal
    n_samples_dec = len(y_dec)
    t_dec = np.arange(n_samples_dec) / fs_bb
    
    # Expected baseband frequencies
    f1_bb = f1_rf - lo_freq  # Should be -0.003 Hz
    f2_bb = f2_rf - lo_freq  # Should be +0.003 Hz
    
    print(f"\nSignal characteristics:")
    print(f"  Original samples: {n_samples}")
    print(f"  Decimated samples: {n_samples_dec}")
    print(f"  Duration: {total_duration:.1f} s")
    print(f"  Frames: {n_frames}")
    
    # 4. EKF initialization & tuning
    from DualEKFAnalyzer import DualEKFAnalyzer
    analyzer = DualEKFAnalyzer(fs_bb)
    
    # Process noise - loosened σ_f as specified
    sigma_phi = 3e-10      # Phase noise
    sigma_f = 5e-10        # Frequency drift (5×10⁻⁴ Hz as specified)
    sigma_A = 1e-10        # Amplitude drift
    Q_ekf = np.diag([sigma_phi**2, sigma_phi**2,     # phi1, phi2
                     sigma_f**2, (sigma_f*.2)**2,          # f1, f2
                     sigma_A**2, sigma_A**2])         # A1, A2
    
    # Measurement noise
    R = 10.4
    
    # FFT seed - zero-pad to 2^20 points as specified
    nfft = 2**20
    Y = np.fft.fft(y_dec[:min(len(y_dec), int(fs_bb))], n=nfft)  # Use 1 second of data
    freqs = np.fft.fftfreq(nfft, 1/fs_bb)
    
    # Find peaks in the window [-window_margin/2, +window_margin/2]
    mask = (np.abs(freqs) <= window_margin/2)
    Y_mag = np.abs(Y[mask])
    freqs_mask = freqs[mask]
    
    # Find two largest peaks
    peaks, _ = scipy_signal.find_peaks(Y_mag, height=np.max(Y_mag)*0.1)
    if len(peaks) >= 2:
        sorted_peaks = peaks[np.argsort(Y_mag[peaks])[::-1][:2]]
        f1_init, f2_init = sorted(freqs_mask[sorted_peaks])
    else:
        # Fallback to expected values
        f1_init, f2_init = f1_bb, f2_bb
    
    print(f"\nFFT seed:")
    print(f"  f1_init: {f1_init:.6f} Hz")
    print(f"  f2_init: {f2_init:.6f} Hz")
    
    # Run EKF with ordering constraint (min separation ~10 mHz)
    min_separation = 0.006  # 10 mHz
    
    # Process entire signal with EKF
    result = analyzer.dual_ekf_tracking(
        signal_data=y_dec,
        f1_init=f1_init,
        f2_init=f2_init,
        Q=Q_ekf,
        R=R,
        track_amplitude=True,
        min_separation_hz=min_separation,
        separation_buffer_hz=0.002,  # 1 mHz buffer
    )
    
    results_ordered = run_from_multiple_initializations(
        analyzer, y_dec, f1_bb, f2_bb, Q=Q_ekf, R=R, enforce_ordering=True
    )
    
    # Print results
    print("\n" + "="*70)
    print(f"EXTENDED KALMAN FILTER RESULTS - {n_frames} FRAMES (HETERODYNED)")
    print("="*70)
    print(f"True: f1={f1_bb:.6f} Hz, f2={f2_bb:.6f} Hz, beat={6.000:.3f} mHz")
    print(f"True: A1=1.000, A2=0.700")
    print("\nInitialization -> Result:")
    
    for result in results_ordered:
        print(f"\n{result['label']}:")
        print(f"  Init: f1={result['f1_init']:.6f}, f2={result['f2_init']:.6f}")
        print(f"  Final: f1={result['f1']:.6f}, f2={result['f2']:.6f}")
        print(f"  Beat: {result['beat']*1000:.3f} mHz (error: {(result['beat']-(f2_bb-f1_bb))*1000:+.3f} mHz)")
        print(f"  Amplitudes: A1={result['A1']:.3f}, A2={result['A2']:.3f}")
        print(f"  Constraint activations: {result.get('total_swaps', 0)}")
    
    # Visualize with ordering comparison
    fig = visualize_ekf_analysis(
        results_ordered, y_dec, fs_bb, f1_bb, f2_bb
    )
    
    # Extract history for plotting
    history = result['history']
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS WITH PROPER HETERODYNE SPECIFICATION")
    print("="*70)
    print(f"True baseband: f1={f1_bb:.6f} Hz, f2={f2_bb:.6f} Hz")
    print(f"True beat: {(f2_bb - f1_bb)*1000:.3f} mHz")
    print(f"\nFinal estimates:")
    print(f"  f1: {result['f1']:.6f} Hz (error: {(result['f1'] - f1_bb)*1000:+.3f} mHz)")
    print(f"  f2: {result['f2']:.6f} Hz (error: {(result['f2'] - f2_bb)*1000:+.3f} mHz)")
    print(f"  Beat: {result['beat']*1000:.3f} mHz (error: {(result['beat'] - (f2_bb - f1_bb))*1000:+.3f} mHz)")
    print(f"  A1: {result['A1']:.3f}, A2: {result['A2']:.3f}")
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    time_samples = np.arange(len(history['freq1'])) / fs_bb
    
    # Frequency estimates
    axes[0].plot(time_samples, history['freq1'], 'b-', label='f1 estimate', linewidth=0.5)
    axes[0].axhline(f1_bb, color='b', linestyle='--', alpha=0.5, label='f1 true')
    axes[0].plot(time_samples, history['freq2'], 'r-', label='f2 estimate', linewidth=0.5)
    axes[0].axhline(f2_bb, color='r', linestyle='--', alpha=0.5, label='f2 true')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Frequency Tracking with Proper Heterodyne')
    
    # Beat frequency
    beat_history = (history['freq2'] - history['freq1']) * 1000
    axes[1].plot(time_samples, beat_history, 'g-', label='Beat estimate', linewidth=0.5)
    axes[1].axhline((f2_bb - f1_bb)*1000, color='g', linestyle='--', alpha=0.5, label='True beat')
    axes[1].set_ylabel('Beat Frequency (mHz)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Beat error
    beat_error = beat_history - (f2_bb - f1_bb) * 1000
    axes[2].plot(time_samples, beat_error, 'k-', linewidth=0.5)
    axes[2].set_ylabel('Beat Error (mHz)')
    axes[2].grid(True)
    axes[2].set_ylim([-0.1, 0.1])
    
    # Frequency separation
    axes[3].plot(time_samples, history['separation']*1000, 'm-', label='Separation', linewidth=0.5)
    axes[3].axhline(min_separation*1000, color='r', linestyle=':', alpha=0.7, label='Min separation')
    axes[3].set_ylabel('Frequency Separation (mHz)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'result': result,
        'f1_true': f1_bb,
        'f2_true': f2_bb,
        'fs_bb': fs_bb,
        'decim_factor': decim_factor,
        'lo_freq': lo_freq
    }

# Run the test
if __name__ == "__main__":
    results = test_ekf_analyzer_with_proper_heterodyne()