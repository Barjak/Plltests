"""Application script for two-tone tracking demonstration."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy import signal as scipy_signal
from signal_tracker_lib import (
    compute_decimation_params,
    design_anti_alias_filter,
    heterodyne,
    decimate,
    ToneRLS
)

# Parameter block - RF tones configuration
f1_rf = 1000.0      # First tone frequency (Hz)
f2_rf = 1000.006    # Second tone frequency (Hz)
fs_orig = 48_000.0  # Input sample rate (Hz)
margin_cents = 50.0 # ± cents around the 100-cent span
min_separation = 0.003

# Signal parameters
A1, A2 = 1.0, 1.0          # Tone amplitudes
phi1, phi2 = 0.0, np.pi/4  # Initial phases
noise_level = 0.5         # Noise standard deviation

# Compute front-end parameters
# This determines the decimation factor and baseband rate based on tone separation
Q, window_margin, fs_bb = compute_decimation_params(fs_orig, f1_rf, f2_rf, margin_cents)
print(f"Decimation factor Q: {Q}")
print(f"Window margin: {window_margin:.3f} Hz")
print(f"Baseband sample rate: {fs_bb:.1f} Hz")

# Design anti-aliasing filter
# Use passband at 80% of Nyquist, 60 dB stopband attenuation
passband_freq = 0.1 * fs_bb  # Conservative passband edge
atten_dB = 10.0
b_lpf, a_lpf = design_anti_alias_filter(fs_orig, fs_bb, passband_freq, atten_dB)
print(f"Filter order: {len(b_lpf)-1}")

# Generate test RF signal
# Create 10 seconds of signal for better visualization
total_duration = 5.0
print(f"Generating {total_duration:.1f} seconds of test signal...")

t = np.arange(int(fs_orig * total_duration)) / fs_orig
signal_rf = (A1 * np.cos(2 * np.pi * f1_rf * t + phi1) + 
             A2 * np.cos(2 * np.pi * f2_rf * t + phi2))
signal_rf += noise_level * np.random.randn(len(t))

# Heterodyne to baseband
# Mix down using center frequency between the two tones
f_center = (f1_rf + f2_rf) / 2
print(f"Heterodyne LO frequency: {f_center:.3f} Hz")
i_het, q_het = heterodyne(signal_rf, fs_orig, f_center)

# Decimate to baseband rate
# Apply anti-aliasing filter and downsample
print("Decimating to baseband...")
i_baseband = decimate(i_het, Q, b_lpf, a_lpf)
q_baseband = decimate(q_het, Q, b_lpf, a_lpf)

# Form complex baseband signal
signal_data = i_baseband + 1j * q_baseband

# After heterodyne, tones appear at ±(f2-f1)/2 from DC
f1_bb = f1_rf - f_center  # Should be negative
f2_bb = f2_rf - f_center  # Should be positive
print(f"Expected baseband frequencies: {f1_bb:.6f} Hz, {f2_bb:.6f} Hz")


def run_tracker_with_history(signal_data, fs, M, T_mem, label, f1_init=None, f2_init=None):
    """Run tracker and collect detailed history."""
    tracker = ToneRLS(M, fs, T_mem)
    
    # Override initial frequency estimates if provided
    if f1_init is not None and f2_init is not None:
        tracker.theta[M] = f1_init
        tracker.theta[M+1] = f2_init
    
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
    
    # Process signal sample by sample
    N = len(signal_data)
    for n in range(N):
        tracker.update(np.real(signal_data[n]), np.imag(signal_data[n]))
        
        # Store history every 10 samples to reduce memory
        if n % 10 == 0:
            state = tracker.get_state()
            history['freq1'].append(state['freqs'][0])
            history['freq2'].append(state['freqs'][1])
            history['A1'].append(state['amplitudes'][0])
            history['A2'].append(state['amplitudes'][1])
            history['separation'].append(abs(state['freqs'][1] - state['freqs'][0]))
            history['phase1'].append(state['phases'][0])
            history['phase2'].append(state['phases'][1])
    
    # Get final state
    final_state = tracker.get_state()
    
    result = {
        'label': label,
        'f1': final_state['freqs'][0],
        'f2': final_state['freqs'][1],
        'beat': final_state['freqs'][1] - final_state['freqs'][0],
        'f1_init': tracker.theta[M].real if f1_init is None else f1_init,
        'f2_init': tracker.theta[M+1].real if f2_init is None else f2_init,
        'history': {k: np.array(v) for k, v in history.items()}
    }
    
    return result


def compute_error_landscape_with_gradient(signal_data, fs, f1_range, f2_range, f1_true, f2_true, n_points=30):
    """Compute MSE landscape and gradient field for frequency pairs."""
    f1_vals = np.linspace(f1_range[0], f1_range[1], n_points)
    f2_vals = np.linspace(f2_range[0], f2_range[1], n_points)
    F1, F2 = np.meshgrid(f1_vals, f2_vals)
    
    errors = np.zeros_like(F1)
    t = np.arange(min(1000, len(signal_data))) / fs  # Use first 1000 samples or less
    
    # Pre-compute time array for efficiency
    omega_t = 2 * np.pi * t
    
    for i in range(n_points):
        for j in range(n_points):
            # Reconstruct signal with test frequencies
            s1 = A1 * np.exp(1j * F1[i, j] * omega_t)
            s2 = A2 * np.exp(1j * F2[i, j] * omega_t)
            reconstruction = s1 + s2
            
            # Compute MSE
            errors[i, j] = np.mean(np.abs(signal_data[:len(t)] - reconstruction)**2)
    
    # Compute gradients using central differences
    df1 = f1_vals[1] - f1_vals[0]
    df2 = f2_vals[1] - f2_vals[0]
    
    grad_f1, grad_f2 = np.gradient(errors, df2, df1)  # Note: gradient returns (y, x) order
    
    return F1, F2, errors, grad_f1, grad_f2


def create_gradient_colormap(grad_f1, grad_f2):
    """Create RGB image encoding gradient direction and magnitude."""
    # Compute gradient magnitude and direction
    magnitude = np.sqrt(grad_f1**2 + grad_f2**2)
    angle = np.arctan2(grad_f2, grad_f1)  # Angle from f1 axis
    
    # Normalize magnitude for brightness (use log scale for better visibility)
    mag_norm = np.log10(magnitude + 1e-10)
    mag_norm = (mag_norm - mag_norm.min()) / (mag_norm.max() - mag_norm.min() + 1e-10)
    
    # Create HSV representation
    # Hue: direction (0-360 degrees)
    hue = (angle + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
    
    # Saturation: always 1 for pure colors
    saturation = np.ones_like(hue)
    
    # Value: magnitude (darker = smaller gradient)
    value = 0.3 + 0.7 * mag_norm  # Keep minimum brightness at 0.3
    
    # Convert HSV to RGB
    from matplotlib.colors import hsv_to_rgb
    hsv = np.stack([hue, saturation, value], axis=2)
    rgb = hsv_to_rgb(hsv)
    
    return rgb, magnitude, angle


def compute_error_landscape(signal_data, fs, f1_range, f2_range, f1_true, f2_true, n_points=30):
    """Compute combined MSE and phase error rate landscape."""
    f1_vals = np.linspace(f1_range[0], f1_range[1], n_points)
    f2_vals = np.linspace(f2_range[0], f2_range[1], n_points)
    F1, F2 = np.meshgrid(f1_vals, f2_vals)
    
    # Compute frequency error (MSE)
    mse_errors = np.zeros_like(F1)
    t = np.arange(len(signal_data)) / fs
    t_eval = t[:1000]  # Use first 1000 samples
    
    for i in range(n_points):
        for j in range(n_points):
            # Reconstruct signal with test frequencies
            s1 = A1 * np.exp(1j * 2 * np.pi * F1[i, j] * t_eval)
            s2 = A2 * np.exp(1j * 2 * np.pi * F2[i, j] * t_eval)
            reconstruction = s1 + s2
            
            # Compute MSE
            mse_errors[i, j] = np.mean(np.abs(signal_data[:1000] - reconstruction)**2)
    
    # Compute phase error rate landscape
    R = np.sqrt(F1**2 + F2**2)
    phase_error_rate = 2 * np.pi * R
    
    # Combine as log-likelihood: -log(L) = log(MSE) + λ*phase_error_rate
    # This represents both frequency mismatch and phase drift
    lambda_weight = 0.1  # Weight for phase error contribution
    combined_errors = np.log(mse_errors + 1e-10) + lambda_weight * phase_error_rate
    
    # Compute gradient of phase error (for quiver plot)
    eps = 1e-8
    U = F1 / (R + eps)
    V = F2 / (R + eps)
    
    return F1, F2, combined_errors, U, V


def visualize_rls_analysis(results, signal_data, fs, f1_true, f2_true, 
                          f1_high=None, f2_high=None, f_lo=None):
    """Visualization of RLS tracker behavior adapted from EKF visualizer."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.3, 
                          width_ratios=[2, 1, 1], height_ratios=[1, 1, 1, 1, 1])
     # 1. 2D Phase Space Trajectories
    ax1 = fig.add_subplot(gs[:, 0])
    
    # Compute and plot error landscape
    f1_range = (f1_true - 0.015, f1_true + 0.015)
    f2_range = (f2_true - 0.015, f2_true + 0.015)
    f1_grid, f2_grid, error_landscape, U, V = compute_error_landscape(
        signal_data, fs, f1_range, f2_range, f1_true, f2_true
    )
    
    # Plot error landscape
    contour = ax1.contourf(f1_grid, f2_grid, error_landscape, 
                           levels=50, cmap='viridis', alpha=0.6)
    
    # Add quiver plot for gradient field
    skip = 3  # Plot every 3rd arrow for clarity
    ax1.quiver(f1_grid[::skip, ::skip], f2_grid[::skip, ::skip], 
               U[::skip, ::skip], V[::skip, ::skip], 
               alpha=0.3, scale=30, width=0.002)
    
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
        title = '2D Frequency Space Trajectories (RLS Tracker)'
    ax1.set_title(title, fontsize=14)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')
    
    # Add colorbar for error landscape
    cbar = plt.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('log10(MSE)', fontsize=10)
    
    # 2. Power Spectrum
    ax2 = fig.add_subplot(gs[0, 1:])
    
    # Compute power spectrum
    nperseg = min(2048, len(signal_data) // 4)
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
    ax2.set_xlim(-0.05, 0.05)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density (dB)')
    ax2.set_title('Power Spectrum of Baseband Signal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Separation Evolution
    ax3 = fig.add_subplot(gs[1, 1:])
    
    samples = np.arange(len(results[0]['history']['separation'])) * 10  # Account for subsampling
    time_axis = samples / fs
    
    for result, color in zip(results, colors):
        history = result['history']
        ax3.plot(time_axis[:len(history['separation'])], 
                history['separation'] * 1000, '-', color=color, 
                linewidth=2, alpha=0.8, label=result['label'])
    
    ax3.axhline(6.0, color='red', linestyle='--', linewidth=2, label='True separation')
    ax3.axhline(3.0, color='black', linestyle=':', linewidth=1.5, label='Min separation')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('|f2 - f1| (mHz)')
    ax3.set_title('Frequency Separation Evolution')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, max(20, ax3.get_ylim()[1])])
    ax3.legend()
    
    # 4. Amplitude Evolution
    ax4 = fig.add_subplot(gs[2, 1:])
    
    # Plot amplitude evolution for first result
    result = results[0]
    history = result['history']
    
    ax4.plot(time_axis[:len(history['A1'])], history['A1'], 'b-', 
            linewidth=2, label='A1 estimate')
    ax4.plot(time_axis[:len(history['A2'])], history['A2'], 'r-', 
            linewidth=2, label='A2 estimate')
    ax4.axhline(A1, color='b', linestyle='--', alpha=0.5, label='A1 true')
    ax4.axhline(A2, color='r', linestyle='--', alpha=0.5, label='A2 true')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Amplitude Tracking')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Convergence Comparison
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
    
    # 6. Signal Reconstruction
    ax6 = fig.add_subplot(gs[4, 1:])
    
    # Use the first result
    history = results[0]['history']
    
    # Reconstruct signals at a few time points
    t = np.arange(200) / fs  # First 200 samples for clarity
    time_indices = [0, len(history['freq1'])//4, len(history['freq1'])//2, -1]
    
    for i, idx in enumerate(time_indices):
        alpha = 0.3 + 0.7 * i / len(time_indices)
        if idx == -1:
            label = 'Final'
        else:
            label = f't={idx*10/fs:.1f}s'
            
        # Reconstruct using tracked parameters
        f1_est = history['freq1'][idx]
        f2_est = history['freq2'][idx]
        A1_est = history['A1'][idx]
        A2_est = history['A2'][idx]
        phi1_est = history['phase1'][idx]
        phi2_est = history['phase2'][idx]
        
        s1 = A1_est * np.exp(1j * (2 * np.pi * f1_est * t + phi1_est))
        s2 = A2_est * np.exp(1j * (2 * np.pi * f2_est * t + phi2_est))
        reconstruction = s1 + s2
        
        if idx == -1:
            ax6.plot(t, np.real(reconstruction), 'k-', 
                    linewidth=1.5, alpha=alpha, label=label)
        else:
            ax6.plot(t, np.real(reconstruction), '-', 
                    linewidth=1, alpha=alpha, label=label)
    
    ax6.plot(t, np.real(signal_data[:200]), 'r--', 
            linewidth=1.5, alpha=0.8, label='True signal')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Real part')
    ax6.set_title('Signal Reconstruction Evolution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Update main title
    if f1_high and f2_high and f_lo:
        main_title = 'RLS Tracker Analysis (Heterodyned Signal)'
    else:
        main_title = 'RLS Tracker Analysis'
    plt.suptitle(main_title, fontsize=16)
    plt.tight_layout()
    
    return fig


# Run tracking with different initializations
print("\nRunning tracking with different initializations...")

# Parameters for tracking
M = 2  # Track 2 tones
T_mem = 2.0  # 1 second memory

# Different initialization strategies
results = []

# 1. Near-truth initialization
print("1. Near-truth initialization...")
result = run_tracker_with_history(signal_data, fs_bb, M, T_mem, "Near Truth",
                                 f1_init=f1_bb + 0.001, f2_init=f2_bb - 0.001)
results.append(result)

# 2. Wide initialization
print("2. Wide initialization...")
result = run_tracker_with_history(signal_data, fs_bb, M, T_mem, "Wide",
                                 f1_init=f1_bb - 0.01, f2_init=f2_bb + 0.01)
results.append(result)

# 3. Narrow initialization
print("3. Narrow initialization...")
result = run_tracker_with_history(signal_data, fs_bb, M, T_mem, "Narrow",
                                 f1_init=-0.001, f2_init=0.001)
results.append(result)

# Print final results
print("\nFinal tracking results:")
for result in results:
    print(f"\n{result['label']}:")
    print(f"  f1: {result['f1']:.6f} Hz (error: {(result['f1']-f1_bb)*1000:.3f} mHz)")
    print(f"  f2: {result['f2']:.6f} Hz (error: {(result['f2']-f2_bb)*1000:.3f} mHz)")
    print(f"  Beat: {result['beat']:.6f} Hz (error: {(result['beat']-(f2_bb-f1_bb))*1000:.3f} mHz)")

# Create comprehensive visualization
print("\nGenerating visualization...")
fig = visualize_rls_analysis(results, signal_data, fs_bb, f1_bb, f2_bb,
                            f1_high=f1_rf, f2_high=f2_rf, f_lo=f_center)

plt.savefig('rls_tracking_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
