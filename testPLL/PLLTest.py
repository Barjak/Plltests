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

def run_tracker_with_history(signal_data, fs, M, T_mem, label, f1_init=None, f2_init=None):
    """Run tracker and collect detailed history."""
    tracker = ToneRLS(M, fs, T_mem)
    
    # Override initial frequency estimates if provided
    if f1_init is not None and f2_init is not None:
        tracker.theta[2*M] = f1_init      # Frequencies now start at index 2*M
        tracker.theta[2*M + 1] = f2_init
    
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
        if n % 1 == 0:
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
        'f1_init': tracker.theta[2*M] if f1_init is None else f1_init,      # Updated index
        'f2_init': tracker.theta[2*M + 1] if f2_init is None else f2_init,  # Updated index
        'history': {k: np.array(v) for k, v in history.items()}
    }
    
    return result
    
    
def compute_error_landscape(signal_data, fs, f1_range, f2_range, f1_true, f2_true, n_points=30, return_colormap=False):
    f1_vals = np.linspace(f1_range[0], f1_range[1], n_points)
    f2_vals = np.linspace(f2_range[0], f2_range[1], n_points)
    F1, F2 = np.meshgrid(f1_vals, f2_vals)
    
    mse_errors = np.zeros_like(F1)
    t_eval = np.arange(min(1000, len(signal_data))) / fs
    
    for i in range(n_points):
        for j in range(n_points):
            s1 = A1 * np.exp(1j * 2 * np.pi * F1[i, j] * t_eval)
            s2 = A2 * np.exp(1j * 2 * np.pi * F2[i, j] * t_eval)
            mse_errors[i, j] = np.mean(np.abs(signal_data[:len(t_eval)] - (s1 + s2))**2)
    
    R = np.sqrt(F1**2 + F2**2)
    combined_errors = np.log(mse_errors + 1e-10) + 0.1 * 2 * np.pi * R
    
    U = F1 / (R + 1e-8)
    V = F2 / (R + 1e-8)
    
    if return_colormap:
        angle = np.arctan2(V, U)
        magnitude = np.sqrt(U**2 + V**2)
        # Simple color: angle → hue, magnitude → brightness
        rgb = plt.cm.hsv((angle + np.pi) / (2 * np.pi))[:, :, :3]
        rgb *= np.clip(magnitude / magnitude.max(), 0.3, 1)[:, :, np.newaxis]
        return F1, F2, combined_errors, U, V, rgb, magnitude, angle
    
    return F1, F2, combined_errors, U, V

def visualize_rls_analysis(results, signal_data, fs, f1_true, f2_true, 
                          f1_high=None, f2_high=None, f_lo=None):
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.3, 
                          width_ratios=[2, 1, 1], height_ratios=[1, 1, 1, 1, 1])
    
    # 1. 2D Phase Space
    ax1 = fig.add_subplot(gs[:, 0])
    f1_range = (f1_true - 0.015, f1_true + 0.015)
    f2_range = (f2_true - 0.015, f2_true + 0.015)
    f1_grid, f2_grid, error_landscape, U, V = compute_error_landscape(
        signal_data, fs, f1_range, f2_range, f1_true, f2_true)
    
    ax1.contourf(f1_grid, f2_grid, error_landscape, levels=50, cmap='viridis')
    
    colors = cm.rainbow(np.linspace(0, 1, len(results)))
    for result, color in zip(results, colors):
        history = result['history']
        ax1.plot(history['freq1'], history['freq2'], '-', color=color)
        ax1.plot(result['f1_init'], result['f2_init'], 'o', color=color, markersize=12)
        ax1.plot(history['freq1'][-1], history['freq2'][-1], 's', color=color, markersize=12)
    
    ax1.plot(f1_true, f2_true, '*', color='red', markersize=25)
    diag_line = np.array([min(f1_range[0], f2_range[0]), max(f1_range[1], f2_range[1])])
    ax1.plot(diag_line, diag_line, 'k--')
    ax1.plot(diag_line, diag_line + 0.003, 'k:')
    ax1.set_xlabel('f1 (Hz)')
    ax1.set_ylabel('f2 (Hz)')
    ax1.set_title('2D Frequency Space Trajectories')
    ax1.grid(True)
    ax1.set_aspect('equal')
    
    # 2. Power Spectrum
    # 2. Power Spectrum
    ax2 = fig.add_subplot(gs[0, 1:])
    
    # Zero pad like crazy for ultra-high resolution
    nfft = len(signal_data) * 128  # Massive zero padding
    frequencies, psd = scipy_signal.welch(signal_data, fs=fs, nperseg=len(signal_data), 
                                    nfft=nfft, return_onesided=False, scaling='density')
    ax2.semilogy(frequencies, np.abs(psd), 'k-', linewidth=1, alpha=0.7)
    ax2.axvline(f1_true, color='red', linestyle='--', linewidth=2, label='True')
    ax2.axvline(f2_true, color='red', linestyle='--', linewidth=2)
    ax2.set_xlim([min(f1_true, f2_true) - 25, max(f1_true, f2_true) + 25])
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD')
    ax2.set_title('Power Spectral Density (128x zero-padded)')
    ax2.grid(True, alpha=0.3)
    # 3. Separation Evolution
    ax3 = fig.add_subplot(gs[1, 1:])
    time_axis = np.arange(len(results[0]['history']['separation'])) * 10 / fs
    for result, color in zip(results, colors):
        ax3.plot(time_axis[:len(result['history']['separation'])], 
                result['history']['separation'] * 1000, color=color)
    ax3.axhline(6.0, color='red', linestyle='--')
    ax3.axhline(3.0, color='black', linestyle=':')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('|f2 - f1| (mHz)')
    ax3.grid(True)
    
    # 4. Amplitude Evolution
    ax4 = fig.add_subplot(gs[2, 1:])
    history = results[0]['history']
    ax4.plot(time_axis[:len(history['A1'])], history['A1'], 'b-')
    ax4.plot(time_axis[:len(history['A2'])], history['A2'], 'r-')
    ax4.axhline(A1, color='b', linestyle='--', alpha=0.5)
    ax4.axhline(A2, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True)
    
    # 5. Convergence Comparison
    ax5 = fig.add_subplot(gs[3, 1:])
    labels = [r['label'] for r in results]
    f1_errors = [(r['f1'] - f1_true) * 1000 for r in results]
    f2_errors = [(r['f2'] - f2_true) * 1000 for r in results]
    beat_errors = [(r['beat'] - (f2_true - f1_true)) * 1000 for r in results]
    x = np.arange(len(labels))
    ax5.bar(x - 0.25, f1_errors, 0.25)
    ax5.bar(x, f2_errors, 0.25)
    ax5.bar(x + 0.25, beat_errors, 0.25)
    ax5.set_ylabel('Error (mHz)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels, rotation=45, ha='right')
    ax5.grid(True, axis='y')
    ax5.axhline(0, color='black')
    
    # 6. Signal Reconstruction
    ax6 = fig.add_subplot(gs[4, 1:])
    t = np.arange(200) / fs
    for idx in [0, len(history['freq1'])//4, len(history['freq1'])//2, -1]:
        s1 = history['A1'][idx] * np.exp(1j * (2*np.pi*history['freq1'][idx]*t + history['phase1'][idx]))
        s2 = history['A2'][idx] * np.exp(1j * (2*np.pi*history['freq2'][idx]*t + history['phase2'][idx]))
        ax6.plot(t, np.real(s1 + s2), alpha=0.5)
    ax6.plot(t, np.real(signal_data[:200]), 'r--')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Real part')
    ax6.grid(True)
    
    plt.tight_layout()
    return fig
    
# ═══════════════════════════════════════════════════════════════════
#                        RF TONE TRACKING SYSTEM
# ═══════════════════════════════════════════════════════════════════

# ─────────────────── Configuration Parameters ──────────────────────

# RF Tones
f1_rf = 1000.0      # Hz
f2_rf = 1000.006    # Hz
fs_orig = 48_000.0  # Hz

# Front-end Settings
margin_cents = 50.0
min_separation = 0.003

# Signal Properties
A1, A2 = 1.0, 1.0
phi1, phi2 = 0.0, np.pi/4
noise_level = 0.2
total_duration = 2.5


# ─────────────────── System Initialization ─────────────────────────

Q, window_margin, fs_bb = compute_decimation_params(fs_orig, f1_rf, f2_rf, margin_cents)
print(f"Decimation factor Q: {Q}")
print(f"Window margin: {window_margin:.3f} Hz")
print(f"Baseband sample rate: {fs_bb:.1f} Hz")

passband_freq = 0.8 * fs_bb
atten_dB = 10.0
b_lpf, a_lpf = design_anti_alias_filter(fs_orig, fs_bb, passband_freq, atten_dB)
print(f"Filter order: {len(b_lpf)-1}")


# ─────────────────── Signal Generation ─────────────────────────────

print(f"\nGenerating {total_duration:.1f} seconds of test signal...")
t = np.arange(int(fs_orig * total_duration)) / fs_orig
signal_rf = (A1 * np.cos(2 * np.pi * f1_rf * t + phi1) + 
             A2 * np.cos(2 * np.pi * f2_rf * t + phi2))
signal_rf += noise_level * np.random.randn(len(t))


# ─────────────────── Baseband Conversion ───────────────────────────

f_center = (f1_rf + f2_rf) / 2
print(f"Heterodyne LO frequency: {f_center:.3f} Hz")
i_het, q_het = heterodyne(signal_rf, fs_orig, f_center)

print("Decimating to baseband...")
i_baseband = decimate(i_het, Q, b_lpf, a_lpf)
q_baseband = decimate(q_het, Q, b_lpf, a_lpf)
signal_data = i_baseband + 1j * q_baseband

f1_bb = f1_rf - f_center
f2_bb = f2_rf - f_center
print(f"Expected baseband frequencies: {f1_bb:.6f} Hz, {f2_bb:.6f} Hz")


# ─────────────────── Tracking Analysis ─────────────────────────────

print("\nRunning tracking with different initializations...")

M = 2        # Number of tones
T_mem = 10.0  # Memory duration

results = []

# Near-truth initialization
print("1. Near-truth initialization...")
result = run_tracker_with_history(signal_data, fs_bb, M, T_mem, "Near Truth",
                                 f1_init=f1_bb + 0.001, f2_init=f2_bb - 0.001)
results.append(result)

# Wide initialization
print("2. Wide initialization...")
result = run_tracker_with_history(signal_data, fs_bb, M, T_mem, "Wide",
                                 f1_init=f1_bb - 0.1, f2_init=f2_bb + 0.1)
results.append(result)
print("2. Wide initialization...")
result = run_tracker_with_history(signal_data, fs_bb, M, T_mem, "Wide",
                                 f1_init=f1_bb - 0.3, f2_init=f2_bb - 0.1)
results.append(result)

# Narrow initialization
print("3. Narrow initialization...")
result = run_tracker_with_history(signal_data, fs_bb, M, T_mem, "Narrow",
                                 f1_init=-0.001, f2_init=0.001)
results.append(result)


# ─────────────────── Results & Visualization ───────────────────────

print("\nFinal tracking results:")
for result in results:
    print(f"\n{result['label']}:")
    print(f"  f1: {result['f1']:.6f} Hz (error: {(result['f1']-f1_bb)*1000:.3f} mHz)")
    print(f"  f2: {result['f2']:.6f} Hz (error: {(result['f2']-f2_bb)*1000:.3f} mHz)")
    print(f"  Beat: {result['beat']:.6f} Hz (error: {(result['beat']-(f2_bb-f1_bb))*1000:.3f} mHz)")

print("\nGenerating visualization...")
fig = visualize_rls_analysis(results, signal_data, fs_bb, f1_bb, f2_bb,
                            f1_high=f1_rf, f2_high=f2_rf, f_lo=f_center)
plt.savefig('rls_tracking_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
