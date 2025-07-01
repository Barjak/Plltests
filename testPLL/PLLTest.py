"""Application script for two-tone tracking demonstration."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy import signal as scipy_signal
from scipy import special
from signal_tracker_lib import (
    ToneRLS,
    ToneLM,
    ToneMUSIC
)
from preprocess import (RingBuffer, StreamingPreprocessor)

    
def compute_error_landscape(signal_data, fs, f1_range, f2_range, f1_true, f2_true, n_points=30, return_colormap=False):
    """Compute error landscape if signal data is available."""
    if signal_data is None or len(signal_data) == 0:
        # Return dummy data if no signal available
        f1_vals = np.linspace(f1_range[0], f1_range[1], n_points)
        f2_vals = np.linspace(f2_range[0], f2_range[1], n_points)
        F1, F2 = np.meshgrid(f1_vals, f2_vals)
        dummy_errors = np.ones_like(F1)
        U = V = np.zeros_like(F1)
        if return_colormap:
            rgb = np.zeros((*F1.shape, 3))
            magnitude = angle = np.zeros_like(F1)
            return F1, F2, dummy_errors, U, V, rgb, magnitude, angle
        return F1, F2, dummy_errors, U, V
    
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
    """Visualize RLS tracking results with full signal reconstruction."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.3, 
                          width_ratios=[2, 1, 1], height_ratios=[1, 1, 1, 1, 1])
    
    # 1. 2D Phase Space
    ax1 = fig.add_subplot(gs[:, 0])
    # Fixed range around 0.0
    f1_range = (-0.015, 0.015)
    f2_range = (-0.015, 0.015)
    
    if signal_data is not None and len(signal_data) > 0:
        f1_grid, f2_grid, error_landscape, U, V = compute_error_landscape(
            signal_data, fs, f1_range, f2_range, f1_true, f2_true)
        ax1.contourf(f1_grid, f2_grid, error_landscape, levels=50, cmap='viridis')
    else:
        ax1.text(0.5, 0.5, 'No signal data available', 
                transform=ax1.transAxes, ha='center', va='center')
    
    colors = cm.rainbow(np.linspace(0, 1, len(results)))
    for result, color in zip(results, colors):
        history = result['history']
        ax1.plot(history['freq1'], history['freq2'], '-', color=color)
        ax1.plot(result['f1_init'], result['f2_init'], 'o', color=color, markersize=12)
        ax1.plot(history['freq1'][-1], history['freq2'][-1], 's', color=color, markersize=12)
    
    ax1.plot(f1_true, f2_true, '*', color='red', markersize=25)
    
    # Diagonal lines constrained to the fixed range
    diag_line = np.array([-0.015, 0.015])
    ax1.plot(diag_line, diag_line, 'k--')
    ax1.plot(diag_line, diag_line + 0.003, 'k:')
    
    # Set fixed axis limits
    ax1.set_xlim(-0.015, 0.015)
    ax1.set_ylim(-0.015, 0.015)
    
    ax1.set_xlabel('f1 (Hz)')
    ax1.set_ylabel('f2 (Hz)')
    ax1.set_title('2D Frequency Space Trajectories')
    ax1.grid(True)
    ax1.set_aspect('equal')
    # 2. Power Spectrum
    ax2 = fig.add_subplot(gs[0, 1:])
    
    if signal_data is not None and len(signal_data) > 0:
        # Zero pad like crazy for ultra-high resolution
        nfft = len(signal_data) * 128  # Massive zero padding
        frequencies, psd = scipy_signal.welch(signal_data, fs=fs, nperseg=len(signal_data), 
                                        nfft=nfft, return_onesided=False, scaling='density')
        ax2.semilogy(frequencies, np.abs(psd), 'k-', linewidth=1, alpha=0.7)
        ax2.axvline(f1_true, color='red', linestyle='--', linewidth=2, label='True')
        ax2.axvline(f2_true, color='red', linestyle='--', linewidth=2)
        ax2.set_xlim([min(f1_true, f2_true) - 25, max(f1_true, f2_true) + 25])
    else:
        ax2.text(0.5, 0.5, 'No signal data available', 
                transform=ax2.transAxes, ha='center', va='center')
    
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
    
    # 6. Signal Reconstruction (now showing complete signal)
    ax6 = fig.add_subplot(gs[4, 1:])
    
    if signal_data is not None and len(signal_data) > 0:
        # Show full signal duration
        t = np.arange(len(signal_data)) / fs
        
        # Plot reconstructed signal at different stages
        history = results[0]['history']
        stages = [0, len(history['freq1'])//4, len(history['freq1'])//2, -1]
        alphas = [0.3, 0.4, 0.5, 0.7]
        
        for idx, alpha in zip(stages, alphas):
            s1 = history['A1'][idx] * np.exp(1j * (2*np.pi*history['freq1'][idx]*t + history['phase1'][idx]))
            s2 = history['A2'][idx] * np.exp(1j * (2*np.pi*history['freq2'][idx]*t + history['phase2'][idx]))
            ax6.plot(t, np.real(s1 + s2), alpha=alpha, label=f't={idx*10/fs:.2f}s' if idx >= 0 else 'final')
        
        # Plot actual signal
        ax6.plot(t, np.real(signal_data), 'r--', alpha=0.5, label='actual')
        ax6.set_xlim(0, min(0.5, t[-1]))  # Show first 0.5 seconds for clarity
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'No signal data available', 
                transform=ax6.transAxes, ha='center', va='center')
    
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Real part')
    ax6.set_title('Signal Reconstruction (full signal)')
    ax6.grid(True)
    
    plt.tight_layout()
    return fig

import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

def preprocess_signal_once(signal_rf, fs_orig, f1_rf, f2_rf, margin_cents, atten_dB):
    """Preprocess RF signal once and cache the result."""
    # Initialize preprocessor
    preprocessor = StreamingPreprocessor(fs_orig, f1_rf, f2_rf, 
                                       margin_cents=margin_cents, 
                                       atten_dB=atten_dB,
                                       passband_ratio=0.8)
    
    # Get derived parameters
    fs_bb = preprocessor.fs_out
    f_center = preprocessor.f_center
    Q = preprocessor.Q
    
    print(f"Preprocessor params: Q={Q}, fs_bb={fs_bb:.1f} Hz, f_center={f_center:.3f} Hz")
    
    # Process entire signal at once
    bb_samples_per_block = int(0.1 * fs_bb)
    rf_samples_per_block = bb_samples_per_block * Q + preprocessor.M
    
    baseband_signal = []
    rf_position = 0
    
    while rf_position < len(signal_rf):
        chunk_end = min(rf_position + rf_samples_per_block, len(signal_rf))
        rf_chunk = signal_rf[rf_position:chunk_end]
        bb_chunk = preprocessor.process(rf_chunk)
        if len(bb_chunk) > 0:
            baseband_signal.append(bb_chunk)
        rf_position = chunk_end
    
    # Concatenate all chunks
    data_bb = np.concatenate(baseband_signal) if baseband_signal else np.array([], dtype=complex)
    
    # Calculate baseband frequencies
    f1_bb = f1_rf - f_center
    f2_bb = f2_rf - f_center
    
    return {
        'data': data_bb,
        'fs_bb': fs_bb,
        'f1_bb': f1_bb,
        'f2_bb': f2_bb,
        'f_center': f_center
    }

def generate_init_points(f1_bb, f2_bb, f_center, num_points=10):
    """Generate random initialization points within ±1 semitone of true RF frequencies."""
    # Semitone ratio
    semitone_ratio = 2**(1/12)  # ≈ 1.0595
    
    # Convert baseband frequencies back to RF
    f1_rf = f1_bb + f_center
    f2_rf = f2_bb + f_center
    
    # Calculate RF bounds (±1 semitone)
    f1_rf_min = f1_rf / semitone_ratio
    f1_rf_max = f1_rf * semitone_ratio
    f2_rf_min = f2_rf / semitone_ratio
    f2_rf_max = f2_rf * semitone_ratio
    
    # Convert bounds back to baseband
    f1_bb_min = f1_rf_min - f_center
    f1_bb_max = f1_rf_max - f_center
    f2_bb_min = f2_rf_min - f_center
    f2_bb_max = f2_rf_max - f_center
    
    print(f"  RF frequencies: f1={f1_rf:.3f} Hz, f2={f2_rf:.3f} Hz")
    print(f"  f1 RF range: [{f1_rf_min:.3f}, {f1_rf_max:.3f}] Hz")
    print(f"  f2 RF range: [{f2_rf_min:.3f}, {f2_rf_max:.3f}] Hz")
    print(f"  f1 BB range: [{f1_bb_min:.3f}, {f1_bb_max:.3f}] Hz")
    print(f"  f2 BB range: [{f2_bb_min:.3f}, {f2_bb_max:.3f}] Hz")
    
    # Generate random points uniformly in the rectangle
    init_points = []
    for _ in range(num_points):
        f1_init = np.random.uniform(f1_bb_min, f1_bb_max)
        f2_init = np.random.uniform(f2_bb_min, f2_bb_max)
        init_points.append((f1_init, f2_init))
    
    return init_points


def evaluate_tracker_single(data_bb, fs_bb, f1_bb, f2_bb, params, f1_init, f2_init):
    """Evaluate tracker performance with given parameters and single initialization."""
    
    # Unpack parameters
    T_mem, beta_smooth, lambda_reg, beta_sep, sigma2_init, q_freq_vel = params
    
    # Initialize tracker with current parameters
    M = 2
    tracker = ToneRLS(M, fs_bb, T_mem,
                     beta_smooth=beta_smooth,
                     lambda_reg=lambda_reg,
                     beta_sep=beta_sep,
                     sigma2_init=sigma2_init,
                     q_freq_vel=q_freq_vel)
    
    # Set initial frequencies
    tracker.theta[2*M] = f1_init
    tracker.theta[2*M + 1] = f2_init
    
    # Track errors over time
    errors_f1 = []
    errors_f2 = []
    errors_beat = []
    raw_errors = []  # For detecting sign
    
    # Process samples
    sample_interval = 100  # Record error every N samples
    
    for i, sample in enumerate(data_bb):
        tracker.update(np.real(sample), np.imag(sample))
        
        if i % sample_interval == 0 and i > 300:  # Skip initial convergence
            state = tracker.get_state()
            f1_est = state['freqs'][0]
            f2_est = state['freqs'][1]
            beat_est = f2_est - f1_est
            beat_true = f2_bb - f1_bb
            
            # Store absolute errors
            errors_f1.append(abs(f1_est - f1_bb))
            errors_f2.append(abs(f2_est - f2_bb))
            errors_beat.append(abs(beat_est - beat_true))
            
            # Store raw errors to check for systematic bias
            raw_errors.extend([f1_est - f1_bb, f2_est - f2_bb, beat_est - beat_true])
    
    # Calculate statistics
    if errors_f1:
        mean_error = np.mean(errors_f1 + errors_f2 + errors_beat)
        std_error = np.std(errors_f1 + errors_f2 + errors_beat)
        beat_error = np.mean(errors_beat) + 2 * np.std(errors_beat)
        mean_raw = np.mean(raw_errors)
        
        # Check for tracker failure conditions
        # Combined metric with penalty for negative bias
        metric = mean_error ** 2 + std_error ** 2
        metric += 0.5 * beat_error ** 2
        # Add penalty if there's systematic negative bias
        if mean_raw < 0:
            penalty = np.exp(np.log(-mean_raw) + 1.4)  # exp(ln(-e) + 3)
            metric += penalty

    else:
        metric = float('inf')
        mean_error = float('inf')
        std_error = float('inf')
    
    return metric, mean_error, std_error, beat_error


def evaluate_tracker(data_bb, fs_bb, f1_bb, f2_bb, params, init_points):
    """Evaluate tracker with multiple initialization points."""
    
    metrics = []
    mean_errors = []
    std_errors = []
    beat_errors = []
    
    # Test with each initialization point
    for f1_init, f2_init in init_points:
        metric, mean_err, std_err, beat_error = evaluate_tracker_single(
            data_bb, fs_bb, f1_bb, f2_bb, params, f1_init, f2_init
        )
        metrics.append(metric)
        mean_errors.append(mean_err)
        std_errors.append(std_err)
        beat_errors.append(beat_error)
    
    # Return average performance across all initializations
    avg_metric = np.mean(metrics)
    avg_mean_error = np.mean(mean_errors)
    avg_std_error = np.mean(std_errors)
    avg_beat_error = np.mean(beat_errors)
    
    return avg_metric, avg_mean_error, avg_std_error, avg_beat_error


def line_search_parameter(data_bb, fs_bb, f1_bb, f2_bb, 
                         base_params, param_idx, param_range, param_name,
                         init_points, log_scale=True):
    """Perform line search on a single parameter."""
    
    results = []
    
    print(f"\n  Testing {param_name}:")
    
    for j, value in enumerate(param_range):
        # Create parameter set with current value
        test_params = list(base_params)
        test_params[param_idx] = value
        
        # Evaluate with multiple initializations
        metric, mean_err, std_err, beat_error = evaluate_tracker(
            data_bb, fs_bb, f1_bb, f2_bb, test_params, init_points
        )
        
        results.append({
            'value': value,
            'metric': metric,
            'mean_error': mean_err,
            'std_error': std_err,
            'beat_error': beat_error
        })
        
        print(f"    [{j+1}] {value:.2e} → metric={metric:.6f}, mean_error={mean_err:.6f}, beat_error={beat_error:.6f}")
    
    # Find best value
    best_idx = np.argmin([r['metric'] for r in results])
    best_value = results[best_idx]['value']
    
    print(f"    Best: {best_value:.2e} (metric={results[best_idx]['metric']:.6f})")
    
    return best_value, results


def hyperparameter_optimization(signal_rf, fs_orig, f1_rf, f2_rf, 
                               margin_cents, atten_dB, num_passes=10):
    """Optimize hyperparameters using iterative line search with multiple init points."""
    
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    # Preprocess signal once
    print("\nPreprocessing signal...")
    cached_data = preprocess_signal_once(signal_rf, fs_orig, f1_rf, f2_rf, 
                                       margin_cents, atten_dB)
    
    data_bb = cached_data['data']
    fs_bb = cached_data['fs_bb']
    f1_bb = cached_data['f1_bb']
    f2_bb = cached_data['f2_bb']
    f_center = cached_data['f_center']
    
    print(f"Baseband signal: {len(data_bb)} samples at {fs_bb:.1f} Hz")
    print(f"Center frequency: {f_center:.3f} Hz")
    print(f"True BB frequencies: f1={f1_bb:.6f} Hz, f2={f2_bb:.6f} Hz")
    
    # Initial parameters
    #========================================================================================


    current_params =    [3.00e+04, 1.02e-02, 8.29e+03, 1.92e-02, 1.41e+04, 2.98e+01]
                       #[5.45e+03, 1.79e-01, 8.60e+04, 9.24e-03, 2.83e+03, 3.49e+03]
    param_names = ['T_mem', 'beta_smooth', 'lambda_reg', 'beta_sep', 'sigma2_init', 'q_freq_vel']
    #========================================================================================

    
    # History for plotting
    param_history = {name: [current_params[i]] for i, name in enumerate(param_names)}
    metric_history = []
    
    # Store ALL parameter combinations tried during optimization
    all_param_combinations = []
    all_param_metrics = []
    
    # Optimization passes
    for pass_num in range(num_passes):
        init_points = generate_init_points(f1_bb, f2_bb, f_center, num_points=60)
        print(f"\n{'─'*50}")
        print(f"Pass {pass_num + 1}/{num_passes}")
        print(f"{'─'*50}")
        
        # Evaluate current parameters
        current_metric, mean_err, std_err, beat_error = evaluate_tracker(
            data_bb, fs_bb, f1_bb, f2_bb, current_params, init_points
        )
        metric_history.append(current_metric)
        
        # Store this combination
        all_param_combinations.append(list(current_params))
        all_param_metrics.append(current_metric)
        
        print(f"Current avg metric: {current_metric:.6f} (mean={mean_err:.6f}, std={std_err:.6f}, beat={beat_error:.6})")
        
        # Store best values found during this pass
        best_values = []
        
        for i, param_name in enumerate(param_names):
            init_points = generate_init_points(f1_bb, f2_bb, f_center, num_points=10)
            
            # Create parameter range centered on current value with Gaussian-like distribution
            current_value = current_params[i]
            
            # Generate points using inverse normal CDF for concentration near center
            # Create percentiles concentrated near 0.5 (median)
            percentiles = np.linspace(0.0001, 0.9999, 9)
            
            # Use inverse error function to get Gaussian-distributed points

            std_dev = 0.5  # Controls spread (smaller = more concentrated)
            gaussian_points = special.erfinv(2 * percentiles - 1) * np.sqrt(2) * std_dev
            
            # Scale to ±1 order of magnitude range
            scaled_points = gaussian_points / 3  # Divide by 3 since most values are within ±3 std dev
            
            # Convert to log space centered on current value
            log_current = np.log10(current_value)
            log_values = log_current + scaled_points
            
            # Convert back to regular space
            param_range = 10 ** log_values
            
            # Ensure positive values and sort
            param_range = np.maximum(param_range, 1e-10)
            param_range = np.sort(param_range)
            
            # Always use log scale for visualization
            log_scale = True        # Line search for each parameter
            
            # Perform line search (using current_params as baseline)
            best_value, search_results = line_search_parameter(
                data_bb, fs_bb, f1_bb, f2_bb,
                current_params, i, param_range, param_name,
                init_points, log_scale
            )
            
            # Store metrics from line search
            all_param_metrics.extend([r['metric'] for r in search_results])
            
            # Store best value for later update
            best_values.append(best_value)
        
        # Now update all parameters at once
        for i, (param_name, best_value) in enumerate(zip(param_names, best_values)):
            old_value = current_params[i]
            new_value = 0.99 * old_value + 0.01* best_value
            current_params[i] = new_value
            param_history[param_name].append(new_value)
    
        print(f"\ncurrent_params = [{', '.join(f'{p:.2e}' for p in current_params)}]")
    
    print("\n" + "="*60)
    print("RE-EVALUATING ALL SOLUTIONS WITH 100 RANDOM POINTS")
    print("="*60)
    
    # Remove duplicates (if any)
    unique_params = []
    seen = set()
    for params in all_param_combinations:
        param_tuple = tuple(params)
        if param_tuple not in seen:
            seen.add(param_tuple)
            unique_params.append(params)
    
    print(f"\nTotal unique parameter combinations to evaluate: {len(unique_params)}")
    
    # Re-evaluate all parameter combinations with 100 random points
    detailed_results = []
    
    for idx, params in enumerate(unique_params):
        if idx % 10 == 0:
            print(f"  Evaluating combination {idx+1}/{len(unique_params)}...")
        
        # Generate 100 random initialization points
        init_points_100 = generate_init_points(f1_bb, f2_bb, f_center, num_points=100)
        
        # Evaluate with 100 points
        metric, mean_err, std_err, beat_error = evaluate_tracker(
            data_bb, fs_bb, f1_bb, f2_bb, params, init_points_100
        )
        
        detailed_results.append({
            'params': params,
            'metric': metric,
            'mean_error': mean_err,
            'std_error': std_err,
            'beat_error': beat_error,
            'index': idx
        })
    
    # Sort by metric
    detailed_results.sort(key=lambda x: x['metric'])
    
    print("\nTop 5 parameter combinations:")
    for i in range(min(5, len(detailed_results))):
        r = detailed_results[i]
        print(f"  {i+1}. Metric: {r['metric']:.6f}")
        for j, name in enumerate(param_names):
            print(f"     {name}: {r['params'][j]:.2e}")
    
    # Create visualizations
    create_optimization_visualizations(detailed_results, param_names, param_history, metric_history)
    
    # Return best parameters and all results
    best_params = detailed_results[0]['params']
    
    return best_params, param_history, metric_history, detailed_results

def create_optimization_visualizations(detailed_results, param_names, param_history, metric_history):
    """Create progressive bar chart and t-SNE visualization."""
    
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Progressive Bar Chart
    ax1 = plt.subplot(2, 2, 1)
    
    # Sort results by evaluation order (index)
    sorted_by_index = sorted(detailed_results, key=lambda x: x['index'])
    
    # Calculate running minimum
    running_min = []
    current_min = float('inf')
    for r in sorted_by_index:
        current_min = min(current_min, r['metric'])
        running_min.append(current_min)
    
    indices = [r['index'] for r in sorted_by_index]
    metrics = [r['metric'] for r in sorted_by_index]
    
    # Plot bars
    bars = ax1.bar(range(len(metrics)), metrics, alpha=0.6, color='lightblue', edgecolor='navy')
    
    # Highlight best solutions
    best_indices = sorted(range(len(metrics)), key=lambda i: metrics[i])[:5]
    for idx in best_indices:
        bars[idx].set_color('red')
        bars[idx].set_alpha(0.8)
    
    # Plot running minimum
    ax1.plot(range(len(running_min)), running_min, 'g-', linewidth=2, label='Running Best')
    
    ax1.set_xlabel('Evaluation Order')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Progressive Performance During Optimization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Metric Distribution
    ax2 = plt.subplot(2, 2, 2)
    
    metrics_array = np.array(metrics)
    ax2.hist(metrics_array[metrics_array < np.percentile(metrics_array, 95)], 
             bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_xlabel('Metric Value')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Metric Values (95th percentile)')
    ax2.grid(True, alpha=0.3)
    
    # 3. t-SNE Visualization
    ax3 = plt.subplot(2, 2, 3)
    
    # Prepare parameter matrix
    param_matrix = np.array([r['params'] for r in detailed_results])
    
    # Standardize parameters
    scaler = StandardScaler()
    param_matrix_scaled = scaler.fit_transform(param_matrix)
    
    # Perform t-SNE
    print("\nPerforming t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(detailed_results)-1))
    embedding = tsne.fit_transform(param_matrix_scaled)
    
    # Create scatter plot colored by metric
    metrics_for_color = [r['metric'] for r in detailed_results]
    scatter = ax3.scatter(embedding[:, 0], embedding[:, 1], 
                         c=metrics_for_color, cmap='viridis_r', 
                         s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Highlight top 5 solutions
    top_5_indices = sorted(range(len(metrics_for_color)), key=lambda i: metrics_for_color[i])[:5]
    ax3.scatter(embedding[top_5_indices, 0], embedding[top_5_indices, 1], 
               c='red', s=200, marker='*', edgecolors='black', linewidth=2, 
               label='Top 5 Solutions')
    
    plt.colorbar(scatter, ax=ax3, label='Metric Value')
    ax3.set_xlabel('t-SNE Component 1')
    ax3.set_ylabel('t-SNE Component 2')
    ax3.set_title('t-SNE Visualization of Parameter Space')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Parameter Importance (variation analysis)
    ax4 = plt.subplot(2, 2, 4)
    
    # Calculate parameter variations for top 10% solutions
    n_top = max(1, len(detailed_results) // 10)
    top_results = detailed_results[:n_top]
    top_params = np.array([r['params'] for r in top_results])
    
    # Calculate coefficient of variation for each parameter
    param_means = np.mean(top_params, axis=0)
    param_stds = np.std(top_params, axis=0)
    cv = param_stds / (np.abs(param_means) + 1e-10)
    
    # Plot
    bars = ax4.bar(param_names, cv, color='orange', alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Coefficient of Variation')
    ax4.set_title('Parameter Variation in Top 10% Solutions')
    ax4.set_xticklabels(param_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, cv):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Additional figure for parameter evolution
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, history) in enumerate(param_history.items()):
        ax = axes[i]
        ax.plot(history, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Optimization Pass')
        ax.set_ylabel(name)
        ax.set_title(f'{name} Evolution')
        if name != 'beta_smooth':
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Parameter Evolution During Optimization', fontsize=16)
    plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
#                    HYPERPARAMETER OPTIMIZATION RUN
# ═══════════════════════════════════════════════════════════════════

# Configuration
f1_rf = 1000.0      # Hz
f2_rf = 1000.006    # Hz
fs_orig = 48_000.0  # Hz
margin_cents = 50.0
atten_dB = 60.0

# Signal generation
A1, A2 = 1.0, 1.0
phi1, phi2 = 0.0, np.pi/4
noise_level = 0.2
total_duration = 2.5  # Longer duration for better statistics

print(f"\nGenerating {total_duration:.1f} seconds of test signal...")
t = np.arange(int(fs_orig * total_duration)) / fs_orig
signal_rf = (A1 * np.cos(2 * np.pi * f1_rf * t + phi1) + 
             A2 * np.cos(2 * np.pi * f2_rf * t + phi2))
signal_rf += noise_level * np.random.randn(len(t))

# Run optimization
optimal_params, param_history, metric_history, detailed_results = hyperparameter_optimization(
    signal_rf, fs_orig, f1_rf, f2_rf, margin_cents, atten_dB, num_passes=30
)

plt.show()


#def run_tracker_streaming(signal_rf, fs_orig, f1_rf, f2_rf, margin_cents, atten_dB,
#                         M, T_mem, label, f1_init=None, f2_init=None):
#    """Run tracker with streaming preprocessor and ring buffer."""
#    
#    # Initialize preprocessor with automatic parameter calculation
#    preprocessor = StreamingPreprocessor(fs_orig, f1_rf, f2_rf, 
#                                       margin_cents=margin_cents, 
#                                       atten_dB=atten_dB,
#                                       passband_ratio=0.8)
#    
#    # Get derived parameters from preprocessor
#    fs_bb = preprocessor.fs_out
#    f_center = preprocessor.f_center
#    Q = preprocessor.Q
#    
#    print(f"  Preprocessor params: Q={Q}, fs_bb={fs_bb:.1f} Hz, f_center={f_center:.3f} Hz")
#    
#    # Initialize ring buffer for 2.5 seconds of baseband samples
#    ring_buffer_size = int(T_mem * fs_bb)
#    ring_buffer = RingBuffer(ring_buffer_size)
#    
#    # Calculate baseband frequencies
#    f1_bb = f1_rf - f_center
#    f2_bb = f2_rf - f_center
#    
#    # Determine frequency search range for baseband signal
#    freq_margin = 1.0  # Hz
#    min_search_freq = min(f1_bb, f2_bb, -1.0) - freq_margin
#    max_search_freq = max(f1_bb, f2_bb, 1.0) + freq_margin
#    
#    # Initialize tracker
#    tracker = ToneRLS(M, fs_bb, T_mem)
#    
#    # Calculate baseband frequencies
#    f1_bb = f1_rf - f_center
#    f2_bb = f2_rf - f_center
#    
#    # Override initial frequency estimates if provided
#    if f1_init is not None and f2_init is not None:
#        tracker.theta[2*M] = f1_init
#        tracker.theta[2*M + 1] = f2_init
#
#    # Calculate chunk size for 0.1 seconds of baseband samples
#    bb_samples_per_block = int(0.1 * fs_bb)
#    # Calculate corresponding RF samples needed (with some margin for filter transients)
#    rf_samples_per_block = bb_samples_per_block * Q + preprocessor.M
#    
#    # Storage for history
#    history = {
#        'freq1': [],
#        'freq2': [],
#        'A1': [],
#        'A2': [],
#        'separation': [],
#        'phase1': [],
#        'phase2': []
#    }
#    
#    # Accumulator for complete baseband signal
#    baseband_accumulator = []
#    
#    # Process signal in chunks
#    total_rf_samples = len(signal_rf)
#    rf_position = 0
#    
#    while rf_position < total_rf_samples:
#        # Get next chunk of RF samples
#        chunk_end = min(rf_position + rf_samples_per_block, total_rf_samples)
#        rf_chunk = signal_rf[rf_position:chunk_end]
#        
#        # Process through streaming preprocessor
#        bb_chunk = preprocessor.process(rf_chunk)
#        
#        # Add to ring buffer and accumulator
#        if len(bb_chunk) > 0:
#            ring_buffer.push(bb_chunk)
#            baseband_accumulator.append(bb_chunk)
#            
#            # Process new samples through tracker sample-by-sample
#            for sample in bb_chunk:
#                tracker.update(np.real(sample), np.imag(sample))
#                
#                # Store history periodically
#                history_interval = 100
#                if ring_buffer.total_samples % history_interval == 0:
#                    state = tracker.get_state()
#                    if M >= 2:
#                        history['freq1'].append(state['freqs'][0])
#                        history['freq2'].append(state['freqs'][1])
#                        history['A1'].append(state['amplitudes'][0])
#                        history['A2'].append(state['amplitudes'][1])
#                        history['separation'].append(abs(state['freqs'][1] - state['freqs'][0]))
#                        history['phase1'].append(state['phases'][0])
#                        history['phase2'].append(state['phases'][1])
#        
#        rf_position = chunk_end
#    
#    # Get final state
#    final_state = tracker.get_state()
#    
#    # Concatenate all baseband chunks
#    data_full = np.concatenate(baseband_accumulator) if baseband_accumulator else np.array([], dtype=complex)
#    
#    # Get last 2.5 seconds from ring buffer
#    data_last_frame = ring_buffer.get_all()
#    
#    result = {
#        'label': label,
#        'f1': final_state['freqs'][0] if M >= 1 else 0.0,
#        'f2': final_state['freqs'][1] if M >= 2 else 0.0,
#        'beat': final_state['freqs'][1] - final_state['freqs'][0] if M >= 2 else 0.0,
#        'f1_init': f1_init if f1_init is not None else target_freq,
#        'f2_init': f2_init if f2_init is not None else target_freq,
#        'history': {k: np.array(v) for k, v in history.items()},
#        'data_full': data_full,
#        'data_last_frame': data_last_frame,
#        'f1_bb': f1_bb,
#        'f2_bb': f2_bb,
#        'fs_bb': fs_bb
#    }
#    
#    return result
#
## ═══════════════════════════════════════════════════════════════════
##                        RF TONE TRACKING SYSTEM
## ═══════════════════════════════════════════════════════════════════
#
## ─────────────────── Configuration Parameters ──────────────────────
#
## RF Tones
#f1_rf = 1000.0      # Hz
#f2_rf = 1000.006    # Hz
#fs_orig = 48_000.0  # Hz
#
## Front-end Settings
#margin_cents = 50.0
#atten_dB = 60.0
#
## Signal Properties
#A1, A2 = 1.0, 1.0
#phi1, phi2 = 0.0, np.pi/4
#noise_level = 0.1
#total_duration = 2.5
#
#
## ─────────────────── Signal Generation ─────────────────────────────
#
#print(f"\nGenerating {total_duration:.1f} seconds of test signal...")
#t = np.arange(int(fs_orig * total_duration)) / fs_orig
#signal_rf = (A1 * np.cos(2 * np.pi * f1_rf * t + phi1) + 
#             A2 * np.cos(2 * np.pi * f2_rf * t + phi2))
#signal_rf += noise_level * np.random.randn(len(t))
#
#
## ─────────────────── Streaming Processing & Tracking ────────────────
#
#print("\nRunning streaming tracker with different initializations...")
#
#M = 2        
#T_mem = 10.0 #OPT 
#
#results = []
#
## Near-truth initialization
#print("\n1. Near-truth initialization...")
#result = run_tracker_streaming(signal_rf, fs_orig, f1_rf, f2_rf, margin_cents, atten_dB,
#                              M, T_mem, "Near Truth",
#                              f1_init=-0.03 + 0.001, f2_init=0.03 - 0.001)
#results.append(result)
#
## Get baseband frequencies from first result for subsequent initializations
#f1_bb = result['f1_bb']
#f2_bb = result['f2_bb']
#fs_bb = result['fs_bb']
#
## Wide initialization
#print("\n2. Wide initialization...")
#result = run_tracker_streaming(signal_rf, fs_orig, f1_rf, f2_rf, margin_cents, atten_dB,
#                              M, T_mem, "Wide",
#                              f1_init=f1_bb - 0.1, f2_init=f2_bb + 0.1)
#results.append(result)
#
## Another wide initialization
#print("\n3. Wide initialization (variant)...")
#result = run_tracker_streaming(signal_rf, fs_orig, f1_rf, f2_rf, margin_cents, atten_dB,
#                              M, T_mem, "Wide (var)",
#                              f1_init=f1_bb - 0.3, f2_init=f2_bb - 0.1)
#results.append(result)
#
## Narrow initialization
#print("\n4. Narrow initialization...")
#result = run_tracker_streaming(signal_rf, fs_orig, f1_rf, f2_rf, margin_cents, atten_dB,
#                              M, T_mem, "Narrow",
#                              f1_init=-0.2, f2_init=0.01)
#results.append(result)
#
#
## ─────────────────── Results & Visualization ───────────────────────
#
#print("\nFinal tracking results:")
#for result in results:
#    print(f"\n{result['label']}:")
#    print(f"  f1: {result['f1']:.6f} Hz (error: {(result['f1']-f1_bb)*1000:.3f} mHz)")
#    print(f"  f2: {result['f2']:.6f} Hz (error: {(result['f2']-f2_bb)*1000:.3f} mHz)")
#    print(f"  Beat: {result['beat']:.6f} Hz (error: {(result['beat']-(f2_bb-f1_bb))*1000:.3f} mHz)")
#
## Choose which data to visualize
#data = results[0]['data_full']  # Use complete signal
#
#print("\nGenerating visualization...")
#fig = visualize_rls_analysis(results, data, fs_bb, f1_bb, f2_bb,
#                            f1_high=f1_rf, f2_high=f2_rf, f_lo=result['f1_bb'] + result['f2_bb'])
#plt.show()
