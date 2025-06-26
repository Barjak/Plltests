import numpy as np
from DualEKFAnalyzer import DualEKFAnalyzer


def test_ekf_analyzer_multi_frame():
    """Test the EKF analyzer with multiple coherent frames and ordering constraint"""
    # Generate test signal with MULTIPLE COHERENT FRAMES
    fs_baseband = 960.0
    frame_duration = 0.3  # seconds per frame
    n_frames = 25  # Number of frames
    total_duration = frame_duration * n_frames
    n_samples = int(fs_baseband * total_duration)
    t = np.arange(n_samples) / fs_baseband
    
    # True frequencies (at baseband) - CONSTANT throughout all frames
    f1_true = 5.625480
    f2_true = 5.631480  # 6 mHz separation
    
    # Generate continuous two-tone signal (no discontinuities!)
    signal_data = (np.exp(1j * 2 * np.pi * f1_true * t) + 
                   0.7 * np.exp(1j * 2 * np.pi * f2_true * t))
    
    # Add some noise
    noise_level = 0.1
    noise = noise_level * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
    signal_data += noise
    
    # Create analyzer
    analyzer = DualEKFAnalyzer(fs_baseband)
    
    # Set up process and measurement noise covariances
    # Process noise (per sample)
    sigma_phi = 1e-7      # Very small phase noise
    sigma_w = 1e-5        # Frequency drift (rad/s)
    sigma_A = 1e-5        # Amplitude drift
    Q = np.diag([sigma_phi**2, sigma_w**2, 
                 sigma_phi**2, sigma_w**2,
                 sigma_A**2, sigma_A**2])
    
    # Measurement noise
    R = 4.0
    
    # Run from multiple initializations WITH ordering enforcement
    print(f"Running Extended Kalman Filter on {n_frames} coherent frames ({total_duration:.1f}s total)...")
    print("WITH frequency ordering constraint (f1 < f2)...")
    results_ordered = analyzer.run_from_multiple_initializations(
        signal_data, f1_true, f2_true, Q=Q, R=R, enforce_ordering=True
    )
    
    # Also run WITHOUT ordering for comparison
    print("\nRunning WITHOUT ordering constraint for comparison...")
    results_unordered = analyzer.run_from_multiple_initializations(
        signal_data, f1_true, f2_true, Q=Q, R=R, enforce_ordering=False
    )
    
    # Print results
    print("\n" + "="*70)
    print(f"EXTENDED KALMAN FILTER RESULTS - {n_frames} FRAMES (WITH ORDERING)")
    print("="*70)
    print(f"True: f1={f1_true:.6f} Hz, f2={f2_true:.6f} Hz, beat={6.000:.3f} mHz")
    print(f"True: A1=1.000, A2=0.700")
    print("\nInitialization -> Result:")
    
    for result in results_ordered:
        print(f"\n{result['label']}:")
        print(f"  Init: f1={result['f1_init']:.6f}, f2={result['f2_init']:.6f}")
        print(f"  Final: f1={result['f1']:.6f}, f2={result['f2']:.6f}")
        print(f"  Beat: {result['beat']*1000:.3f} mHz (error: {(result['beat']-(f2_true-f1_true))*1000:+.3f} mHz)")
        print(f"  Amplitudes: A1={result['A1']:.3f}, A2={result['A2']:.3f}")
        print(f"  Total swaps during tracking: {result.get('total_swaps', 0)}")
    
    # Visualize with ordering comparison
    fig = analyzer.visualize_ekf_analysis(
        results_ordered, signal_data, f1_true, f2_true
    )
    
    # Test ordering enforcement specifically
    print("\n" + "="*70)
    print("TESTING ORDERING ENFORCEMENT")
    print("="*70)
    
    # Test cases that should trigger swapping
    swap_test_cases = [
        (f2_true, f1_true, "Initial swap"),
        (f1_true + 0.010, f1_true, "f1_init > f2_init"),
        (f2_true - 0.002, f1_true + 0.002, "Crossing initialization"),
    ]
    
    for f1_init, f2_init, label in swap_test_cases:
        result_ordered = analyzer.dual_ekf_tracking(
            signal_data, f1_init, f2_init, Q=Q, R=R, enforce_ordering=True
        )
        result_unordered = analyzer.dual_ekf_tracking(
            signal_data, f1_init, f2_init, Q=Q, R=R, enforce_ordering=False
        )
        
        print(f"\n{label}:")
        print(f"  Init: f1={f1_init:.6f}, f2={f2_init:.6f}")
        print(f"  With ordering:")
        print(f"    Final: f1={result_ordered['f1']:.6f}, f2={result_ordered['f2']:.6f}")
        print(f"    Beat error: {(result_ordered['beat']-(f2_true-f1_true))*1000:+.3f} mHz")
        print(f"    Swaps: {result_ordered.get('total_swaps', 0)}")
        print(f"  Without ordering:")
        print(f"    Final: f1={result_unordered['f1']:.6f}, f2={result_unordered['f2']:.6f}")
        print(f"    Beat error: {(result_unordered['beat']-(f2_true-f1_true))*1000:+.3f} mHz")
    
    # Test different durations with ordering
    print("\n" + "="*70)
    print("TESTING DIFFERENT SIGNAL DURATIONS (WITH ORDERING)")
    print("="*70)
    
    frame_counts = [1, 5, 20, 50]
    duration_results = []
    
    for n_frames in frame_counts:
        # Generate signal
        duration = frame_duration * n_frames
        n_samples = int(fs_baseband * duration)
        t = np.arange(n_samples) / fs_baseband
        
        signal_data = (np.exp(1j * 2 * np.pi * f1_true * t) + 
                       0.7 * np.exp(1j * 2 * np.pi * f2_true * t))
        noise = noise_level * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
        signal_data += noise
        
        # Run EKF with truth initialization and ordering
        result = analyzer.dual_ekf_tracking(
            signal_data, f1_true, f2_true, Q=Q, R=R, enforce_ordering=True
        )
        
        duration_results.append({
            'n_frames': n_frames,
            'duration': duration,
            'beat_error': (result['beat'] - (f2_true - f1_true)) * 1000,
            'final_std_f1': np.sqrt(result['history']['P'][-1][1,1])/(2*np.pi)*1000,
            'final_std_f2': np.sqrt(result['history']['P'][-1][3,3])/(2*np.pi)*1000,
            'total_swaps': result.get('total_swaps', 0)
        })
        
        print(f"\n{n_frames} frames ({duration:.1f}s):")
        print(f"  Beat error: {duration_results[-1]['beat_error']:+.3f} mHz")
        print(f"  Final σ(f1): {duration_results[-1]['final_std_f1']:.3f} mHz")
        print(f"  Final σ(f2): {duration_results[-1]['final_std_f2']:.3f} mHz")
        print(f"  Total swaps: {duration_results[-1]['total_swaps']}")
    
    return analyzer, results_ordered



# Run the test
if __name__ == "__main__":
    analyzer, results = test_ekf_analyzer_multi_frame()
