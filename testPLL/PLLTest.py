import numpy as np
from DualEKFAnalyzer import DualEKFAnalyzer
def test_ekf_analyzer_multi_frame():
    """Test the EKF analyzer with multiple coherent frames and reparameterized state"""
    # Generate test signal with MULTIPLE COHERENT FRAMES
    fs_baseband = 960.0
    frame_duration = 0.3  # seconds per frame
    n_frames = 5  # Number of frames
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
    
    # Set up process and measurement noise covariances for reparameterized state
    # State: [phi1, phi2, w_sum, w_diff, A1, A2]
    sigma_phi = 1e-7      # Phase noise (very small)
    sigma_w_sum = 1e-3    # Average frequency drift (allow more)
    sigma_w_diff = 1e-8   # Frequency difference drift (restrict more)
    sigma_A = 1e-5        # Amplitude drift
    Q = np.diag([sigma_phi**2, sigma_phi**2,     # phi1, phi2
                 sigma_w_sum**2, sigma_w_diff**2,  # w_sum, w_diff
                 sigma_A**2, sigma_A**2])          # A1, A2
    
    # Measurement noise
    R = 4.10
    
    # Run from multiple initializations WITH ordering enforcement
    print(f"Running Extended Kalman Filter on {n_frames} coherent frames ({total_duration:.1f}s total)...")
    print("WITH reparameterized state (w_sum, w_diff) and minimum separation constraint...")
    results_ordered = analyzer.run_from_multiple_initializations(
        signal_data, f1_true, f2_true, Q=Q, R=R, enforce_ordering=True
    )
    
    # Print results
    print("\n" + "="*70)
    print(f"EXTENDED KALMAN FILTER RESULTS - {n_frames} FRAMES (REPARAMETERIZED)")
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
        print(f"  Constraint activations: {result.get('total_swaps', 0)}")  # Will always be 0 now
    
    # Visualize with ordering comparison
    fig = analyzer.visualize_ekf_analysis(
        results_ordered, signal_data, f1_true, f2_true
    )
    
    # Test cases with challenging initializations
    # These test the pseudo-measurement constraint instead of swapping
    constraint_test_cases = [
        (f2_true, f1_true, "Reversed initialization"),
        (f1_true + 0.010, f1_true, "f1_init > f2_init"),
        (f2_true - 0.002, f1_true + 0.002, "Crossing initialization"),
        (f1_true + 0.0025, f2_true - 0.0025, "Very close initialization (1 mHz)"),
        ((f1_true + f2_true) / 2 - 0.001, (f1_true + f2_true) / 2 + 0.001, "Near-degenerate (2 mHz)"),
    ]
    
    print("\n" + "="*70)
    print("CONSTRAINT ENFORCEMENT TESTS")
    print("="*70)
    
    for f1_init, f2_init, label in constraint_test_cases:
        result_ordered = analyzer.dual_ekf_tracking(
            signal_data, f1_init, f2_init, Q=Q, R=R, enforce_ordering=True
        )
        result_unordered = analyzer.dual_ekf_tracking(
            signal_data, f1_init, f2_init, Q=Q, R=R, enforce_ordering=False
        )
        
        print(f"\n{label}:")
        print(f"  Init: f1={f1_init:.6f}, f2={f2_init:.6f} (sep: {abs(f2_init-f1_init)*1000:.1f} mHz)")
        print(f"  With constraint:")
        print(f"    Final: f1={result_ordered['f1']:.6f}, f2={result_ordered['f2']:.6f}")
        print(f"    Final sep: {result_ordered['beat']*1000:.3f} mHz")
        print(f"    Beat error: {(result_ordered['beat']-(f2_true-f1_true))*1000:+.3f} mHz")
        print(f"  Without constraint:")
        print(f"    Final: f1={result_unordered['f1']:.6f}, f2={result_unordered['f2']:.6f}")
        print(f"    Final sep: {abs(result_unordered['f2']-result_unordered['f1'])*1000:.3f} mHz")
        print(f"    Beat error: {(result_unordered['beat']-(f2_true-f1_true))*1000:+.3f} mHz")
    
    # Test different process noise ratios
    print("\n" + "="*70)
    print("PROCESS NOISE RATIO TESTS")
    print("="*70)
    
    noise_ratios = [0.1, 1.0, 10.0]  # sigma_w_diff / sigma_w_sum ratios
    
    for ratio in noise_ratios:
        sigma_w_sum = 1e-4
        sigma_w_diff = sigma_w_sum * ratio
        Q_test = np.diag([sigma_phi**2, sigma_phi**2,
                          sigma_w_sum**2, sigma_w_diff**2,
                          sigma_A**2, sigma_A**2])
        
        result = analyzer.dual_ekf_tracking(
            signal_data, f1_true + 0.002, f2_true - 0.002, Q=Q_test, R=R
        )
        
        print(f"\nProcess noise ratio {ratio:.1f} (σ_diff/σ_sum):")
        print(f"  Final: f1={result['f1']:.6f}, f2={result['f2']:.6f}")
        print(f"  Beat error: {(result['beat']-(f2_true-f1_true))*1000:+.3f} mHz")
        print(f"  Min separation during tracking: {np.min(result['history']['separation'])*1000:.3f} mHz")
    
    return analyzer, results_ordered




# Run the test
if __name__ == "__main__":
    analyzer, results = test_ekf_analyzer_multi_frame()
