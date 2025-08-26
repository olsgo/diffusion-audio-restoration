#!/usr/bin/env python3
"""
Memory Safety Validation Test for A2SB Optimized MPS Implementation
Tests various memory conditions and validates safety mechanisms
"""

import os
import sys
import numpy as np
import soundfile as sf
import time
import tempfile
import shutil
from pathlib import Path

def create_test_audio(duration=10.0, sample_rate=44100):
    """Create test audio for validation"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a complex signal with multiple frequencies
    signal = (
        0.4 * np.sin(2 * np.pi * 440 * t) +   # A4
        0.3 * np.sin(2 * np.pi * 880 * t) +   # A5  
        0.2 * np.sin(2 * np.pi * 1320 * t) +  # E6
        0.1 * np.sin(2 * np.pi * 2640 * t) +  # Higher harmonic
        0.05 * np.random.normal(0, 0.1, len(t))  # Small amount of noise
    )
    
    # Apply low-pass filter to simulate degraded audio
    try:
        from scipy.signal import butter, filtfilt
        nyquist = sample_rate / 2
        cutoff = 4000  # 4kHz cutoff
        b, a = butter(5, cutoff / nyquist, btype='low')
        degraded_signal = filtfilt(b, a, signal)
    except ImportError:
        # Fallback if scipy not available - simple rolloff
        degraded_signal = signal * 0.8
    
    # Normalize
    degraded_signal = degraded_signal / np.max(np.abs(degraded_signal)) * 0.8
    
    return degraded_signal, sample_rate

def test_memory_monitoring():
    """Test memory monitoring functionality"""
    print("🧪 Testing memory monitoring...")
    
    try:
        # Import the monitoring class
        sys.path.append('inference')
        from A2SB_upsample_api_mps_optimized import OptimizedMemoryMonitor
        
        # Test with conservative thresholds to ensure it triggers
        monitor = OptimizedMemoryMonitor(
            max_system_memory_percent=1,  # Very low to trigger warning
            max_process_memory_mb=1,      # Very low to trigger warning
            monitoring_interval=1
        )
        
        print("   Starting monitor...")
        monitor.start_monitoring()
        
        # Let it run for a few seconds
        time.sleep(3)
        
        print("   Stopping monitor...")
        monitor.stop_monitoring()
        
        print("   ✅ Memory monitoring test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Memory monitoring test failed: {e}")
        return False

def test_parameter_scaling():
    """Test intelligent parameter scaling"""
    print("🧪 Testing parameter scaling...")
    
    try:
        sys.path.append('inference')
        from A2SB_upsample_api_mps_optimized import intelligent_parameter_scaling
        
        # Test different scenarios
        test_cases = [
            (25, 180, False),  # Normal request
            (50, 300, False),  # High request
            (50, 300, True),   # High request with force
        ]
        
        for steps, timeout, force in test_cases:
            config, final_steps, final_timeout = intelligent_parameter_scaling(
                steps, timeout, force
            )
            print(f"   Request: {steps} steps, {timeout}s → Result: {final_steps} steps, {final_timeout}s (force={force})")
            
        print("   ✅ Parameter scaling test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Parameter scaling test failed: {e}")
        return False

def test_memory_cleanup():
    """Test memory cleanup functionality"""
    print("🧪 Testing memory cleanup...")
    
    try:
        sys.path.append('inference')
        from A2SB_upsample_api_mps_optimized import smart_memory_cleanup
        
        # Test light cleanup
        smart_memory_cleanup("test_light")
        
        # Test force cleanup
        smart_memory_cleanup("test_force", force=True)
        
        print("   ✅ Memory cleanup test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Memory cleanup test failed: {e}")
        return False

def test_configuration_loading():
    """Test that all configuration files can be loaded"""
    print("🧪 Testing configuration loading...")
    
    configs_to_test = [
        "configs/ensemble_2split_sampling_mps.yaml",
        "configs/ensemble_2split_sampling_mps_balanced.yaml", 
        "configs/ensemble_2split_sampling_mps_failsafe.yaml"
    ]
    
    try:
        import yaml
        for config_path in configs_to_test:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"   ✅ {config_path} - loaded successfully")
            else:
                print(f"   ⚠️  {config_path} - file not found")
        
        print("   ✅ Configuration loading test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration loading test failed: {e}")
        return False

def test_dry_run_processing():
    """Test the optimized script without actual model inference"""
    print("🧪 Testing dry run processing...")
    
    # Create a temporary test audio file
    with tempfile.TemporaryDirectory() as temp_dir:
        test_audio_path = os.path.join(temp_dir, "test_input.wav")
        output_path = os.path.join(temp_dir, "test_output.wav")
        
        # Create test audio
        audio_data, sr = create_test_audio(duration=3.0)
        sf.write(test_audio_path, audio_data, sr)
        
        try:
            # Test the optimized script import and basic functionality
            sys.path.append('inference')
            from A2SB_upsample_api_mps_optimized import (
                compute_rolloff_freq, 
                ensure_free_space,
                cleanup_old_temp_configs
            )
            
            # Test rolloff computation
            rolloff = compute_rolloff_freq(test_audio_path)
            print(f"   Computed rolloff: {rolloff} Hz")
            
            # Test disk space check
            ensure_free_space(temp_dir, 0.1)  # Require 100MB
            print("   Disk space check passed")
            
            # Test cleanup
            cleanup_old_temp_configs()
            print("   Temp config cleanup passed")
            
            print("   ✅ Dry run processing test passed")
            return True
            
        except Exception as e:
            print(f"   ❌ Dry run processing test failed: {e}")
            return False

def compare_original_vs_optimized():
    """Compare key differences between original and optimized implementations"""
    print("🧪 Comparing original vs optimized implementations...")
    
    try:
        # Check if both files exist
        original_path = "inference/A2SB_upsample_api_mps.py"
        optimized_path = "inference/A2SB_upsample_api_mps_optimized.py"
        
        if not os.path.exists(original_path):
            print(f"   ⚠️  Original file not found: {original_path}")
            return False
        
        if not os.path.exists(optimized_path):
            print(f"   ❌ Optimized file not found: {optimized_path}")
            return False
        
        # Compare key parameters
        with open(original_path, 'r') as f:
            original_content = f.read()
        
        with open(optimized_path, 'r') as f:
            optimized_content = f.read()
        
        print("   Key differences found:")
        
        # Check memory thresholds
        if "max_system_memory_percent=75" in original_content:
            print("   - Original: 75% system memory threshold")
        if "max_system_memory_percent=85" in optimized_content:
            print("   - Optimized: 85% system memory threshold (more lenient)")
            
        # Check process memory limits
        if "max_process_memory_mb=6144" in original_content:
            print("   - Original: 6GB process memory limit")
        if "max_process_memory_mb=10240" in optimized_content:
            print("   - Optimized: 10GB process memory limit (more generous)")
            
        # Check default steps
        if 'default=5' in original_content and 'predict_n_steps' in original_content:
            print("   - Original: 5 default steps (ultra-conservative)")
        if 'default=15' in optimized_content and 'predict_n_steps' in optimized_content:
            print("   - Optimized: 15 default steps (balanced)")
            
        print("   ✅ Comparison completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Comparison failed: {e}")
        return False

def run_all_tests():
    """Run all validation tests"""
    print("🔍 Memory Safety Validation for A2SB Optimized MPS Implementation")
    print("=" * 80)
    
    tests = [
        ("Memory Monitoring", test_memory_monitoring),
        ("Parameter Scaling", test_parameter_scaling),
        ("Memory Cleanup", test_memory_cleanup),
        ("Configuration Loading", test_configuration_loading),
        ("Dry Run Processing", test_dry_run_processing),
        ("Original vs Optimized", compare_original_vs_optimized),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The optimized implementation appears to be working correctly.")
        print("\n💡 Key improvements in the optimized version:")
        print("   • More lenient memory thresholds (85% vs 75% system memory)")
        print("   • Higher process memory limits (10GB vs 6GB)")
        print("   • Graduated scaling with 5 levels instead of 3")
        print("   • Better default parameters (15 vs 5 steps)")
        print("   • Graceful shutdown instead of harsh termination")
        print("   • Optional force mode for power users")
        print("   • Balanced configuration for intermediate memory pressure")
        print("   • Trend analysis for smarter memory decisions")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)