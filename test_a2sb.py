#!/usr/bin/env python3
"""
Simple test script for A2SB Audio Restoration Model
This script tests the bandwidth extension functionality.
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

def create_test_audio():
    """Create a simple test audio file for bandwidth extension."""
    # Generate a simple test signal (sine wave with harmonics)
    duration = 3.0  # seconds
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a signal with multiple frequencies
    signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note
        0.2 * np.sin(2 * np.pi * 1320 * t) + # E6 note
        0.1 * np.sin(2 * np.pi * 2640 * t)   # Higher harmonic
    )
    
    # Apply low-pass filter to simulate degraded audio (remove high frequencies)
    from scipy.signal import butter, filtfilt
    nyquist = sample_rate / 2
    cutoff = 4000  # 4kHz cutoff
    b, a = butter(5, cutoff / nyquist, btype='low')
    degraded_signal = filtfilt(b, a, signal)
    
    # Normalize
    degraded_signal = degraded_signal / np.max(np.abs(degraded_signal)) * 0.8
    
    return degraded_signal, sample_rate

def test_bandwidth_extension():
    """Test the bandwidth extension functionality."""
    print("Creating test audio...")
    test_audio, sr = create_test_audio()
    
    # Save test audio
    test_file = "test_degraded.wav"
    sf.write(test_file, test_audio, sr)
    print(f"Test audio saved as {test_file}")
    
    # Test the A2SB upsample API
    print("\nTesting A2SB bandwidth extension...")
    output_file = "test_restored.wav"
    
    # Get absolute paths before changing directory
    abs_test_file = os.path.abspath(test_file)
    abs_output_file = os.path.abspath(output_file)
    
    # Change to inference directory and run the API
    original_dir = os.getcwd()
    try:
        os.chdir("inference")
        
        # Run the bandwidth extension with conda environment activated
        cmd = f"conda run -n a2sb python A2SB_upsample_api.py -f {abs_test_file} -o {abs_output_file} -n 50"
        print(f"Running: {cmd}")
        
        result = os.system(cmd)
        
        if result == 0:
            print(f"✅ Bandwidth extension completed successfully!")
            print(f"Restored audio saved as {output_file}")
            
            # Check if output file exists in multiple possible locations
            output_locations = [output_file, f"../{output_file}"]
            output_found = False
            
            for location in output_locations:
                if os.path.exists(location):
                    print(f"✅ Output file found at {location}")
                    
                    # Load and analyze the results
                    original, _ = sf.read(f"../{test_file}")
                    restored, _ = sf.read(location)
                    
                    print(f"\nResults:")
                    print(f"Original audio length: {len(original)} samples")
                    print(f"Restored audio length: {len(restored)} samples")
                    print(f"Original RMS: {np.sqrt(np.mean(original**2)):.4f}")
                    print(f"Restored RMS: {np.sqrt(np.mean(restored**2)):.4f}")
                    
                    # Get file info
                    file_size = os.path.getsize(location)
                    print(f"📊 Output file size: {file_size} bytes")
                    
                    output_found = True
                    break
            
            if not output_found:
                print(f"❌ Output file {output_file} was not created")
                return False
                
            return True
        else:
            print(f"❌ Bandwidth extension failed with exit code {result}")
            return False
            
    finally:
        os.chdir(original_dir)

def main():
    """Main test function."""
    print("A2SB Audio Restoration Model Test")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("inference/A2SB_upsample_api.py"):
        print("❌ Error: Please run this script from the diffusion-audio-restoration directory")
        sys.exit(1)
    
    # Check if checkpoints exist
    checkpoint_dir = "pretrained/1"
    if not os.path.exists(checkpoint_dir):
        print("❌ Error: Pretrained checkpoints not found. Please download them first.")
        sys.exit(1)
    
    # Check if config is updated
    config_file = "configs/ensemble_2split_sampling.yaml"
    with open(config_file, 'r') as f:
        config_content = f.read()
        if "PATH/TO/" in config_content:
            print("❌ Error: Configuration file still contains placeholder paths")
            sys.exit(1)
    
    print("✅ All prerequisites check passed")
    print("\nStarting bandwidth extension test...")
    
    success = test_bandwidth_extension()
    
    if success:
        print("\n🎉 A2SB model is working correctly!")
        print("You can now use the model for audio restoration.")
        print("\n📁 Default export folder is set to:")
        print("/Users/gjb/_sound/_novaera/_samples/difussion audio restoration")
        print("\n💡 Usage examples:")
        print("  # Uses default export folder:")
        print("  python A2SB_upsample_api.py -f input.wav -o output.wav -n 50")
        print("  # Uses custom path:")
        print("  python A2SB_upsample_api.py -f input.wav -o /custom/path/output.wav -n 50")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()