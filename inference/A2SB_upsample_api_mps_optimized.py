# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np 
import json
import argparse
import glob
from subprocess import Popen, PIPE, TimeoutExpired
import yaml
import time 
from datetime import datetime, timedelta
import shutil
import csv
from tqdm import tqdm

import librosa
import soundfile as sf
import psutil
import gc
import threading
import signal
import sys


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def save_yaml(data, prefix="configs/temp"):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd_num = np.random.rand()
    rnd_num = rnd_num - rnd_num % 0.000001
    file_name = f"{prefix}_{timestamp}_{rnd_num}.yaml"
    with open(file_name, 'w') as f:
        yaml.dump(data, f)
    return file_name


class OptimizedMemoryMonitor:
    """Balanced memory monitoring to prevent crashes while allowing normal operation"""
    def __init__(self, max_system_memory_percent=85, max_process_memory_mb=10240, 
                 max_swap_percent=70, monitoring_interval=5):
        # More reasonable thresholds for M1 Max with 64GB RAM
        self.max_system_memory_percent = max_system_memory_percent  # 85% vs 75%
        self.max_process_memory_mb = max_process_memory_mb  # 10GB vs 6GB  
        self.max_swap_percent = max_swap_percent  # 70% vs 50%
        self.monitoring_interval = monitoring_interval  # 5s vs 2s
        self.monitoring = False
        self.monitor_thread = None
        self.memory_trend = []  # Track memory usage over time
        
    def start_monitoring(self):
        """Start continuous memory monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            
    def _monitor_loop(self):
        """Continuous monitoring loop with trend analysis"""
        while self.monitoring:
            try:
                self.check_memory_with_trend()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                print(f"Memory monitor error: {e}")
                time.sleep(10)
                
    def check_memory_with_trend(self):
        """Check memory usage with trend analysis for smarter decisions"""
        # System memory check
        system_memory = psutil.virtual_memory()
        system_used_percent = system_memory.percent
        
        # Process memory check
        process = psutil.Process(os.getpid())
        process_memory_mb = process.memory_info().rss / 1024 ** 2
        
        # Swap usage check
        swap = psutil.swap_memory()
        swap_used_percent = swap.percent if swap.total > 0 else 0
        
        # Track memory trend (keep last 10 readings)
        current_reading = {
            'time': time.time(),
            'system_percent': system_used_percent,
            'process_mb': process_memory_mb,
            'swap_percent': swap_used_percent
        }
        self.memory_trend.append(current_reading)
        if len(self.memory_trend) > 10:
            self.memory_trend.pop(0)
        
        # Analyze trend
        rapid_increase = self._is_memory_increasing_rapidly()
        
        # Log current state less frequently
        if len(self.memory_trend) % 6 == 1:  # Every 30 seconds at 5s intervals
            print(f"[{time.strftime('%H:%M:%S')}] Memory: SYS:{system_used_percent:.1f}% | PROC:{process_memory_mb:.0f}MB | SWAP:{swap_used_percent:.1f}%{' ⚠️RISING' if rapid_increase else ''}")
        
        # CRITICAL: Emergency shutdown conditions (more lenient)
        critical_system = system_used_percent > self.max_system_memory_percent
        critical_process = process_memory_mb > self.max_process_memory_mb
        critical_swap = swap_used_percent > self.max_swap_percent
        
        if critical_system or critical_process or critical_swap:
            print(f"\n🚨 EMERGENCY SHUTDOWN TRIGGERED 🚨")
            print(f"System Memory: {system_used_percent:.1f}% (limit: {self.max_system_memory_percent}%)")
            print(f"Process Memory: {process_memory_mb:.0f}MB (limit: {self.max_process_memory_mb}MB)")
            print(f"Swap Usage: {swap_used_percent:.1f}% (limit: {self.max_swap_percent}%)")
            print(f"Shutting down gracefully to prevent system crash...")
            
            # Graceful shutdown instead of harsh os._exit()
            self._graceful_shutdown()
            
        # WARNING: Approaching limits or rapid increase
        elif (system_used_percent > self.max_system_memory_percent - 10 or
              process_memory_mb > self.max_process_memory_mb - 2048 or
              rapid_increase):
            if rapid_increase:
                print(f"⚠️  WARNING: Rapid memory increase detected - monitor closely")
            else:
                print(f"⚠️  WARNING: Approaching memory limits")
            
    def _is_memory_increasing_rapidly(self):
        """Detect if memory usage is increasing rapidly"""
        if len(self.memory_trend) < 5:
            return False
        
        recent = self.memory_trend[-3:]
        old = self.memory_trend[-6:-3] if len(self.memory_trend) >= 6 else self.memory_trend[:-3]
        
        if not old:
            return False
        
        recent_avg = sum(r['system_percent'] for r in recent) / len(recent)
        old_avg = sum(r['system_percent'] for r in old) / len(old)
        
        # Consider rapid if >10% increase in ~15-30 seconds
        return recent_avg - old_avg > 10
            
    def _graceful_shutdown(self):
        """Graceful shutdown with cleanup"""
        try:
            print("Performing emergency cleanup...")
            gc.collect()
            # Try to free MPS cache
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
            # Exit gracefully instead of os._exit(1)
            sys.exit(1)
        except Exception:
            # If graceful fails, then use os._exit as last resort
            os._exit(1)


def smart_memory_cleanup(stage="", force=False):
    """Intelligent memory cleanup that adapts to current usage"""
    system_memory = psutil.virtual_memory()
    
    # Only do aggressive cleanup if memory usage is actually high
    if system_memory.percent > 70 or force:
        print(f"🧹 Memory cleanup [{stage}] - {system_memory.percent:.1f}% usage...")
        
        # Multiple rounds of garbage collection
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                print(f"   Round {i+1}: Collected {collected} objects")
        
        # Clear MPS cache if available
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
        except Exception:
            pass
        
        # Memory trimming (best effort)
        try:
            import ctypes
            libc = ctypes.CDLL("libc.dylib")
            libc.malloc_trim(0)
        except Exception:
            pass
    else:
        # Light cleanup for normal conditions
        gc.collect()


def intelligent_parameter_scaling(requested_steps, requested_timeout, force_user_choice=False):
    """Intelligently scale parameters based on memory state and user intent"""
    system_memory = psutil.virtual_memory()
    available_gb = system_memory.available / 1024 ** 3
    memory_percent = system_memory.percent
    
    # Define graduated scaling levels instead of just 3
    if memory_percent > 90 or available_gb < 2:
        # CRITICAL: Extreme memory pressure
        config_file = "configs/ensemble_2split_sampling_mps_failsafe.yaml"
        max_steps = 3
        max_timeout = 60
        urgency = "🚨 CRITICAL"
        recommended_steps = min(requested_steps, max_steps)
        recommended_timeout = min(requested_timeout, max_timeout)
        
    elif memory_percent > 85 or available_gb < 4:
        # SEVERE: High memory pressure
        config_file = "configs/ensemble_2split_sampling_mps_failsafe.yaml"
        max_steps = 5
        max_timeout = 90
        urgency = "🔴 SEVERE"
        recommended_steps = min(requested_steps, max_steps)
        recommended_timeout = min(requested_timeout, max_timeout)
        
    elif memory_percent > 75 or available_gb < 6:
        # HIGH: Moderate memory pressure
        config_file = "configs/ensemble_2split_sampling_mps_balanced.yaml"
        max_steps = 15
        max_timeout = 150
        urgency = "🟠 HIGH"
        recommended_steps = min(requested_steps, max_steps)
        recommended_timeout = min(requested_timeout, max_timeout)
        
    elif memory_percent > 65 or available_gb < 8:
        # MODERATE: Some memory pressure
        config_file = "configs/ensemble_2split_sampling_mps.yaml"
        max_steps = 20
        max_timeout = 180
        urgency = "🟡 MODERATE"
        recommended_steps = min(requested_steps, max_steps)
        recommended_timeout = min(requested_timeout, max_timeout)
        
    else:
        # NORMAL: Good memory conditions
        config_file = "configs/ensemble_2split_sampling_mps.yaml"
        max_steps = 50  # Allow user's full choice
        max_timeout = 300
        urgency = "✅ NORMAL"
        recommended_steps = requested_steps
        recommended_timeout = requested_timeout
    
    # Handle user choice override
    if force_user_choice and urgency not in ["🚨 CRITICAL", "🔴 SEVERE"]:
        print(f"\n🎛️  FORCE MODE: Using user parameters despite {urgency} memory state")
        recommended_steps = requested_steps
        recommended_timeout = requested_timeout
    
    # Show scaling decision
    print(f"\n📊 MEMORY STATE: {urgency}")
    print(f"   Available: {available_gb:.1f}GB | Usage: {memory_percent:.1f}%")
    print(f"   Configuration: {config_file}")
    
    if recommended_steps != requested_steps or recommended_timeout != requested_timeout:
        print(f"   🔧 AUTO-SCALED: Steps {requested_steps}→{recommended_steps}, Timeout {requested_timeout}s→{recommended_timeout}s")
    else:
        print(f"   ✅ USER CHOICE PRESERVED: {requested_steps} steps, {requested_timeout}s timeout")
    
    return config_file, recommended_steps, recommended_timeout


def enhanced_memory_logging(stage="", verbose=False):
    """Enhanced memory logging with optional verbosity"""
    if not verbose and stage not in ["start", "end", "error"]:
        return  # Skip intermediate logging unless verbose
        
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_mb = memory_info.rss / 1024 ** 2
    
    system_memory = psutil.virtual_memory()
    system_used_percent = system_memory.percent
    system_available_gb = system_memory.available / 1024 ** 3
    
    print(f"[{time.strftime('%H:%M:%S')}] Memory [{stage}]: Process={rss_mb:.1f}MB | System={system_used_percent:.1f}% | Available={system_available_gb:.1f}GB")


def optimized_shell_command(cmd, timeout_seconds=240, use_nice=True, verbose=False):
    """Optimized shell command execution with better resource management"""
    # More reasonable timeout limits
    timeout_seconds = min(timeout_seconds, 300)  # Max 5 minutes vs 90 seconds
    
    if use_nice:
        cmd = f"nice -n 10 {cmd}"  # Moderate priority vs extreme -n 19
    
    if verbose:
        print(f'Running: {cmd}')
    
    # Balanced environment settings
    env = os.environ.copy()
    # Allow more threads but still limit
    env["OMP_NUM_THREADS"] = "2"  # vs "1"
    env["MKL_NUM_THREADS"] = "2"
    env["VECLIB_MAXIMUM_THREADS"] = "4"  # vs "1"
    
    # MPS settings - less aggressive
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.3"  # vs "0.0" (allow some caching)
    env["PYTORCH_MPS_ALLOCATOR_POLICY"] = "native"  # vs "garbage_collection"
    
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, env=env)
    try:
        stdout, stderr = p.communicate(timeout=timeout_seconds)
    except TimeoutExpired:
        try:
            p.kill()
        except Exception:
            pass
        stdout, stderr = p.communicate()
        print(f"Process timed out after {timeout_seconds}s - this may indicate insufficient resources")
        if stderr:
            print(f"Error: {stderr.decode(errors='ignore')}")
        return 124

    if p.returncode != 0:
        print(f"Error: {stderr.decode(errors='ignore')}")
    elif verbose:
        print(stdout.decode(errors='ignore'))
    return p.returncode


def compute_rolloff_freq(audio_file, roll_percent=0.99):
    y, sr = librosa.load(audio_file, sr=None)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)[0]
    print(f"{int(roll_percent*100)} percent rolloff: {int(np.median(rolloff))}")
    return int(np.median(rolloff))


def cleanup_old_temp_configs(configs_dir="configs", max_age_minutes=60):
    """Remove stale temporary YAMLs"""
    cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
    if not os.path.isdir(configs_dir):
        return
    try:
        for name in os.listdir(configs_dir):
            if not name.startswith("temp_") or not name.endswith(".yaml"):
                continue
            full_path = os.path.join(configs_dir, name)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(full_path))
                if mtime < cutoff_time:
                    os.remove(full_path)
            except Exception:
                pass
    except Exception:
        pass


def ensure_free_space(path: str, min_free_gb: float = 1.5):
    """Ensure sufficient free disk space"""
    try:
        usage = shutil.disk_usage(os.path.abspath(path))
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_free_gb:
            raise RuntimeError(f"Insufficient free disk space: {free_gb:.2f} GB available, require at least {min_free_gb:.2f} GB")
    except FileNotFoundError:
        parent = os.path.dirname(os.path.abspath(path)) or "/"
        usage = shutil.disk_usage(parent)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_free_gb:
            raise RuntimeError(f"Insufficient free disk space at {parent}: {free_gb:.2f} GB available, require at least {min_free_gb:.2f} GB")


# Global optimized memory monitor
_memory_monitor = OptimizedMemoryMonitor()


def graceful_signal_handler(signum, frame):
    """Handle signals gracefully"""
    print(f"\n🛑 Signal {signum} received - shutting down gracefully...")
    _memory_monitor.stop_monitoring()
    gc.collect()
    sys.exit(1)


def optimized_upsample_mps(audio_filename, output_audio_filename, predict_n_steps=15, 
                          timeout_seconds=180, min_free_gb=1.5, force_user_choice=False,
                          verbose=False, use_nice=True):
    """Optimized MPS upsampling with balanced safety and performance"""
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, graceful_signal_handler)
    signal.signal(signal.SIGINT, graceful_signal_handler)
    
    # Start memory monitoring
    print("🔍 Starting intelligent memory monitoring...")
    _memory_monitor.start_monitoring()
    
    try:
        enhanced_memory_logging(stage="start", verbose=verbose)

        # Intelligent parameter scaling
        optimal_config, final_steps, final_timeout = intelligent_parameter_scaling(
            predict_n_steps, timeout_seconds, force_user_choice
        )
        
        # Pre-flight checks
        ensure_free_space(".", min_free_gb)
        smart_memory_cleanup(stage="initialization")
        
        # Load base configuration and modify
        inference_config = load_yaml('configs/inference_files_upsampling.yaml')
        abs_audio_filename = os.path.abspath(audio_filename)
        inference_config['data']['predict_filelist'] = [{
            'filepath': abs_audio_filename,
            'output_subdir': '.'
        }]

        enhanced_memory_logging(stage="before_rolloff", verbose=verbose)
        
        # Compute rolloff frequency
        cutoff_freq = compute_rolloff_freq(abs_audio_filename)
        inference_config['data']['transforms_aug'][0]['init_args']['upsample_mask_kwargs'] = {
            'min_cutoff_freq': cutoff_freq,
            'max_cutoff_freq': cutoff_freq
        }

        # Save temporary config
        temporary_yaml_file = save_yaml(inference_config)
        
        enhanced_memory_logging(stage="before_processing", verbose=verbose)
        smart_memory_cleanup(stage="before_processing")

        # Build processing command
        cmd = f"conda run -n a2sb python ensembled_inference_api.py predict -c {optimal_config} -c {temporary_yaml_file} --model.predict_n_steps={final_steps} --model.output_audio_filename='{output_audio_filename}'"
        
        print(f"\n=== Optimized A2SB MPS Processing ===")
        print(f"Input: {audio_filename}")
        print(f"Output: {output_audio_filename}")
        print(f"Steps: {final_steps}")
        print(f"Rolloff frequency: {cutoff_freq} Hz")
        print(f"Timeout: {final_timeout}s")
        print("Using optimized Apple Silicon MPS acceleration...\n")
        
        # Execute processing
        result = optimized_shell_command(cmd, timeout_seconds=final_timeout, 
                                       use_nice=use_nice, verbose=verbose)
        
        # Post-processing cleanup
        smart_memory_cleanup(stage="after_processing")
        
        # Remove temporary file
        try:
            os.remove(temporary_yaml_file)
        except Exception:
            pass
        
        enhanced_memory_logging(stage="end", verbose=verbose)
        
        # Report results
        if result == 0:
            print(f"\n✅ Processing completed successfully!")
            print(f"📁 Output saved: {output_audio_filename}")
        elif result == 124:
            print(f"\n⏱️ Processing timed out after {final_timeout}s")
            print("💡 Try reducing steps or splitting the audio file")
        else:
            print(f"\n❌ Processing failed with exit code: {result}")
            
    except Exception as e:
        enhanced_memory_logging(stage="error", verbose=True)
        print(f"\n🚨 ERROR: {e}")
        raise
    finally:
        print("🔍 Stopping memory monitoring...")
        _memory_monitor.stop_monitoring()
        smart_memory_cleanup(stage="final_cleanup", force=True)


def main():
    cleanup_old_temp_configs()
    
    parser = argparse.ArgumentParser(
        description="Optimized A2SB MPS Audio Restoration with Intelligent Memory Management"
    )
    parser.add_argument("-f", "--audio_filename", help="Input audio file", required=True)
    parser.add_argument("-o", "--output_audio_filename", default="a2sb_output.wav",
                       help="Output audio file")
    parser.add_argument("-n", "--predict_n_steps", type=int, default=15,
                       help="Number of diffusion steps (default: 15, balanced quality/speed)")
    parser.add_argument("--timeout_seconds", type=int, default=180,
                       help="Max seconds for processing (default: 180)")
    parser.add_argument("--min_free_gb", type=float, default=1.5,
                       help="Required free disk space (GB)")
    parser.add_argument("--force", action="store_true",
                       help="Force user parameters even with memory pressure (risky)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--no_nice", action="store_true",
                       help="Disable CPU priority lowering")

    args = parser.parse_args()

    optimized_upsample_mps(
        audio_filename=args.audio_filename,
        output_audio_filename=args.output_audio_filename,
        predict_n_steps=args.predict_n_steps,
        timeout_seconds=args.timeout_seconds,
        min_free_gb=args.min_free_gb,
        force_user_choice=args.force,
        verbose=args.verbose,
        use_nice=not args.no_nice
    )


if __name__ == "__main__":
    main()

# Example usage:
# python A2SB_upsample_api_mps_optimized.py -f input.mp3 -o output.wav -n 25
# python A2SB_upsample_api_mps_optimized.py -f input.mp3 -o output.wav -n 50 --force --verbose