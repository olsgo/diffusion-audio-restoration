# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for A2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os

# # If there is Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
# os.environ["MKL_THREADING_LAYER"] = "GNU"
# import numpy as np
# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

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


class SystemMemoryMonitor:
    """Aggressive system memory monitoring to prevent crashes"""
    def __init__(self, max_system_memory_percent=75, max_process_memory_mb=6144):
        self.max_system_memory_percent = max_system_memory_percent
        self.max_process_memory_mb = max_process_memory_mb
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start continuous memory monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring:
            try:
                self.check_memory_critical()
                time.sleep(2)  # Check every 2 seconds
            except Exception as e:
                print(f"Memory monitor error: {e}")
                time.sleep(5)
                
    def check_memory_critical(self):
        """Check if memory usage is approaching dangerous levels"""
        # System memory check
        system_memory = psutil.virtual_memory()
        system_used_percent = system_memory.percent
        
        # Process memory check
        process = psutil.Process(os.getpid())
        process_memory_mb = process.memory_info().rss / 1024 ** 2
        
        # Swap usage check (critical indicator)
        swap = psutil.swap_memory()
        swap_used_percent = swap.percent if swap.total > 0 else 0
        
        # Log current state
        print(f"[{time.strftime('%H:%M:%S')}] SYS: {system_used_percent:.1f}% | PROC: {process_memory_mb:.0f}MB | SWAP: {swap_used_percent:.1f}%")
        
        # CRITICAL: Emergency shutdown conditions
        if (system_used_percent > self.max_system_memory_percent or 
            process_memory_mb > self.max_process_memory_mb or
            swap_used_percent > 50):  # High swap usage is dangerous
            
            print(f"\n🚨 EMERGENCY SHUTDOWN TRIGGERED 🚨")
            print(f"System Memory: {system_used_percent:.1f}% (limit: {self.max_system_memory_percent}%)")
            print(f"Process Memory: {process_memory_mb:.0f}MB (limit: {self.max_process_memory_mb}MB)")
            print(f"Swap Usage: {swap_used_percent:.1f}% (danger threshold: 50%)")
            print(f"Terminating immediately to prevent system crash...")
            
            # Force immediate cleanup and exit
            self._emergency_cleanup()
            os._exit(1)
            
        # WARNING: Approaching limits
        elif (system_used_percent > self.max_system_memory_percent - 10 or
              process_memory_mb > self.max_process_memory_mb - 1024):
            print(f"⚠️  WARNING: Approaching memory limits - consider stopping processing")
            
    def _emergency_cleanup(self):
        """Emergency cleanup before shutdown"""
        try:
            gc.collect()  # Force garbage collection
            # Try to clear any cached data
            if hasattr(psutil.Process(), 'memory_full_info'):
                psutil.Process().memory_full_info()  # Force memory info update
        except Exception:
            pass

# Global memory monitor instance
_memory_monitor = SystemMemoryMonitor()

def aggressive_memory_cleanup(stage=""):
    """Perform aggressive memory cleanup and garbage collection"""
    print(f"🧹 Aggressive memory cleanup [{stage}]...")
    
    # Multiple rounds of garbage collection
    for i in range(3):
        collected = gc.collect()
        if collected > 0:
            print(f"   Round {i+1}: Collected {collected} objects")
    
    # Force memory trimming if available
    try:
        import ctypes
        libc = ctypes.CDLL("libc.dylib")
        libc.malloc_trim(0)  # Trim malloc heap
    except Exception:
        pass
    
    # Clear any cached data
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()  # Clear MPS cache
            torch.mps.synchronize()  # Ensure operations complete
    except Exception:
        pass
    
    # Log memory state after cleanup
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / 1024 ** 2
    print(f"   Memory after cleanup: {rss_mb:.1f}MB")

def check_system_memory_pressure():
    """Check if system is under severe memory pressure"""
    system_memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    # Critical indicators of memory pressure
    high_memory_usage = system_memory.percent > 85
    high_swap_usage = swap.percent > 30 if swap.total > 0 else False
    low_available = system_memory.available < (2 * 1024 ** 3)  # Less than 2GB available
    
    if high_memory_usage or high_swap_usage or low_available:
        print(f"\n⚠️  SEVERE MEMORY PRESSURE DETECTED:")
        print(f"   System Memory: {system_memory.percent:.1f}% used")
        print(f"   Available: {system_memory.available / 1024**3:.2f}GB")
        print(f"   Swap: {swap.percent:.1f}% used")
        print(f"   RECOMMENDATION: Stop all non-essential processes immediately")
        return True
    
    return False

def auto_scale_parameters_for_memory():
    """Automatically scale processing parameters based on available memory"""
    system_memory = psutil.virtual_memory()
    available_gb = system_memory.available / 1024 ** 3
    memory_percent = system_memory.percent
    
    # Determine configuration and parameters based on memory state
    if memory_percent > 80 or available_gb < 4:
        # CRITICAL: Use fail-safe configuration
        config_file = "configs/ensemble_2split_sampling_mps_failsafe.yaml"
        max_steps = 3
        timeout = 60
        print(f"🚨 CRITICAL MEMORY STATE: Using fail-safe configuration")
        print(f"   Available: {available_gb:.1f}GB | Usage: {memory_percent:.1f}%")
        print(f"   Max steps: {max_steps} | Timeout: {timeout}s")
    elif memory_percent > 70 or available_gb < 8:
        # WARNING: Use conservative settings
        config_file = "configs/ensemble_2split_sampling_mps.yaml"
        max_steps = 5
        timeout = 90
        print(f"⚠️  HIGH MEMORY USAGE: Using conservative settings")
        print(f"   Available: {available_gb:.1f}GB | Usage: {memory_percent:.1f}%")
        print(f"   Max steps: {max_steps} | Timeout: {timeout}s")
    else:
        # NORMAL: Use standard MPS configuration
        config_file = "configs/ensemble_2split_sampling_mps.yaml"
        max_steps = 10
        timeout = 120
        print(f"✅ NORMAL MEMORY STATE: Using standard MPS settings")
        print(f"   Available: {available_gb:.1f}GB | Usage: {memory_percent:.1f}%")
        print(f"   Max steps: {max_steps} | Timeout: {timeout}s")
    
    return config_file, max_steps, timeout

def log_memory_usage(stage=""):
    """Enhanced memory logging with system-wide monitoring"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_mb = memory_info.rss / 1024 ** 2
    vms_mb = memory_info.vms / 1024 ** 2
    
    # System memory info
    system_memory = psutil.virtual_memory()
    system_used_percent = system_memory.percent
    system_available_gb = system_memory.available / 1024 ** 3
    
    # Swap info
    swap = psutil.swap_memory()
    swap_used_percent = swap.percent if swap.total > 0 else 0
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] MEMORY [{stage}]:")
    print(f"  Process: RSS={rss_mb:.2f}MB, VMS={vms_mb:.2f}MB")
    print(f"  System: {system_used_percent:.1f}% used, {system_available_gb:.2f}GB available")
    print(f"  Swap: {swap_used_percent:.1f}% used")
    
    # Force garbage collection at each checkpoint
    gc.collect()
    
    # Check for immediate danger
    _memory_monitor.check_memory_critical()

def cleanup_old_temp_configs(configs_dir="configs", max_age_minutes=60):
    """Remove stale temporary YAMLs created by this script to reclaim disk space."""
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
                # Best-effort cleanup; ignore errors
                pass
    except Exception:
        pass


def ensure_free_space(path: str, min_free_gb: float = 1.5):
    """Ensure there's sufficient free disk space at the filesystem containing path."""
    try:
        usage = shutil.disk_usage(os.path.abspath(path))
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_free_gb:
            raise RuntimeError(f"Insufficient free disk space: {free_gb:.2f} GB available, require at least {min_free_gb:.2f} GB")
    except FileNotFoundError:
        # If path doesn't exist yet, check its parent
        parent = os.path.dirname(os.path.abspath(path)) or "/"
        usage = shutil.disk_usage(parent)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_free_gb:
            raise RuntimeError(f"Insufficient free disk space at {parent}: {free_gb:.2f} GB available, require at least {min_free_gb:.2f} GB")


def shell_run_cmd(cmd, timeout_seconds: int = 240, use_nice: bool = True):
    # EMERGENCY: Ultra-conservative timeout to prevent long-running processes
    timeout_seconds = min(timeout_seconds, 90)  # Max 90 seconds
    
    # Force low priority scheduling
    if use_nice:
        cmd = f"nice -n 19 {cmd}"  # Lowest priority
    print('running:', cmd)
    
    # EMERGENCY: Aggressive memory and resource limits
    env = os.environ.copy()
    # Thread limits - single threaded to minimize memory
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    
    # MPS memory controls - CRITICAL for preventing crashes
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable MPS caching completely
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    env["PYTORCH_MPS_ALLOCATOR_POLICY"] = "garbage_collection"  # Aggressive cleanup
    
    # Additional memory pressure controls
    env["MALLOC_TRIM_THRESHOLD_"] = "65536"  # Aggressive malloc trimming
    env["PYTHONHASHSEED"] = "0"  # Deterministic to reduce memory fragmentation
    
    # Force minimal precision and memory usage
    env["PYTORCH_JIT"] = "0"  # Disable JIT compilation to save memory
    env["PYTORCH_TENSORPIPE_INIT_METHOD"] = "env://"
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, env=env)
    try:
        stdout, stderr = p.communicate(timeout=timeout_seconds)
    except TimeoutExpired:
        # Kill the process tree on timeout to protect system resources
        try:
            p.kill()
        except Exception:
            pass
        stdout, stderr = p.communicate()
        print(f"Process timed out after {timeout_seconds}s and was terminated to protect system resources.")
        if stderr:
            print(f"Error: {stderr.decode(errors='ignore')}")
        return 124  # Common timeout exit code

    if p.returncode != 0:
        print(f"Error: {stderr.decode(errors='ignore')}")
    else:
        print(stdout.decode(errors='ignore'))
    return p.returncode


def compute_rolloff_freq(audio_file, roll_percent=0.99):
    y, sr = librosa.load(audio_file, sr=None)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)[0]
    print(f"{int(roll_percent*100)} percent rolloff: {int(np.median(rolloff))}")
    return int(np.median(rolloff))


def emergency_signal_handler(signum, frame):
    """Handle emergency signals to prevent system crash"""
    print(f"\n🚨 EMERGENCY SIGNAL {signum} RECEIVED - TERMINATING IMMEDIATELY 🚨")
    _memory_monitor.stop_monitoring()
    gc.collect()
    os._exit(1)

def upsample_one_sample_mps(audio_filename, output_audio_filename, predict_n_steps=25, timeout_seconds: int = 240, min_free_gb: float = 1.5, cleanup_temp_minutes: int = 120, use_nice: bool = True,
                            no_nice: bool = False,
                            ):
    # Set up emergency signal handlers
    signal.signal(signal.SIGTERM, emergency_signal_handler)
    signal.signal(signal.SIGINT, emergency_signal_handler)
    
    # Start aggressive memory monitoring
    print("🔍 Starting aggressive memory monitoring to prevent system crashes...")
    _memory_monitor.start_monitoring()
    
    try:
        log_memory_usage(stage="start_upsample_one_sample_mps")

        # AUTO-SCALE: Automatically adjust parameters based on memory state
        original_steps = predict_n_steps
        original_timeout = timeout_seconds
        
        # Get optimal configuration and parameters for current memory state
        optimal_config, max_safe_steps, max_safe_timeout = auto_scale_parameters_for_memory()
        
        predict_n_steps = min(predict_n_steps, max_safe_steps)
        timeout_seconds = min(timeout_seconds, max_safe_timeout)
        
        print(f"\n🛡️  AUTO-SCALED SAFETY MODE:")
        print(f"   Configuration: {optimal_config}")
        print(f"   Steps: {original_steps} → {predict_n_steps} (auto-scaled)")
        print(f"   Timeout: {original_timeout}s → {timeout_seconds}s (auto-scaled)")
        print(f"   Memory monitoring: ACTIVE (75% system limit, 6GB process limit)")
        print(f"   Emergency shutdown: ENABLED (prevents system freeze)")

        # 1. Pre-flight checks for disk space
        ensure_free_space(".", min_free_gb)
        log_memory_usage(stage="after_disk_check")
        aggressive_memory_cleanup(stage="after_disk_check")
        
        # Check for memory pressure before proceeding
        if check_system_memory_pressure():
            print("\n🚨 ABORTING: System under severe memory pressure")
            raise RuntimeError("System memory pressure too high - aborting to prevent crash")

        inference_config = load_yaml('configs/inference_files_upsampling.yaml')
        # Convert to absolute path to handle directory changes
        abs_audio_filename = os.path.abspath(audio_filename)
        inference_config['data']['predict_filelist'] = [{
            'filepath': abs_audio_filename,
            'output_subdir': '.'
        }]

        log_memory_usage(stage="before_rolloff_computation")
        aggressive_memory_cleanup(stage="before_rolloff_computation")
        
        cutoff_freq = compute_rolloff_freq(abs_audio_filename, roll_percent=0.99)
        
        log_memory_usage(stage="after_rolloff_computation")
        aggressive_memory_cleanup(stage="after_rolloff_computation")
        
        inference_config['data']['transforms_aug'][0]['init_args']['upsample_mask_kwargs'] = {
            'min_cutoff_freq': cutoff_freq,
            'max_cutoff_freq': cutoff_freq
        }
        
        # EMERGENCY: Ultra-minimal resource usage to prevent crashes
        inference_config['data']['batch_size'] = 1  # Absolute minimum
        inference_config['data']['num_workers'] = 0  # No parallel workers
        inference_config['data']['persistent_workers'] = False  # No worker persistence
        inference_config['data']['pin_memory'] = False  # Disable memory pinning
        
        # Force absolute minimal memory usage
        if 'trainer' in inference_config:
            inference_config['trainer']['precision'] = '16-mixed'  # Half precision
            inference_config['trainer']['enable_checkpointing'] = False  # No checkpoints
            inference_config['trainer']['logger'] = False  # No logging
            inference_config['trainer']['enable_progress_bar'] = False  # No progress bar
        log_memory_usage(stage="after_config_setup")
        aggressive_memory_cleanup(stage="after_config_setup")
        
        # Final memory pressure check before processing
        if check_system_memory_pressure():
            print("\n🚨 ABORTING: Memory pressure increased during setup")
            raise RuntimeError("Memory pressure too high before processing - aborting")
        
        temporary_yaml_file = save_yaml(inference_config)

        # Use auto-selected configuration based on memory state
        cmd = "conda run -n a2sb python ensembled_inference_api.py predict \
                -c {} \
                -c {} \
                --model.predict_n_steps={} \
                --model.output_audio_filename='{}'".format(optimal_config, temporary_yaml_file, predict_n_steps, output_audio_filename)
        
        print(f"\n=== MPS-Optimized A2SB Processing ===")
        print(f"Input: {audio_filename}")
        print(f"Output: {output_audio_filename}")
        print(f"Steps: {predict_n_steps} (optimized for MPS)")
        print(f"Rolloff frequency: {cutoff_freq} Hz")
        print("Using Apple Silicon MPS acceleration...\n")
        
        # Final cleanup before intensive processing
        aggressive_memory_cleanup(stage="before_processing")
        
        result = shell_run_cmd(cmd, timeout_seconds=timeout_seconds, use_nice=use_nice)
        
        # Immediate cleanup after processing
        aggressive_memory_cleanup(stage="after_processing")
        
        # Always best-effort remove the temp yaml
        try:
            os.remove(temporary_yaml_file)
        except Exception:
            pass
        
        if result == 0:
            print(f"\n✅ Processing completed successfully!")
        elif result == 124:
            print(f"\n⚠️ Processing timed out after {timeout_seconds}s and was safely terminated.")
        else:
            print(f"\n❌ Processing failed with exit code: {result}")
            
    except Exception as e:
        print(f"\n🚨 CRITICAL ERROR: {e}")
        print("Emergency termination to protect system...")
        raise
    finally:
        # Always stop memory monitoring
        print("\n🔍 Stopping memory monitoring...")
        _memory_monitor.stop_monitoring()
        gc.collect()  # Final cleanup


def main():
    log_memory_usage(stage="start_main")
    cleanup_old_temp_configs()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--audio_filename", help="Input audio file", required=True
    )
    parser.add_argument(
        "-o",
        "--output_audio_filename",
        default="a2sb_output.wav",
        help="Output audio file",
    )
    parser.add_argument(
        "-n", "--predict_n_steps", type=int, default=5, help="Number of diffusion steps (ULTRA-SAFE: max 5 to prevent crashes)"
    )
    parser.add_argument(
        "--timeout_seconds",
        type=int,
        default=90,
        help="Max seconds to allow the prediction subprocess to run (ULTRA-SAFE: max 90s)",
    )
    parser.add_argument(
        "--min_free_gb",
        type=float,
        default=1.5,
        help="Required free disk space (GB) before starting",
    )
    parser.add_argument(
        "--no_nice",
        action="store_true",
        help="Disable lowering CPU scheduling priority for the subprocess",
    )

    args = parser.parse_args()

    upsample_one_sample_mps(
        audio_filename=args.audio_filename,
        output_audio_filename=args.output_audio_filename,
        predict_n_steps=args.predict_n_steps,
        timeout_seconds=args.timeout_seconds,
        min_free_gb=args.min_free_gb,
        use_nice=not args.no_nice,
    )

    log_memory_usage(stage="end_main")


if __name__ == "__main__":
    main()

    # Usage: python A2SB_upsample_api_mps.py -f <INPUT_FILENAME> -o <OUTPUT_FILENAME> -n <N_STEPS>
    # Example: python A2SB_upsample_api_mps.py -f input.mp3 -o output_mps.wav -n 15