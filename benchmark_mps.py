#!/usr/bin/env python3
"""
A2SB MPS Performance Benchmark Script
Optimized for Apple Silicon M1 Max
"""

import time
import os
import subprocess
import argparse
from datetime import datetime
import shutil
import psutil


def bytes_to_readable(num):
    for unit in ['B','KB','MB','GB','TB']:
        if num < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def print_resource_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    mem_mb = mem / (1024 ** 2)
    cpu = psutil.cpu_percent(interval=0.1)
    disk = shutil.disk_usage('.')
    print(f"[RES]{' ' + tag if tag else ''} | RAM: {bytes_to_readable(mem)} | CPU: {cpu:4.1f}% | Disk free: {bytes_to_readable(disk.free)}")
    
    # EMERGENCY: Monitor for dangerous memory usage
    if mem_mb > 4096:  # 4GB warning
        print(f"⚠️ WARNING: High memory usage detected: {mem_mb:.1f} MB")
    if mem_mb > 8192:  # 8GB emergency
        print(f"🚨 EMERGENCY: Dangerous memory usage: {mem_mb:.1f} MB - TERMINATING")
        os._exit(1)


def run_benchmark(audio_file, steps_list=[5, 8, 10], output_dir="benchmark_results"):
    """
    Benchmark A2SB performance with different step counts on MPS
    """
    
    # Create output directory in repo to avoid writing outside workspace
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 A2SB MPS Performance Benchmark")
    print(f"Audio file: {audio_file}")
    print(f"Testing steps: {steps_list}")
    print("=" * 50)
    print_resource_usage("start")
    
    results = []
    
    # Limit threads globally for child processes
    env_base = os.environ.copy()
    env_base.setdefault("OMP_NUM_THREADS", "1")
    env_base.setdefault("MKL_NUM_THREADS", "1")
    env_base.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env_base.setdefault("NUMEXPR_NUM_THREADS", "1")
    env_base.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    
    for steps in steps_list:
        print(f"\n📊 Testing with {steps} sampling steps...")
        print_resource_usage(f"before {steps}")
        
        output_file = os.path.join(output_dir, f"benchmark_{steps}steps.wav")
        
        # Ensure no stale temp configs explode in number
        for f in os.listdir("configs"):
            if f.startswith("temp_") and f.endswith(".yaml"):
                try:
                    os.remove(os.path.join("configs", f))
                except Exception:
                    pass
        
        start_time = time.time()
        cmd = [
            "python", "inference/A2SB_upsample_api_mps.py",
            "-f", audio_file,
            "-o", output_file,
            "-n", str(steps)
        ]
        try:
            # EMERGENCY: Use 60 second timeout - still safe but allows completion
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", env=env_base, timeout=60)
            end_time = time.time()
            processing_time = end_time - start_time
            print_resource_usage(f"after {steps}")
            
            if result.returncode == 0 and os.path.exists(output_file):
                print(f"✅ Completed in {processing_time:.1f} seconds")
                results.append({
                    'steps': steps,
                    'time': processing_time,
                    'status': 'success'
                })
            else:
                if result.returncode != 0:
                    print(f"❌ Failed with exit code {result.returncode}")
                    print(result.stderr.strip()[-1000:])
                else:
                    print("❌ Failed - output file not found")
                results.append({
                    'steps': steps,
                    'time': processing_time,
                    'status': 'failed'
                })
        except subprocess.TimeoutExpired:
            print(f"⏱️ Timeout at {steps} steps, killing process to protect resources")
            results.append({'steps': steps, 'time': 180, 'status': 'timeout'})
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append({'steps': steps, 'time': 0, 'status': f'exception: {e}'})
        finally:
            # Cleanup temp yaml files after each run
            for f in os.listdir("configs"):
                if f.startswith("temp_") and f.endswith(".yaml"):
                    try:
                        os.remove(os.path.join("configs", f))
                    except Exception:
                        pass
            print_resource_usage(f"cleanup {steps}")
    
    print("\n" + "=" * 50)
    print("📈 BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    for result in results:
        status = result['status']
        mark = "✅" if status == 'success' else "❌" if status != 'timeout' else "⏱️"
        print(f"Steps: {result['steps']:2d} | Time: {result['time']:6.1f}s | Status: {mark} {status}")
    
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        fastest = min(successful_results, key=lambda x: x['time'])
        print(f"\n🏆 Fastest successful run: {fastest['steps']} steps in {fastest['time']:.1f}s")
        print("\n💡 Recommendations:")
        print(f"   • Quick processing: {min(successful_results, key=lambda x: x['time'])['steps']} steps")
        if len(successful_results) > 1:
            balanced = sorted(successful_results, key=lambda x: x['steps'])[len(successful_results)//2]
            print(f"   • Balanced quality: {balanced['steps']} steps")
        best_quality = max(successful_results, key=lambda x: x['steps'])
        print(f"   • Best quality: {best_quality['steps']} steps")
    
    return results


def check_mps_availability():
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
        if torch.backends.mps.is_available():
            print("✅ MPS acceleration is ready!")
            return True
        else:
            print("❌ MPS not available")
            return False
    except ImportError:
        print("❌ PyTorch not found")
        return False


def main():
    parser = argparse.ArgumentParser(description='Benchmark A2SB MPS performance')
    parser.add_argument('-f', '--audio_file', type=str, required=True,
                       help='Audio file to process for benchmarking')
    parser.add_argument('-s', '--steps', type=int, nargs='+', default=[1, 3],
                       help='List of step counts to test (default: 1 3 - ULTRA SAFE MODE)')
    parser.add_argument('-o', '--output_dir', type=str, default='benchmark_results',
                       help='Output directory for benchmark results')
    parser.add_argument('--check-mps', action='store_true',
                       help='Only check MPS availability and exit')
    
    args = parser.parse_args()
    
    if args.check_mps:
        check_mps_availability()
        return
    
    print("🔍 Checking MPS availability...")
    if not check_mps_availability():
        print("\n❌ MPS not available. Please check your PyTorch installation.")
        return
    
    print("\n" + "=" * 60)
    print(f"Starting benchmark at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        return
    
    results = run_benchmark(args.audio_file, args.steps, args.output_dir)
    
    print(f"\n🎯 Benchmark completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {args.output_dir}/")

if __name__ == '__main__':
    main()

# Example usage:
# python benchmark_mps.py -f "test_audio.mp3" -s 10 15 20 25
# python benchmark_mps.py --check-mps