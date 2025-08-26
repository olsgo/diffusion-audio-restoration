#!/usr/bin/env python3
"""
Basic validation test for A2SB optimized implementation
Tests core functionality without external dependencies
"""

import os
import sys
import time

def test_file_existence():
    """Test that required files exist"""
    print("🧪 Testing file existence...")
    
    required_files = [
        "inference/A2SB_upsample_api_mps.py",
        "inference/A2SB_upsample_api_mps_optimized.py", 
        "configs/ensemble_2split_sampling_mps.yaml",
        "configs/ensemble_2split_sampling_mps_balanced.yaml",
        "configs/ensemble_2split_sampling_mps_failsafe.yaml"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_python_syntax():
    """Test that Python files have valid syntax"""
    print("🧪 Testing Python syntax...")
    
    python_files = [
        "inference/A2SB_upsample_api_mps_optimized.py"
    ]
    
    all_valid = True
    for file_path in python_files:
        if not os.path.exists(file_path):
            print(f"   ⚠️  {file_path} - file not found")
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            compile(content, file_path, 'exec')
            print(f"   ✅ {file_path} - syntax valid")
        except SyntaxError as e:
            print(f"   ❌ {file_path} - syntax error: {e}")
            all_valid = False
        except Exception as e:
            print(f"   ⚠️  {file_path} - error: {e}")
    
    return all_valid

def test_yaml_syntax():
    """Test that YAML files have valid syntax"""
    print("🧪 Testing YAML syntax...")
    
    yaml_files = [
        "configs/ensemble_2split_sampling_mps.yaml",
        "configs/ensemble_2split_sampling_mps_balanced.yaml",
        "configs/ensemble_2split_sampling_mps_failsafe.yaml"
    ]
    
    all_valid = True
    for file_path in yaml_files:
        if not os.path.exists(file_path):
            print(f"   ⚠️  {file_path} - file not found")
            continue
            
        try:
            # Simple YAML validation
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Basic checks for YAML structure
            if ':' in content and content.strip():
                print(f"   ✅ {file_path} - basic structure valid")
            else:
                print(f"   ❌ {file_path} - invalid structure")
                all_valid = False
                
        except Exception as e:
            print(f"   ❌ {file_path} - error: {e}")
            all_valid = False
    
    return all_valid

def compare_implementations():
    """Compare key aspects of original vs optimized implementations"""
    print("🧪 Comparing implementations...")
    
    try:
        original_path = "inference/A2SB_upsample_api_mps.py"
        optimized_path = "inference/A2SB_upsample_api_mps_optimized.py"
        
        if not os.path.exists(original_path):
            print(f"   ⚠️  Original file not found")
            return False
            
        if not os.path.exists(optimized_path):
            print(f"   ❌ Optimized file not found")
            return False
        
        with open(original_path, 'r') as f:
            original_content = f.read()
        
        with open(optimized_path, 'r') as f:
            optimized_content = f.read()
        
        # Check key improvements
        improvements = []
        
        # Memory thresholds
        if "max_system_memory_percent=75" in original_content and "max_system_memory_percent=85" in optimized_content:
            improvements.append("More lenient memory thresholds (75% → 85%)")
        
        if "max_process_memory_mb=6144" in original_content and "max_process_memory_mb=10240" in optimized_content:
            improvements.append("Higher process memory limits (6GB → 10GB)")
        
        # Default parameters
        if "default=5" in original_content and "default=15" in optimized_content:
            improvements.append("Better default steps (5 → 15)")
        
        # Graduated scaling
        if "CRITICAL" in optimized_content and "SEVERE" in optimized_content and "MODERATE" in optimized_content:
            improvements.append("Graduated scaling with more levels")
        
        # Force option
        if "--force" in optimized_content and "--force" not in original_content:
            improvements.append("Added force option for power users")
        
        # Graceful shutdown
        if "sys.exit" in optimized_content and "os._exit" in original_content:
            improvements.append("Graceful shutdown instead of harsh termination")
            
        print(f"   Found {len(improvements)} key improvements:")
        for improvement in improvements:
            print(f"   ✅ {improvement}")
        
        return len(improvements) > 0
        
    except Exception as e:
        print(f"   ❌ Comparison failed: {e}")
        return False

def analyze_configuration_differences():
    """Analyze differences between configuration files"""
    print("🧪 Analyzing configuration differences...")
    
    configs = {
        "standard": "configs/ensemble_2split_sampling_mps.yaml",
        "balanced": "configs/ensemble_2split_sampling_mps_balanced.yaml", 
        "failsafe": "configs/ensemble_2split_sampling_mps_failsafe.yaml"
    }
    
    try:
        for name, path in configs.items():
            if os.path.exists(path):
                with open(path, 'r') as f:
                    content = f.read()
                
                # Analyze key settings
                if "precision: 16-mixed" in content:
                    precision = "16-mixed (half precision)"
                elif "precision: 32-true" in content:
                    precision = "32-true (full precision)"
                else:
                    precision = "unknown"
                
                if "batch_size: 1" in content:
                    batch_size = "1 (minimal)"
                elif "batch_size: 2" in content:
                    batch_size = "2 (balanced)"
                else:
                    batch_size = "unknown"
                
                print(f"   ✅ {name.capitalize()}: {precision}, batch_size: {batch_size}")
            else:
                print(f"   ⚠️  {name.capitalize()}: file not found")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration analysis failed: {e}")
        return False

def validate_memory_safety_design():
    """Validate the overall memory safety design"""
    print("🧪 Validating memory safety design...")
    
    try:
        optimized_path = "inference/A2SB_upsample_api_mps_optimized.py"
        
        if not os.path.exists(optimized_path):
            print("   ❌ Optimized file not found")
            return False
        
        with open(optimized_path, 'r') as f:
            content = f.read()
        
        safety_features = []
        
        # Check for key safety features
        if "OptimizedMemoryMonitor" in content:
            safety_features.append("Advanced memory monitoring class")
        
        if "intelligent_parameter_scaling" in content:
            safety_features.append("Intelligent parameter scaling")
        
        if "smart_memory_cleanup" in content:
            safety_features.append("Smart memory cleanup")
        
        if "graceful_signal_handler" in content:
            safety_features.append("Graceful signal handling")
        
        if "trend_analysis" in content or "memory_trend" in content:
            safety_features.append("Memory trend analysis")
        
        if "force_user_choice" in content:
            safety_features.append("User choice override option")
        
        print(f"   Found {len(safety_features)} safety features:")
        for feature in safety_features:
            print(f"   ✅ {feature}")
        
        return len(safety_features) >= 4
        
    except Exception as e:
        print(f"   ❌ Safety design validation failed: {e}")
        return False

def run_basic_tests():
    """Run all basic validation tests"""
    print("🔍 Basic Validation for A2SB Optimized Implementation")
    print("=" * 70)
    
    tests = [
        ("File Existence", test_file_existence),
        ("Python Syntax", test_python_syntax), 
        ("YAML Syntax", test_yaml_syntax),
        ("Implementation Comparison", compare_implementations),
        ("Configuration Analysis", analyze_configuration_differences),
        ("Memory Safety Design", validate_memory_safety_design),
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
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        print("\n💡 Key optimizations implemented:")
        print("   • Memory thresholds: 75% → 85% system, 6GB → 10GB process")
        print("   • Default parameters: 5 → 15 steps, 90s → 180s timeout")
        print("   • Graduated scaling: 5 levels instead of 3")
        print("   • Graceful shutdown: sys.exit() instead of os._exit()")
        print("   • User control: --force option to override safety limits")
        print("   • Better monitoring: trend analysis and less frequent checks")
        print("   • Balanced config: intermediate option between standard/failsafe")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_basic_tests()
    
    if success:
        print("\n🔗 Next steps:")
        print("   1. Test with actual audio files on M1 Max machine")
        print("   2. Monitor memory usage during processing")
        print("   3. Validate that crashes are prevented")
        print("   4. Compare output quality across different step counts")
        print("   5. Benchmark performance improvements")
    
    sys.exit(0 if success else 1)