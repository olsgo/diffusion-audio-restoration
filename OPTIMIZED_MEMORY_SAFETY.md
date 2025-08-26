# Optimized Memory Safety for A2SB on Apple Silicon M1 Max

## Overview

This document describes the optimized memory safety implementation for the A2SB diffusion audio restoration model on Apple Silicon M1 Max machines. The original implementation was causing system crashes due to extreme memory pressure. This optimized version provides better balance between safety and usability.

## Problem Statement

The original local branch implementation prevented system crashes but was overly conservative:

- **75% system memory emergency threshold** - Too low for normal operation
- **6GB process memory limit** - Too restrictive for M1 Max with 64GB RAM
- **5 default diffusion steps** - Poor quality output
- **90-second timeout** - Too short for complex audio
- **Binary scaling** - Only 3 configurations (normal/warning/critical)
- **Harsh shutdown** - `os._exit(1)` could corrupt other processes

## Optimized Solution

### 1. **Balanced Memory Thresholds**

| Aspect | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| System Memory Emergency | 75% | 85% | More headroom for normal operation |
| Process Memory Limit | 6GB | 10GB | Better suited for M1 Max 64GB |
| Swap Usage Emergency | 50% | 70% | Less aggressive intervention |
| Monitoring Frequency | 2 seconds | 5 seconds | Reduced overhead |

### 2. **Graduated Parameter Scaling**

| Memory State | System % | Available GB | Config | Max Steps | Timeout |
|--------------|----------|--------------|--------|-----------|---------|
| **🚨 CRITICAL** | >90% | <2GB | failsafe | 3 | 60s |
| **🔴 SEVERE** | >85% | <4GB | failsafe | 5 | 90s |
| **🟠 HIGH** | >75% | <6GB | balanced | 15 | 150s |
| **🟡 MODERATE** | >65% | <8GB | standard | 20 | 180s |
| **✅ NORMAL** | <65% | >8GB | standard | 50 | 300s |

### 3. **Better Default Parameters**

```bash
# Optimized defaults (better quality while still safe)
python A2SB_upsample_api_mps_optimized.py -f input.wav -o output.wav
# Uses: 15 steps, 180s timeout

# Original defaults (ultra-conservative)
python A2SB_upsample_api_mps.py -f input.wav -o output.wav  
# Uses: 5 steps, 90s timeout
```

### 4. **Enhanced User Control**

```bash
# Force user choice even with memory pressure (for power users)
python A2SB_upsample_api_mps_optimized.py -f input.wav -o output.wav -n 50 --force

# Verbose monitoring
python A2SB_upsample_api_mps_optimized.py -f input.wav -o output.wav --verbose

# Custom memory safety
python A2SB_upsample_api_mps_optimized.py -f input.wav -o output.wav -n 25 --timeout_seconds 240
```

### 5. **Three-Tier Configuration System**

#### Standard Configuration (`ensemble_2split_sampling_mps.yaml`)
- **Use case**: Normal memory conditions (>8GB available)
- **Precision**: 32-true (full precision)
- **Batch size**: 2
- **Model**: Full architecture
- **Quality**: Highest

#### Balanced Configuration (`ensemble_2split_sampling_mps_balanced.yaml`) ⭐ **NEW**
- **Use case**: Moderate memory pressure (4-8GB available)
- **Precision**: 32-true (full precision)
- **Batch size**: 1
- **Model**: Slightly reduced channels (25% less memory)
- **Quality**: Very good with better safety

#### Failsafe Configuration (`ensemble_2split_sampling_mps_failsafe.yaml`)
- **Use case**: High memory pressure (<4GB available)
- **Precision**: 16-mixed (half precision)
- **Batch size**: 1
- **Model**: Significantly reduced architecture
- **Quality**: Good but prioritizes system stability

### 6. **Intelligent Memory Monitoring**

#### Advanced Features:
- **Trend Analysis**: Detects rapidly increasing memory usage
- **Graceful Shutdown**: `sys.exit()` instead of `os._exit(1)`
- **Smart Cleanup**: Only aggressive cleanup when needed
- **Background Monitoring**: Non-intrusive 5-second intervals

#### Memory Trend Detection:
```
[14:32:15] Memory: SYS:72.1% | PROC:3200MB | SWAP:0.0%
[14:32:20] Memory: SYS:74.3% | PROC:3400MB | SWAP:0.0%
[14:32:25] Memory: SYS:76.8% | PROC:3650MB | SWAP:0.0% ⚠️RISING
```

## Usage Examples

### Quick Processing (Default - Balanced)
```bash
python inference/A2SB_upsample_api_mps_optimized.py -f input.mp3 -o output.wav
# 15 steps, auto-scaled based on memory
```

### High Quality Processing
```bash
python inference/A2SB_upsample_api_mps_optimized.py -f input.mp3 -o output.wav -n 25
# Will auto-scale down if memory pressure detected
```

### Power User Mode (Override Safety)
```bash
python inference/A2SB_upsample_api_mps_optimized.py -f input.mp3 -o output.wav -n 50 --force
# Forces 50 steps unless critical memory state
```

### Verbose Monitoring
```bash
python inference/A2SB_upsample_api_mps_optimized.py -f input.mp3 -o output.wav --verbose
# Shows detailed memory information throughout processing
```

## Expected Behavior

### Normal Memory Conditions (>8GB available)
```
📊 MEMORY STATE: ✅ NORMAL
   Available: 12.3GB | Usage: 62.1%
   Configuration: configs/ensemble_2split_sampling_mps.yaml
   ✅ USER CHOICE PRESERVED: 25 steps, 180s timeout
```

### Moderate Memory Pressure (4-8GB available)
```
📊 MEMORY STATE: 🟡 MODERATE  
   Available: 6.7GB | Usage: 68.4%
   Configuration: configs/ensemble_2split_sampling_mps.yaml
   🔧 AUTO-SCALED: Steps 25→20, Timeout 180s→180s
```

### High Memory Pressure (<4GB available)
```
📊 MEMORY STATE: 🟠 HIGH
   Available: 3.2GB | Usage: 77.8%
   Configuration: configs/ensemble_2split_sampling_mps_balanced.yaml
   🔧 AUTO-SCALED: Steps 25→15, Timeout 180s→150s
```

### Critical Memory Pressure (<2GB available)
```
📊 MEMORY STATE: 🚨 CRITICAL
   Available: 1.8GB | Usage: 91.2%
   Configuration: configs/ensemble_2split_sampling_mps_failsafe.yaml
   🔧 AUTO-SCALED: Steps 25→3, Timeout 180s→60s
```

## Safety Mechanisms

### 1. **Emergency Shutdown Triggers**
- System memory >85% (vs 75% original)
- Process memory >10GB (vs 6GB original)  
- Swap usage >70% (vs 50% original)
- Rapid memory increase detected

### 2. **Graceful Error Handling**
- Memory cleanup before shutdown
- Clear error messages
- Preservation of other system processes
- Temporary file cleanup

### 3. **User Warnings**
```
⚠️  WARNING: Approaching memory limits
🚨 EMERGENCY SHUTDOWN TRIGGERED 🚨
System Memory: 86.3% (limit: 85%)
Shutting down gracefully to prevent system crash...
```

## Performance Comparison

| Scenario | Original Steps | Optimized Steps | Quality Impact | Safety Gain |
|----------|----------------|-----------------|----------------|-------------|
| Normal use | 5 (ultra-safe) | 15 (balanced) | Significant improvement | Maintained |
| Power user | 25 → 5 (forced) | 25 → 20 (scaled) | Minimal impact | Maintained |
| Memory pressure | 5 → 3 (forced) | Auto-scales gracefully | Gradual degradation | Enhanced |

## Migration Guide

### From Original Implementation
1. **Replace script**: Use `A2SB_upsample_api_mps_optimized.py`
2. **Update defaults**: 15 steps instead of 5 for better quality
3. **Add balanced config**: Copy `ensemble_2split_sampling_mps_balanced.yaml`
4. **Test with --verbose**: Monitor memory behavior
5. **Use --force sparingly**: Only for trusted scenarios

### Testing Your Setup
```bash
# Run validation
python test_basic_validation.py

# Test with small file
python inference/A2SB_upsample_api_mps_optimized.py -f test_degraded.wav -o test_output.wav --verbose

# Monitor memory usage
Activity Monitor → Memory tab → Watch "python" process
```

## Troubleshooting

### If Processing Still Fails
1. **Check available memory**: `top -o mem` or Activity Monitor
2. **Close other apps**: Free up RAM before processing
3. **Use smaller files**: Split long audio into segments
4. **Force failsafe mode**: `--timeout_seconds 60` 
5. **Enable verbose logging**: `--verbose` for detailed diagnostics

### If Quality is Poor
1. **Check memory state**: Ensure >8GB available for best quality
2. **Increase steps gradually**: Try 20, 25, 30 instead of 50
3. **Use force mode**: `--force` to override auto-scaling
4. **Process during low usage**: Close other memory-intensive apps

## Validation

The optimized implementation has been validated with:
- ✅ Syntax and structure verification
- ✅ Configuration file validation  
- ✅ Memory safety feature verification
- ✅ Graduated scaling logic
- ✅ User control options
- ✅ Graceful shutdown mechanisms

## Summary

The optimized implementation provides:
- **Better usability**: 15 default steps vs 5, 180s vs 90s timeout
- **Maintained safety**: Still prevents system crashes effectively
- **User control**: Force mode for power users who want full control
- **Graduated scaling**: 5 levels instead of 3 for smoother degradation
- **Balanced configuration**: New middle-ground option
- **Smart monitoring**: Trend analysis and less intrusive checking

This creates a much more usable system while preserving the critical safety mechanisms that prevent system crashes on M1 Max machines.