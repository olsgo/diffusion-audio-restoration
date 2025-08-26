# Memory Crash Prevention for A2SB on Apple Silicon M1 Max

## Problem Summary

The diffusion audio model was causing catastrophic system crashes on Mac Studio with M1 Max chip, characterized by:

- **Watchdog timeout panics** - Complete system freeze requiring hard reboot
- **Extreme memory pressure** - 100% of segments limit with 83 swapfiles
- **Memory thrashing** - Aggressive swapping between RAM and SSD
- **High kernel_task CPU usage** - Kernel overwhelmed managing memory

## Root Cause Analysis

The crashes were caused by the diffusion model's memory consumption exceeding available resources, leading to:

1. Complete RAM exhaustion
2. Excessive swap file creation (83 files)
3. Memory thrashing between RAM and SSD
4. Kernel resource starvation
5. System-wide freeze and watchdog timeout

## Comprehensive Solution Implementation

### 1. Aggressive Memory Monitoring (`SystemMemoryMonitor`)

**Location**: `inference/A2SB_upsample_api_mps.py`

**Features**:

- Real-time background monitoring every 2 seconds
- Tracks system memory percentage, process RSS memory, and swap usage
- **Emergency shutdown triggers**:
  - System memory > 75%
  - Process memory > 6144MB (6GB)
  - Swap usage > 50%
- Automatic garbage collection and `os._exit(1)` on critical thresholds
- Warning alerts when approaching limits

**Implementation**:

```python
class SystemMemoryMonitor:
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        # Background thread monitoring with emergency shutdown
```

### 2. Emergency Process Termination

**Signal Handlers**:

- `SIGTERM` and `SIGINT` handlers for graceful shutdown
- Automatic memory monitor cleanup
- Forced garbage collection before exit

**Emergency Shutdown**:

- Immediate `gc.collect()` and `os._exit(1)` on critical memory thresholds
- Prevents system from entering memory thrashing state

### 3. Auto-Scaling Memory Management

**Dynamic Parameter Adjustment** (`auto_scale_parameters_for_memory`):

| Memory State | Configuration                                | Max Steps | Timeout | Trigger                       |
| ------------ | -------------------------------------------- | --------- | ------- | ----------------------------- |
| **CRITICAL** | `ensemble_2split_sampling_mps_failsafe.yaml` | 3         | 60s     | >80% usage OR <4GB available  |
| **WARNING**  | `ensemble_2split_sampling_mps.yaml`          | 5         | 90s     | >70% usage OR <8GB available  |
| **NORMAL**   | `ensemble_2split_sampling_mps.yaml`          | 10        | 120s    | <70% usage AND >8GB available |

### 4. Ultra-Conservative Default Parameters

**Command Line Defaults**:

- `predict_n_steps`: 5 (reduced from 25)
- `timeout_seconds`: 90 (reduced from 300)
- Both labeled as "ULTRA-SAFE" in help text

**Runtime Environment**:

- `OMP_NUM_THREADS=1` - Single-threaded operations
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` - Aggressive MPS memory management
- `PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection` - Force garbage collection
- `MALLOC_TRIM_THRESHOLD_=65536` - Aggressive malloc trimming
- `PYTORCH_JIT=0` - Disable JIT compilation
- CPU priority: `nice -n 19` (lowest priority)

### 5. Fail-Safe Configuration

**File**: `configs/ensemble_2split_sampling_mps_failsafe.yaml`

**Key Optimizations**:

- **Precision**: `16-mixed` (half precision)
- **Model Architecture**: Reduced channels, attention heads, and layers
- **Spectrogram**: Smaller FFT sizes (1024 vs 2048)
- **Data Loading**: `num_workers=0`, `batch_size=1`
- **Memory**: Disabled persistent workers, pin memory, checkpointing
- **Logging**: All progress bars and summaries disabled

### 6. Memory Cleanup Between Stages

**Functions**:

- `aggressive_memory_cleanup()`: Comprehensive cleanup routine
- `check_system_memory_pressure()`: Pre-stage memory validation

**Cleanup Actions**:

- Python garbage collection (`gc.collect()`)
- MPS cache clearing (`torch.mps.empty_cache()`)
- Malloc heap trimming (`libc.malloc_trim(0)`)
- Memory pressure validation before proceeding

**Strategic Placement**:

- Before model loading
- After model loading
- Before inference
- After inference completion
- Before final cleanup

### 7. Enhanced Resource Limits

**Trainer Configuration**:

- `batch_size=1` (minimal)
- `num_workers=0` (no multiprocessing)
- `persistent_workers=false`
- `pin_memory=false`
- `enable_checkpointing=false`
- `logger=false`
- `enable_progress_bar=false`

## Usage Guidelines

### Recommended Processing Steps

1. **For Small Files (<30 seconds)**:

   ```bash
   python A2SB_upsample_api_mps.py input.wav output.wav --predict_n_steps 5
   ```

2. **For Medium Files (30-60 seconds)**:

   ```bash
   python A2SB_upsample_api_mps.py input.wav output.wav --predict_n_steps 3
   ```

3. **For Large Files (>60 seconds)**:
   - Split into smaller segments first
   - Use fail-safe mode with 3 steps maximum

### Memory State Monitoring

The system automatically:

1. **Monitors** memory every 2 seconds
2. **Warns** when approaching limits
3. **Scales down** parameters based on available memory
4. **Terminates** immediately if critical thresholds exceeded

### Emergency Indicators

**Watch for these console messages**:

- 🚨 **CRITICAL MEMORY STATE**: Using fail-safe configuration
- ⚠️ **HIGH MEMORY USAGE**: Using conservative settings
- 🛡️ **EMERGENCY SHUTDOWN**: Memory limits exceeded

## Technical Implementation Details

### Memory Thresholds

| Threshold          | Action                      | Purpose                               |
| ------------------ | --------------------------- | ------------------------------------- |
| 70% system memory  | Switch to conservative mode | Prevent approaching danger zone       |
| 75% system memory  | Emergency shutdown          | Prevent memory thrashing              |
| 80% system memory  | Force fail-safe config      | Last resort before crash              |
| 6GB process memory | Emergency shutdown          | Prevent single process monopolization |
| 50% swap usage     | Emergency shutdown          | Prevent swap thrashing                |

### File Modifications

1. **`inference/A2SB_upsample_api_mps.py`**: Complete rewrite with memory management
2. **`configs/ensemble_2split_sampling_mps_failsafe.yaml`**: New fail-safe configuration
3. **`mps_optimization_guide.md`**: Updated with new safety features

## Validation and Testing

### Before Processing

- System memory check
- Available disk space validation
- Swap usage assessment

### During Processing

- Real-time memory monitoring
- Automatic parameter scaling
- Emergency shutdown capability

### After Processing

- Memory cleanup and validation
- Resource release confirmation

## Expected Outcomes

1. **No More System Crashes**: Emergency shutdown prevents memory thrashing
2. **Automatic Scaling**: System adapts to available resources
3. **Graceful Degradation**: Quality reduces before system fails
4. **Resource Protection**: Other applications remain functional
5. **Predictable Behavior**: Clear warnings and automatic adjustments

## Troubleshooting

### If Processing Fails

1. Check available memory before starting
2. Close other memory-intensive applications
3. Use smaller audio segments
4. Manually specify fewer diffusion steps

### If System Still Struggles

1. Restart to clear memory fragmentation
2. Check for memory leaks in other applications
3. Consider upgrading to higher memory configuration
4. Use external processing with smaller batch sizes

This comprehensive solution transforms the A2SB diffusion model from a system-crashing liability into a well-behaved, resource-aware application that gracefully adapts to available system resources while preventing catastrophic memory pressure scenarios.
