# Project Modifications and Setup Guide

This document outlines the essential modifications and setup instructions for this enhanced version of the audio restoration repository.

## Environment Setup

This project requires a specific Conda environment to ensure all dependencies are met.

**Note**: The `a2sb` environment has already been created locally using miniforge3 (conda-forge distribution). You can activate it directly.

**Important**: All globally installed Python packages have been cleaned up to ensure a clean, isolated environment. The project now exclusively uses the conda environment for all dependencies.

### 1. Create Conda Environment

Create the environment from the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

### 2. Activate Conda Environment

Activate the environment before running any scripts:

```bash
conda activate a2sb
```

## Key Modifications

### 1. Apple Silicon (M-series) Support

- **MPS Backend**: The repository has been updated to leverage Apple's Metal Performance Shaders (MPS) for significant performance improvements on M-series chips.
- **New Scripts**:
  - `inference/A2SB_upsample_api_mps.py`: A new inference script specifically for MPS.
  - `benchmark_mps.py`: A script to benchmark performance on MPS.
- **Configuration**:
  - `configs/ensemble_2split_sampling_mps.yaml`: A new configuration file for MPS-based ensembled inference.

### 2. Standalone Inference Scripts

- **API-style Scripts**: New scripts in the `inference/` directory provide a streamlined way to run inference on single audio files.
- **Command-line Arguments**: These scripts accept command-line arguments for input/output files, number of steps, and other parameters.

### 3. Memory Profiling and Debugging

- **Memory Logging**: The script `inference/A2SB_upsample_api_mps.py` has been instrumented with memory profiling using `psutil` to diagnose and prevent memory-related issues.
- **Error Handling**: Improved error handling and timeouts have been added to the inference process.

### 4. FFmpeg Integration

- **Audio Conversion**: The scripts now use `ffmpeg` for robust audio file handling, including converting MP3s to WAV format to prevent potential codec issues.

## Usage Example

To run inference using the MPS-accelerated script:

```bash
conda activate a2sb
python inference/A2SB_upsample_api_mps.py -f /path/to/your/audio.wav -o /path/to/output.wav -n 15
```
