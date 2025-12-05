# VRSwap Installation Guide - Windows 11 + Conda

## Prerequisites
- Windows 11
- NVIDIA GPU (RTX 4060 Ti 16GB optimal)
- Conda installed (Miniconda or Anaconda)
- FFmpeg installed

## Quick Setup (5 minutes)

### 1. Install FFmpeg
Download from: https://ffmpeg.org/download.html
Or use Windows Package Manager:
```cmd
winget install ffmpeg
```

### 2. Create Conda Environment
```cmd
conda env create -f environment.yml
conda activate vrswap
```

This single command installs:
- Python 3.12
- PyTorch 2.1.0 with CUDA 11.8
- All dependencies (InsightFace, TensorFlow, ONNX Runtime, OpenCV, etc.)

### 3. Download Model
Download `inswapper_128.onnx` and place in root directory:
```cmd
cd C:\path\to\vrswap
REM Download the model file
```

## Usage

### Process Frames with GPU (Fastest)
```cmd
conda activate vrswap
python swap.py --frames_folder "C:\path\to\frames" --face "C:\path\to\source_face.jpg" --gpu
```

### Optional Parameters
```cmd
# Batch processing (default: 4)
python swap.py --frames_folder "C:\path\to\frames" --face "source.jpg" --batch_size 8

# 8K tile processing (default: 512px tiles, use 0 to disable)
python swap.py --frames_folder "..." --face "..." --tile_size 512

# Number of GPU threads (default: 5)
python swap.py --frames_folder "..." --face "..." --gpu_threads 8

# Force CPU mode
python swap.py --frames_folder "..." --face "..." --cpu
```

### Examples

**4K Video (RTX 4060 Ti):**
```cmd
python swap.py --frames_folder "D:\video\frames" --face "D:\faces\source.jpg" --gpu --gpu_threads 5
```

**8K Video (with tile processing):**
```cmd
python swap.py --frames_folder "D:\video\8k_frames" --face "source.jpg" --gpu --tile_size 512
```

**CPU-only (no GPU):**
```cmd
python swap.py --frames_folder "..." --face "..." --cpu
```

## Performance on RTX 4060 Ti

| Resolution | Mode | Speed | Memory |
|-----------|------|-------|--------|
| 4K (3840x2160) | GPU + FP16 | ~2-3 fps | 8-10 GB |
| 8K (7680x4320) | GPU + Tiles | ~0.5-1 fps | 12-14 GB |
| 1080p | GPU | ~10-15 fps | 4-6 GB |

## Troubleshooting

### CUDA not available
```cmd
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch
conda remove torch torchvision torchaudio pytorch-cuda
conda install -c pytorch pytorch::pytorch pytorch::torchvision pytorch::torchaudio pytorch::pytorch-cuda=11.8
```

### Memory errors (OOM)
1. Reduce `--gpu_threads` to 2-3
2. Reduce tile size: `--tile_size 256`
3. Use CPU mode: `--cpu`
4. Process smaller frames

### Import errors
```cmd
# Reinstall all dependencies
conda env remove --name vrswap
conda env create -f environment.yml
conda activate vrswap
```

### ONNX Runtime errors
```cmd
# Windows-specific ONNX fix
pip install --force-reinstall onnxruntime-gpu==1.16.0
```

## File Locations

Windows paths automatically handle backslashes:
```
C:\Users\YourName\Videos\project
├── frames\           # Input JPG frames
├── processing\       # Temporary processing files
├── source_face.jpg   # Source face image
└── swap.py           # Script
```

## Environment Variables (Optional)

Set for better performance:
```cmd
set CUDA_LAUNCH_BLOCKING=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set OMP_NUM_THREADS=8
```

## Updating Dependencies

If you need newer versions:
```cmd
# Update all packages
conda update -n vrswap --all

# Or specific package
conda install -n vrswap pytorch::pytorch=2.1.0 pytorch::pytorch-cuda=11.8
```

## Conda Commands Reference

```cmd
# Activate environment
conda activate vrswap

# Deactivate
conda deactivate

# List environments
conda env list

# Remove environment
conda env remove --name vrswap

# Update pip packages within Conda
conda activate vrswap
pip install --upgrade pip
pip install -r requirements.txt
```

## Next Steps

1. Extract frames from video:
```cmd
ffmpeg -i input.mp4 -f image2 frames\%06d.jpg
```

2. Run swapping:
```cmd
python swap.py --frames_folder frames --face source_face.jpg --gpu
```

3. Convert frames back to video:
```cmd
ffmpeg -framerate 30 -i frames\%06d.jpg -c:v libx264 output.mp4
```

## Contact & Support

For issues, check:
- CUDA version compatibility
- GPU driver version (update via NVIDIA Control Panel)
- Available VRAM (16 GB minimum for 4K)
