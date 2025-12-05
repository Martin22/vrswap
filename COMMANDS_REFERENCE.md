# VRSwap Commands - Windows 11 Quick Reference

## 🚀 SIMPLEST WAY - process_video.py (NEW!)

All in one command! No manual FFmpeg needed.

```cmd
conda activate vrswap

REM Basic usage
python process_video.py --video input.mp4 --faces ./faces --output output.mp4

REM With GPU optimization
python process_video.py --video video.mp4 --faces faces/ --output result.mp4 --gpu --gpu_threads 5

REM 8K with tiles
python process_video.py --video 8k.mp4 --faces faces/ --output 8k_out.mp4 --gpu --tile_size 512

REM CPU only
python process_video.py --video video.mp4 --faces faces/ --output output.mp4 --cpu
```

**That's it!** The script does:
1. ✅ Extracts frames from video
2. ✅ Detects faces in each frame
3. ✅ Swaps with all faces from folder
4. ✅ Encodes back to video
5. ✅ Cleans up temporary files

---

## Initial Setup (First Time)

```cmd
REM 1. Install Conda environment
conda env create -f environment.yml

REM 2. Activate environment
conda activate vrswap

REM 3. Download inswapper_128.onnx model
REM https://huggingface.co/...

REM 4. Verify GPU setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Processing Workflow

### Step 1: Extract Frames from Video
```cmd
cd C:\path\to\vrswap
ffmpeg -i "C:\input\video.mp4" -f image2 "frames\%06d.jpg"
```

### Step 2: Prepare Source Face
```cmd
REM Copy source face image to project folder
copy "C:\faces\source.jpg" "source_face.jpg"
```

### Step 3: Run Face Swap

**4K Video (Fast - RTX 4060 Ti):**
```cmd
conda activate vrswap
python swap.py --frames_folder frames --face source_face.jpg --gpu --gpu_threads 5
```

**8K Video (with tile processing):**
```cmd
python swap.py --frames_folder frames --face source_face.jpg --gpu --tile_size 512 --gpu_threads 4
```

**CPU-Only (No GPU):**
```cmd
python swap.py --frames_folder frames --face source_face.jpg --cpu
```

**Custom Configuration:**
```cmd
python swap.py ^
    --frames_folder frames ^
    --face source_face.jpg ^
    --gpu ^
    --gpu_threads 8 ^
    --tile_size 512 ^
    --batch_size 4
```

### Step 4: Optional Upscaling
```cmd
python upscale.py --frames_folder frames --threads 4 --upscale_factor 2
```

### Step 5: Convert Back to Video
```cmd
REM 30 fps (adjust as needed)
ffmpeg -framerate 30 -i "frames\%06d.jpg" -c:v libx264 -crf 18 "output.mp4"

REM With audio from original
ffmpeg -framerate 30 -i "frames\%06d.jpg" -i "input.mp4" -c:v libx264 -c:a aac -shortest "output.mp4"
```

## Common Use Cases

### Standard 1080p Processing (Fast)
```cmd
ffmpeg -i input.mp4 -f image2 frames\%06d.jpg
python swap.py --frames_folder frames --face source.jpg --gpu
ffmpeg -framerate 30 -i frames\%06d.jpg output.mp4
```

### 4K Video (RTX 4060 Ti)
```cmd
ffmpeg -i 4k_video.mp4 -f image2 4k_frames\%06d.jpg
python swap.py --frames_folder 4k_frames --face source.jpg --gpu --gpu_threads 5
ffmpeg -framerate 30 -i 4k_frames\%06d.jpg -c:v libx264 4k_output.mp4
```

### 8K Video (with tiles)
```cmd
ffmpeg -i 8k_video.mp4 -f image2 8k_frames\%06d.jpg
python swap.py --frames_folder 8k_frames --face source.jpg --gpu --tile_size 512
ffmpeg -framerate 24 -i 8k_frames\%06d.jpg -c:v libx265 -crf 18 8k_output.mp4
```

### Batch Processing Multiple Videos
```cmd
for %%F in (videos\*.mp4) do (
    ffmpeg -i "%%F" -f image2 "frames\%%~nF\%06d.jpg"
    python swap.py --frames_folder "frames\%%~nF" --face source.jpg --gpu
    ffmpeg -framerate 30 -i "frames\%%~nF\%06d.jpg" "output\%%~nF.mp4"
)
```

## Advanced Parameters

```cmd
python swap.py --help
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| --frames_folder | str | REQUIRED | Path to frames directory |
| --face | str | REQUIRED | Source face image path |
| --gpu | flag | True | Enable GPU acceleration |
| --cpu | flag | False | Force CPU mode |
| --gpu_threads | int | 5 | Number of parallel threads |
| --tile_size | int | 512 | Tile size for 8K (0=disable) |
| --batch_size | int | 4 | Batch processing size |

## GPU Memory Management

**Memory Usage per Resolution (RTX 4060 Ti):**
- 720p: 3-4 GB
- 1080p: 4-6 GB
- 4K: 8-10 GB
- 8K (with tiles): 12-14 GB

**If OOM Occurs:**
```cmd
REM Reduce threads
python swap.py --frames_folder frames --face source.jpg --gpu --gpu_threads 2

REM Smaller tiles
python swap.py --frames_folder frames --face source.jpg --gpu --tile_size 256

REM Force CPU
python swap.py --frames_folder frames --face source.jpg --cpu
```

## Conda Commands

```cmd
REM List environments
conda env list

REM Activate environment
conda activate vrswap

REM Deactivate
conda deactivate

REM Update all packages
conda update -n vrswap --all

REM Install specific package
conda install -n vrswap pytorch::pytorch=2.1.0

REM Remove environment
conda env remove --name vrswap

REM Create from file (initial setup)
conda env create -f environment.yml
```

## Troubleshooting Commands

```cmd
REM Check Python version
python --version

REM Check PyTorch/CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

REM Check GPU properties
python -c "import torch; print(torch.cuda.get_device_properties(0))"

REM Check FFmpeg
ffmpeg -version

REM Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

REM Test face swap functionality
python -c "from core.swapper import get_face_swapper; print('Face swapper OK')"

REM Check environment
conda info
```

## Performance Monitoring

```cmd
REM Monitor GPU usage (requires nvidia-smi)
nvidia-smi -l 1

REM Check system resources
tasklist /v
```

## Environment Variables (Optional)

```cmd
REM Set for better performance
set CUDA_LAUNCH_BLOCKING=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set OMP_NUM_THREADS=8

REM Then run
python swap.py --frames_folder frames --face source.jpg --gpu
```

## Path Examples

```cmd
REM Windows paths (automatically converted)
D:\Videos\project\frames
C:\Users\Name\Downloads\video.mp4
E:\Storage\source_face.jpg

REM Relative paths (same folder as swap.py)
frames
source_face.jpg
processing

REM Mixed (works fine)
python swap.py --frames_folder "C:\data\frames" --face "source.jpg"
```

## Complete Pipeline Example

```cmd
REM Create working directory
mkdir C:\vrswap_work
cd C:\vrswap_work

REM Copy source face
copy "C:\faces\person.jpg" source_face.jpg

REM Extract frames (30fps, skip to 720p for speed)
ffmpeg -i "input.mp4" -vf "fps=30,scale=1280:720" -f image2 frames\%06d.jpg

REM Activate Conda
conda activate vrswap

REM Run face swap (4 threads for faster iteration)
python "C:\path\to\vrswap\swap.py" --frames_folder frames --face source_face.jpg --gpu --gpu_threads 4

REM Convert back (keeping original audio)
ffmpeg -framerate 30 -i frames\%06d.jpg -i input.mp4 -c:v libx264 -c:a aac -shortest output.mp4

REM Upscale if needed
python "C:\path\to\vrswap\upscale.py" --frames_folder frames --threads 2

REM Done
echo Face swap complete! Output: output.mp4
```

## Quick Fixes

**CUDA not available:**
```cmd
REM Reinstall PyTorch with CUDA
conda remove torch torchvision torchaudio pytorch-cuda
conda install -c pytorch pytorch::pytorch pytorch::torchvision pytorch::torchaudio pytorch::pytorch-cuda=11.8
```

**Slow processing:**
```cmd
REM Increase threads (if memory allows)
python swap.py --frames_folder frames --face source.jpg --gpu --gpu_threads 8
```

**Memory errors:**
```cmd
REM Decrease resources
python swap.py --frames_folder frames --face source.jpg --gpu --gpu_threads 2 --tile_size 256
```

**Import errors:**
```cmd
REM Reinstall environment
conda env remove --name vrswap
conda env create -f environment.yml
conda activate vrswap
```

## Performance Tips

1. **Start with 720p** - Test processing before 4K
2. **Use GPU** - ~10x faster than CPU
3. **Adjust threads** - 5 is default, 8 if you have 16GB+
4. **Enable FP16** - Automatic, saves memory
5. **Tile for 8K** - Use 512px tiles
6. **Batch processing** - Default batch_size=4 is good

## Need Help?

```cmd
REM Check all parameters
python swap.py --help

REM Test installation
python -c "import core.swapper, core.analyser, core.globals; print('All imports OK')"

REM See README
type INSTALL_WINDOWS.md
type OPTIMIZATION_SUMMARY.md
```
