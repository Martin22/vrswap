# process_video.py - Complete Video Processing Guide

## Overview

`process_video.py` je all-in-one skript, který automaticky:
1. Extrahuje framy z videa (FFmpeg)
2. Detekuje obličeje v každém framu
3. Swapuje obličeje se všemi faces ze složky
4. Enkóduje výsledné framy zpět na video
5. Vyčistí temp soubory

**Žádné ručné kroky nejsou potřeba!**

## Installation

```bash
conda activate vrswap
```

## Usage

### Nejjednoduší Způsob

```bash
python process_video.py --video input.mp4 --faces ./faces --output output.mp4
```

To je vše! Skript:
- Vezme `input.mp4`
- Vezme VŠECHNY obličeje z `./faces` (jpg, png, jpeg)
- Vypíše výsledek do `output.mp4`

### S Parametry

```bash
# GPU optimization
python process_video.py \
    --video input.mp4 \
    --faces ./faces \
    --output output.mp4 \
    --gpu \
    --gpu_threads 5

# 8K s tile processingem
python process_video.py \
    --video 8k_video.mp4 \
    --faces faces/ \
    --output 8k_output.mp4 \
    --gpu \
    --tile_size 512

# CPU only
python process_video.py \
    --video input.mp4 \
    --faces faces/ \
    --output output.mp4 \
    --cpu
```

## Parameters

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| --video | str | ✅ Yes | - | Path to input video file |
| --faces | str | ✅ Yes | - | Folder containing source face images |
| --output | str | ✅ Yes | - | Output video file path |
| --gpu | flag | ❌ No | True | Use GPU acceleration |
| --cpu | flag | ❌ No | - | Force CPU mode (overrides --gpu) |
| --gpu_threads | int | ❌ No | 5 | Number of parallel GPU threads |
| --tile_size | int | ❌ No | 512 | Tile size for 8K (0=disable) |

## Examples

### Example 1: Simple Usage

```bash
python process_video.py \
    --video "C:\Videos\myvideo.mp4" \
    --faces "C:\Faces" \
    --output "C:\Output\result.mp4"
```

**What happens:**
1. Reads all JPG/PNG from C:\Faces folder
2. Detects faces in each image
3. Extracts all frames from myvideo.mp4
4. Swaps faces with all detected faces
5. Encodes back to result.mp4

### Example 2: Multiple Face Images

Folder structure:
```
C:\Faces\
├── person1.jpg
├── person2.png
├── person3.jpg
└── person4.jpeg
```

Command:
```bash
python process_video.py \
    --video video.mp4 \
    --faces "C:\Faces" \
    --output result.mp4
```

**Result:** Each target face in video will be swapped with ALL 4 source faces!

### Example 3: 4K Video (Fast)

```bash
python process_video.py \
    --video 4k_video.mp4 \
    --faces faces/ \
    --output 4k_result.mp4 \
    --gpu \
    --gpu_threads 8
```

Expected: ~2-3 fps on RTX 4060 Ti

### Example 4: 8K Video (with Tiles)

```bash
python process_video.py \
    --video 8k_video.mp4 \
    --faces faces/ \
    --output 8k_result.mp4 \
    --gpu \
    --tile_size 512
```

Expected: ~0.5-1 fps on RTX 4060 Ti

### Example 5: CPU Processing

```bash
python process_video.py \
    --video video.mp4 \
    --faces faces/ \
    --output result.mp4 \
    --cpu
```

Slower but works on any machine.

## What the Script Does - Step by Step

### Step 1: Load Source Faces
- Scans faces folder for JPG/PNG/JPEG
- Detects face in each image
- Prepares for swapping
- Shows how many faces were loaded

### Step 2: Analyze Video
- Gets FPS, resolution, frame count
- Shows video information
- Prepares for frame extraction

### Step 3: Extract Frames
- Uses FFmpeg to extract frames
- Stores in temporary folder
- Shows extraction progress

### Step 4: Process Frames
- For each frame:
  - Detects all faces
  - Swaps with each source face
  - Uses FP16 if GPU available
  - Cleans GPU memory
- Shows processing progress
- Returns count of processed frames

### Step 5: Encode Video
- Uses FFmpeg with optimal settings
- Preset: medium (good balance)
- CRF: 18 (high quality)
- Maintains original FPS

### Step 6: Cleanup
- Deletes temporary directory
- Keeps only output video

## Output

When script completes:

```
======================================================================
✅ Processing Complete!
Output: C:\Output\result.mp4
Processed: 300/300 frames
======================================================================
```

## Performance

| Resolution | GPU Mode | Time | Memory |
|-----------|----------|------|--------|
| 1080p | GPU+FP16 | ~5 sec/frame | 5-6 GB |
| 4K | GPU+FP16 | ~20 sec/frame | 10 GB |
| 8K | GPU+Tiles | ~60 sec/frame | 14 GB |

**Estimates for RTX 4060 Ti 16GB**

## Troubleshooting

### "Video not found"
```bash
# Use full path
python process_video.py --video "C:\full\path\video.mp4" ...
```

### "No face images found"
```bash
# Check folder contains JPG/PNG/JPEG
# At least one face should be detected
dir C:\Faces\

# If faces are there, try different format
# jpg, png, jpeg all supported
```

### "No face detected in source"
```bash
# Make sure faces are clear and frontal
# Blurry or extreme angles may not work
# Try different face image
```

### Out of Memory (OOM)
```bash
# Reduce GPU threads
python process_video.py ... --gpu_threads 2

# Or use CPU mode
python process_video.py ... --cpu

# Or use smaller tile size
python process_video.py ... --tile_size 256
```

### CUDA not available
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, use CPU
python process_video.py ... --cpu
```

### FFmpeg not found
```bash
# Install FFmpeg
winget install ffmpeg

# Or add to PATH if already installed
```

## Advanced Usage

### Batch Processing Multiple Videos

```bash
@echo off
setlocal enabledelayedexpansion

set FACES_DIR=C:\Faces
set OUTPUT_DIR=C:\Output

for %%F in (C:\Videos\*.mp4) do (
    echo Processing %%~nF...
    python process_video.py ^
        --video "%%F" ^
        --faces "%FACES_DIR%" ^
        --output "%OUTPUT_DIR%\%%~nF_result.mp4" ^
        --gpu
)
```

### Script with GPU Monitoring

```bash
REM In separate terminal:
nvidia-smi -l 1

REM In main terminal:
python process_video.py --video video.mp4 --faces faces/ --output result.mp4 --gpu
```

### Different Quality Settings

To modify quality, edit subprocess FFmpeg command in process_video.py:

```python
# Current (high quality)
"-crf", "18",  # Lower = better quality, takes longer
"-preset", "medium",  # fast/medium/slow

# For speed (lower quality)
"-crf", "25",
"-preset", "fast",

# For maximum quality (slower)
"-crf", "12",
"-preset", "slow",
```

## Files and Directories

### Input
- **Video:** Any format FFmpeg supports (MP4, MOV, AVI, etc.)
- **Faces:** JPG, PNG, JPEG images in folder

### Output
- **Video:** MP4 file with swapped faces

### Temporary (automatically deleted)
- Extracted frames stored in temp folder during processing
- Deleted after encoding completes

## Performance Tips

1. **Start with lower resolution** to test
2. **Use GPU** - much faster than CPU
3. **Adjust --gpu_threads** based on your VRAM
   - 16GB: 5-8 threads
   - 8GB: 2-4 threads
   - 4GB: 1-2 threads
4. **For 8K, use tiles** - --tile_size 512
5. **Monitor GPU** - use nvidia-smi in another terminal

## Integration with Other Tools

### Extract Audio from Video (keep original audio)

```bash
# Extract audio
ffmpeg -i input.mp4 -q:a 0 -map a audio.aac

# Process video
python process_video.py --video input.mp4 --faces faces/ --output temp.mp4 --gpu

# Add audio back
ffmpeg -i temp.mp4 -i audio.aac -c:v copy -c:a aac -shortest final.mp4
```

### Different Output Format

Edit process_video.py or use FFmpeg after:

```bash
# Convert to MOV
ffmpeg -i result.mp4 -c:v libx264 -crf 18 result.mov

# Convert to WebM (smaller file)
ffmpeg -i result.mp4 -c:v libvpx-vp9 -crf 25 result.webm
```

## FAQ

**Q: Can I use only one face?**
A: Yes! Put only one JPG in faces folder.

**Q: Can I use multiple faces?**
A: Yes! Each face will be applied to targets in video.

**Q: Does it preserve audio?**
A: No - output has no audio. See "Integration" section to add it back.

**Q: Can I interrupt processing?**
A: Yes - Ctrl+C stops. Temp files cleaned up automatically.

**Q: How long does it take?**
A: Depends on resolution and GPU. See Performance table above.

**Q: Can I run multiple videos in parallel?**
A: Not recommended - use one at a time for best performance.

**Q: What's the output video codec?**
A: H.264 (libx264) for compatibility.

## Contact & Support

For issues:
1. Check Troubleshooting section above
2. Read INSTALL_WINDOWS.md
3. Verify GPU is working: `nvidia-smi`
4. Try CPU mode to isolate GPU issues
