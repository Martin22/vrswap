#!/usr/bin/env python3
"""
VRSwap Complete Video Processing Pipeline
Extracts frames → Face swap with multiple source faces → Encode to video
Windows 11 + Python 3.12 compatible + NVIDIA GPU acceleration
"""

import os
import sys
import cv2
import glob
import shutil
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import tempfile
import numpy as np

import core.globals
from core.swapper import get_face_swapper
from core.analyser import get_face, get_faces
from core.advanced_blending import AdvancedFaceBlender
from core.face_warping import warp_face, unwarp_face, create_soft_mask


class VideoProcessor:
    def __init__(self, video_path, faces_folder, output_path, gpu=True, threads=8, tile_size=512, fast_mode=False):
        """Initialize video processor - RTX 4060 Ti OPTIMIZED
        
        Args:
            threads: RTX 4060 Ti optimal = 8 (ne víc, GPU bottleneck)
            tile_size: 512 je sweet spot pro 4K/8K
        """
        self.video_path = os.path.normpath(video_path)
        self.faces_folder = os.path.normpath(faces_folder)
        self.output_path = os.path.normpath(output_path)
        self.gpu = gpu
        self.threads = min(threads, 8)  # RTX 4060 Ti: Max 8 threads optimal
        self.tile_size = tile_size
        self.fast_mode = fast_mode
        
        # RTX 4060 Ti: Batch processing settings
        self.batch_size = 4 if gpu else 1
        self.memory_cleanup_interval = 10  # Cleanup každých 10 frames
        
        # Temporary working directory
        self.work_dir = None
        self.frames_dir = None
        self.source_faces = []
        self.swapper = None
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate all input parameters"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        if not os.path.isdir(self.faces_folder):
            raise FileNotFoundError(f"Faces folder not found: {self.faces_folder}")
        
        # Check for face images
        face_files = glob.glob(os.path.join(self.faces_folder, "*.jpg"))
        face_files += glob.glob(os.path.join(self.faces_folder, "*.png"))
        face_files += glob.glob(os.path.join(self.faces_folder, "*.jpeg"))
        
        if not face_files:
            raise ValueError(f"No face images found in: {self.faces_folder}")
        
        print(f"[INFO] Found {len(face_files)} source face(s)")
        
        # Create output directory if needed
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    def _extract_faces(self):
        """Load and extract faces from folder"""
        print("[STEP 1/4] Loading source faces...")
        
        face_files = glob.glob(os.path.join(self.faces_folder, "*.jpg"))
        face_files += glob.glob(os.path.join(self.faces_folder, "*.png"))
        face_files += glob.glob(os.path.join(self.faces_folder, "*.jpeg"))
        face_files = sorted(face_files)
        
        for face_file in face_files:
            try:
                face_img = cv2.imread(face_file)
                if face_img is None:
                    print(f"[WARNING] Could not read: {face_file}")
                    continue
                
                # Detect face in image
                detected_face = get_face(face_img)
                if detected_face is None:
                    print(f"[WARNING] No face detected in: {os.path.basename(face_file)}")
                    continue
                
                # Ulož embedding pro pozdější výběr nejbližší tváře (rychlejší než zkoušet všechny)
                embedding = None
                if hasattr(detected_face, "normed_embedding") and detected_face.normed_embedding is not None:
                    embedding = np.asarray(detected_face.normed_embedding, dtype=np.float32)
                
                self.source_faces.append({
                    'name': os.path.basename(face_file),
                    'data': detected_face,
                    'image': face_img,
                    'embedding': embedding
                })
                print(f"  ✓ Loaded: {os.path.basename(face_file)}")
            except Exception as e:
                print(f"[ERROR] Failed to load {face_file}: {e}")
                continue
        
        if not self.source_faces:
            raise ValueError("No valid faces detected in source folder")
        
        print(f"[INFO] Successfully loaded {len(self.source_faces)} face(s)")
    
    def _get_video_info(self):
        """Get video information (fps, resolution, frame count)"""
        print("[STEP 2/4] Analyzing video...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {frame_count}")
        
        return fps, frame_count, (width, height)
    
    def _extract_frames(self):
        """Extract frames from video using NVIDIA GPU acceleration"""
        print("[STEP 2/4] Extracting frames from video (GPU accelerated)...")
        
        self.work_dir = tempfile.mkdtemp(prefix="vrswap_")
        self.frames_dir = os.path.join(self.work_dir, "frames")
        os.makedirs(self.frames_dir)
        
        output_pattern = os.path.join(self.frames_dir, "%06d.jpg")
        
        # FFmpeg command with NVIDIA GPU acceleration for decoding
        if self.gpu:
            cmd = [
                "ffmpeg",
                "-hwaccel", "cuda",                    # Enable CUDA hardware acceleration
                "-hwaccel_device", "0",                # Use first GPU
                "-i", self.video_path,
                "-qscale:v", "2",                      # High quality JPG
                "-v", "error",
                "-stats",
                output_pattern
            ]
        else:
            # CPU fallback
            cmd = [
                "ffmpeg",
                "-i", self.video_path,
                "-qscale:v", "2",
                "-v", "error",
                "-stats",
                output_pattern
            ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=False)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e}")
        
        frame_files = sorted(glob.glob(os.path.join(self.frames_dir, "*.jpg")))
        print(f"  ✓ Extracted {len(frame_files)} frames")
        
        return frame_files
    
    def _process_frames(self, frame_files):
        """Process frames with face swapping - RTX 4060 Ti OPTIMIZED"""
        from core.border_blur import apply_border_blur_gpu, apply_border_blur
        
        print("[STEP 3/4] Processing frames with face swap (RTX 4060 Ti mode)...")
        
        if self.swapper is None:
            self.swapper = get_face_swapper()
        
        processed_count = 0
        frame_counter = 0
        
        # RTX 4060 Ti: Use CUDA stream
        use_gpu_blur = self.gpu and core.globals.device == 'cuda'

        def select_best_source(target_face):
            """Najdi nejlepší zdrojovou tvář podle embedding similarity (urychlí proces)."""
            if not self.source_faces:
                return None
            if not hasattr(target_face, "normed_embedding") or target_face.normed_embedding is None:
                return self.source_faces[0]
            target_emb = np.asarray(target_face.normed_embedding, dtype=np.float32)
            best = self.source_faces[0]
            best_score = -1.0
            for source in self.source_faces:
                if source.get('embedding') is None:
                    continue
                score = float(np.dot(target_emb, source['embedding']))
                if score > best_score:
                    best_score = score
                    best = source
            return best
        
        with tqdm(total=len(frame_files), desc="Swapping faces", unit="frame") as pbar:
            for frame_file in frame_files:
                try:
                    frame = cv2.imread(frame_file)
                    if frame is None:
                        print(f"[WARNING] Could not read frame: {frame_file}")
                        pbar.update(1)
                        continue
                    
                    # Get all target faces in this frame
                    target_faces = get_faces(frame)
                    
                    if target_faces:
                        for target_face in target_faces:
                            try:
                                best_source = select_best_source(target_face)
                                if best_source is None:
                                    continue
                                source_face = best_source['data']
                                
                                # RTX 4060 Ti: Direct swap s FP16
                                if core.globals.use_fp16 and core.globals.device == 'cuda':
                                    import torch
                                    with torch.autocast('cuda', dtype=torch.float16):
                                        frame = self.swapper.get(frame, target_face, source_face, paste_back=True)
                                else:
                                    frame = self.swapper.get(frame, target_face, source_face, paste_back=True)
                                
                                # RTX 4060 Ti: GPU-accelerated border blur
                                bbox = target_face.bbox
                                if use_gpu_blur:
                                    frame = apply_border_blur_gpu(frame, bbox, blur_strength=15, device='cuda')
                                else:
                                    frame = apply_border_blur(frame, bbox, blur_strength=15)
                                
                            except Exception as e:
                                print(f"[DEBUG] Swap error: {e}")
                                continue
                        
                        # Save processed frame
                        cv2.imwrite(frame_file, frame)
                        processed_count += 1
                        
                        # RTX 4060 Ti: Aggressive memory cleanup
                        frame_counter += 1
                        if frame_counter % self.memory_cleanup_interval == 0 and core.globals.device == 'cuda':
                            import torch
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"[ERROR] Frame processing error: {e}")
                    pbar.update(1)
                    continue
        
        # Final cleanup
        if core.globals.device == 'cuda':
            import torch
            torch.cuda.empty_cache()
        
        print(f"  ✓ Processed {processed_count}/{len(frame_files)} frames")
        return processed_count
    
    def _encode_video(self, fps, output_resolution):
        """Encode frames back to video using NVIDIA NVENC - RTX 4060 Ti OPTIMIZED"""
        print("[STEP 4/4] Encoding video (GPU accelerated - RTX 4060 Ti NVENC)...")
        
        input_pattern = os.path.join(self.frames_dir, "%06d.jpg")
        
        # RTX 4060 Ti: NVENC hardware encoding s p5 preset (best quality)
        if self.gpu:
            cmd = [
                "ffmpeg",
                "-y",                                   # Overwrite output
                "-framerate", str(fps),
                "-i", input_pattern,
                "-c:v", "h264_nvenc",                   # NVIDIA hardware encoder
                "-preset", "p5",                        # p5 = High Quality (p2=fast, p4=medium, p5=slow/HQ)
                "-tune", "hq",                          # High quality tuning
                "-rc", "vbr",                           # Variable bitrate
                "-cq", "18",                            # Quality level (lower=better, 0-51, 18=visually lossless)
                "-b:v", "0",                            # Let rate control decide bitrate
                "-maxrate", "20M",                      # Max bitrate cap
                "-bufsize", "40M",                      # Buffer size
                "-pix_fmt", "yuv420p",
                "-gpu", "0",                            # Use first GPU
                "-movflags", "+faststart",              # Web optimization
                self.output_path
            ]
        else:
            # CPU fallback (libx264)
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate", str(fps),
                "-i", input_pattern,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                self.output_path
            ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  ✓ Video encoded: {self.output_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg encoding failed: {e.stderr.decode() if e.stderr else e}")
    
    def process(self):
        """Run complete processing pipeline"""
        try:
            print(f"\n{'='*70}")
            print("VRSwap Video Processing Pipeline")
            if self.gpu:
                print("Mode: NVIDIA GPU Accelerated")
            else:
                print("Mode: CPU")
            print(f"{'='*70}\n")
            
            # Step 1: Load source faces
            self._extract_faces()
            
            # Step 2: Get video info and extract frames
            fps, frame_count, resolution = self._get_video_info()
            frame_files = self._extract_frames()
            
            # Step 3: Process frames
            processed = self._process_frames(frame_files)
            
            # Step 4: Encode video
            self._encode_video(fps, resolution)
            
            print(f"\n{'='*70}")
            print("✅ Processing Complete!")
            print(f"Output: {self.output_path}")
            print(f"Processed: {processed}/{frame_count} frames")
            print(f"{'='*70}\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Cleanup temporary directory
            if self.work_dir and os.path.exists(self.work_dir):
                print("[INFO] Cleaning up temporary files...")
                shutil.rmtree(self.work_dir)


def main():
    parser = argparse.ArgumentParser(
        description="VRSwap - Complete video processing with face swapping (NVIDIA GPU accelerated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with GPU
  python process_video.py --video input.mp4 --faces ./faces --output output.mp4

  # With GPU optimization
  python process_video.py --video video.mp4 --faces faces/ --output result.mp4 --gpu --gpu_threads 5

  # 8K processing with tiles
  python process_video.py --video 8k_video.mp4 --faces faces/ --output output.mp4 --gpu --tile_size 512

  # CPU only
  python process_video.py --video video.mp4 --faces faces/ --output output.mp4 --cpu
        """
    )
    
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--faces", required=True, help="Folder with source face images")
    parser.add_argument("--output", required=True, help="Output video file path")
    parser.add_argument("--gpu", dest="gpu", action="store_true", default=True,
                       help="Use GPU acceleration (default: True)")
    parser.add_argument("--cpu", dest="gpu", action="store_false",
                       help="Force CPU mode")
    parser.add_argument("--gpu_threads", type=int, default=5,
                       help="Number of GPU threads (default: 5)")
    parser.add_argument("--tile_size", type=int, default=512,
                       help="Tile size for 8K processing (0=disable, default: 512)")
    parser.add_argument("--fast", action="store_true", default=False,
                       help="Fast mode - skip color matching")
    
    parser.add_argument("--execution-provider", choices=['cuda', 'tensorrt', 'cpu'], default='cuda',
                       help="Execution provider (default: cuda)")
    
    args = parser.parse_args()
    
    # Initialize GPU settings
    if not args.gpu or args.execution_provider == 'cpu':
        args.gpu = False
        core.globals.providers = ['CPUExecutionProvider']
        print("[INFO] Using CPU mode")
    elif args.execution_provider == 'tensorrt':
        core.globals.providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        print("[INFO] Using GPU mode (TensorRT)")
    else:
        core.globals.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("[INFO] Using GPU mode (CUDA)")
    
    # Create processor and run
    try:
        processor = VideoProcessor(
            video_path=args.video,
            faces_folder=args.faces,
            output_path=args.output,
            gpu=args.gpu,
            threads=args.gpu_threads,
            tile_size=args.tile_size,
            fast_mode=args.fast
        )
        
        success = processor.process()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
