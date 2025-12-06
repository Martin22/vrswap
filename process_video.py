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
from core.color_transfer import adaptive_color_blend


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

        def _detect_codec(path):
            try:
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", path
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )
                return probe.stdout.strip()
            except Exception:
                return None

        decoder_map = {
            "h264": "h264_cuvid",
            "hevc": "hevc_cuvid",
            "h265": "hevc_cuvid",
            "av1": "av1_cuvid",
            "vp9": "vp9_cuvid",
            "mpeg2video": "mpeg2_cuvid",
        }
        codec_name = _detect_codec(self.video_path) if self.gpu else None
        chosen_decoder = decoder_map.get(codec_name) if codec_name else None
        
        # FFmpeg command with NVIDIA GPU acceleration for decoding
        if self.gpu:
            cmd = [
                "ffmpeg",
                "-hwaccel", "cuda",                    # Enable CUDA hardware acceleration
                "-hwaccel_device", "0",                # Use first GPU
                # Choose fastest NVDEC decoder per codec if known
                *( ["-c:v", chosen_decoder] if chosen_decoder else [] ),
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
            if self.swapper is None:
                print("[ERROR] get_face_swapper returned None")
        
        processed_count = 0
        frame_counter = 0
        
        # RTX 4060 Ti: Use CUDA stream
        use_gpu_blur = self.gpu and core.globals.device == 'cuda'
        # Cache for perspective remap grids to reduce overhead in VR-like pole handling
        persp_cache = {}

        def cached_perspective_grid(theta, phi, fov, size, w, h):
            key = (round(theta, 2), round(phi, 2), round(fov, 1), size, w, h)
            if key in persp_cache:
                return persp_cache[key]

            equ_cx = (w - 1) / 2.0
            equ_cy = (h - 1) / 2.0

            wFOV = fov
            hFOV = float(size) / size * wFOV
            w_len = np.tan(np.radians(wFOV / 2.0))
            h_len = np.tan(np.radians(hFOV / 2.0))

            x_map = np.ones([size, size], np.float32)
            y_map = np.tile(np.linspace(-w_len, w_len, size), [size, 1])
            z_map = -np.tile(np.linspace(-h_len, h_len, size), [size, 1]).T

            D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
            xyz = np.stack((x_map, y_map, z_map), axis=2) / np.repeat(D[:, :, np.newaxis], 3, axis=2)

            y_axis = np.array([0.0, 1.0, 0.0], np.float32)
            z_axis = np.array([0.0, 0.0, 1.0], np.float32)
            R1, _ = cv2.Rodrigues(z_axis * np.radians(theta))
            R2, _ = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-phi))

            xyz = xyz.reshape([size * size, 3]).T
            xyz = np.dot(R1, xyz)
            xyz = np.dot(R2, xyz).T
            lat = np.arcsin(xyz[:, 2])
            lon = np.arctan2(xyz[:, 1], xyz[:, 0])

            lon = lon.reshape([size, size]) / np.pi * 180
            lat = -lat.reshape([size, size]) / np.pi * 180

            lon_map = lon / 180 * equ_cx + equ_cx
            lat_map = lat / 90 * equ_cy + equ_cy

            persp_cache[key] = (lon_map.astype(np.float32), lat_map.astype(np.float32))
            return persp_cache[key]

        def soft_bbox_mask(h, w, border):
            mask = np.zeros((h, w), dtype=np.float32)
            inner_y1 = max(0, border)
            inner_y2 = max(inner_y1 + 1, h - border)
            inner_x1 = max(0, border)
            inner_x2 = max(inner_x1 + 1, w - border)
            mask[inner_y1:inner_y2, inner_x1:inner_x2] = 1.0
            ksize = max(3, border * 2 + 1)
            if ksize % 2 == 0:
                ksize += 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), border / 3.0 + 1e-6)
            if mask.max() > 1e-6:
                mask = mask / mask.max()
            return mask

        def safe_swap_call(frame_in, tgt_face, src_face, paste_back=True, retry_without_trt=False):
            """Call swapper with debug diagnostics; auto-fallback if TensorRT returns empty outputs."""
            if not hasattr(self.swapper, 'get') or not callable(getattr(self.swapper, 'get', None)):
                print(f"[ERROR] Swapper.get is not callable. swapper={type(self.swapper)}, attrs={dir(self.swapper) if self.swapper else 'None'}")
                raise RuntimeError("Swapper.get not callable")
            try:
                if core.globals.use_fp16 and core.globals.device == 'cuda':
                    import torch
                    with torch.autocast('cuda', dtype=torch.float16):
                        return self.swapper.get(frame_in, tgt_face, src_face, paste_back=paste_back)
                return self.swapper.get(frame_in, tgt_face, src_face, paste_back=paste_back)
            except Exception as exc:
                import traceback
                bbox_dbg = getattr(tgt_face, 'bbox', None)
                kps_dbg = getattr(tgt_face, 'kps', None)
                print(f"[DEBUG] Swapper call failed. swapper={type(self.swapper)}, device={core.globals.device}, providers={core.globals.providers}, paste_back={paste_back}, bbox={bbox_dbg}, kps_shape={(kps_dbg.shape if kps_dbg is not None else None)}, frame_shape={frame_in.shape if hasattr(frame_in, 'shape') else None}, tgt_face={tgt_face}")
                traceback.print_exc()

                # TensorRT sometimes returns an empty output list; retry once without TensorRT EP
                if (not retry_without_trt) and ('list index out of range' in str(exc)):
                    try:
                        import core.swapper as swapper_module
                        orig_providers = list(core.globals.providers or [])
                        orig_options = core.globals.provider_options
                        if 'TensorrtExecutionProvider' in orig_providers:
                            fallback_providers = [p for p in orig_providers if p != 'TensorrtExecutionProvider']
                            fallback_options = None
                            if isinstance(orig_options, list) and len(orig_options) == len(orig_providers):
                                # drop the first provider option corresponding to TensorRT
                                trt_index = orig_providers.index('TensorrtExecutionProvider')
                                fallback_options = [opt for i, opt in enumerate(orig_options) if i != trt_index]

                            core.globals.providers = fallback_providers
                            core.globals.provider_options = fallback_options
                            swapper_module.FACE_SWAPPER = None  # force reload
                            self.swapper = get_face_swapper()
                            print("[DEBUG] Retrying swap without TensorRT Execution Provider after IndexError")
                            return safe_swap_call(frame_in, tgt_face, src_face, paste_back=paste_back, retry_without_trt=True)
                    except Exception:
                        traceback.print_exc()

                raise

        def needs_pole_stabilization(bbox, frame_shape):
            h, w = frame_shape[:2]
            x1, y1, x2, y2 = bbox
            bh = y2 - y1
            bw = x2 - x1
            # Velmi velký obličej nebo příliš nahoře/dole → hůře mapovatelné na equirect
            large = (bh * bw) / (h * w) > 0.18
            near_pole = y1 < 0.12 * h or y2 > 0.88 * h
            return large or near_pole

        def swap_in_perspective(frame, target_face, source_face):
            """Reproject local area to perspective to reduce equirect distortion, swap there, map back.

            Experimental: re-detect face on the perspective patch to avoid bbox mismatch.
            Errors are swallowed and fall back to normal swap.
            """
            try:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = target_face.bbox
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                # FOV podle velikosti boxu; clamp na rozumné hodnoty
                fov = float(np.clip((x2 - x1) / w * 200.0, 80.0, 150.0))
                size = 640
                equ_cx = (w - 1) / 2.0
                equ_cy = (h - 1) / 2.0

                theta = (cx - equ_cx) / equ_cx * 180.0  # left/right
                phi = -(cy - equ_cy) / equ_cy * 90.0    # up/down

                lon_map, lat_map = cached_perspective_grid(theta, phi, fov, size, w, h)
                persp = cv2.remap(frame, lon_map, lat_map, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

                # Re-detect face in perspective space to avoid bbox mismatch
                persp_faces = get_faces(persp)
                if persp_faces is None:
                    print("[DEBUG] get_faces returned None in perspective path")
                    return None
                if not persp_faces:
                    return None
                persp_target = persp_faces[0]

                # Run swap on perspective patch
                swapped = safe_swap_call(persp, persp_target, source_face, paste_back=True)

                # Map back to equirect by scattering pixels
                lx = np.clip(np.rint(lon_map), 0, w - 1).astype(np.int32)
                ly = np.clip(np.rint(lat_map), 0, h - 1).astype(np.int32)
                valid = (lx >= 0) & (lx < w) & (ly >= 0) & (ly < h)
                frame_out = frame.copy()
                frame_out[ly[valid], lx[valid]] = swapped[valid]
                return frame_out
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[DEBUG] Perspective swap failed: {e}")
                return None

        def shifted_face(face, dx, dy):
            """Return a shallow copy of face with bbox/kps shifted by dx,dy."""
            import copy
            f = copy.deepcopy(face)
            if hasattr(f, 'bbox') and f.bbox is not None:
                f.bbox = [f.bbox[0] + dx, f.bbox[1] + dy, f.bbox[2] + dx, f.bbox[3] + dy]
            if hasattr(f, 'kps') and f.kps is not None:
                f.kps = f.kps + np.array([dx, dy])
            for attr in ['landmark_2d_106', 'landmark_3d_68', 'landmark_2d_5']:
                if hasattr(f, attr) and getattr(f, attr) is not None:
                    setattr(f, attr, getattr(f, attr) + np.array([dx, dy]))
            return f

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

                    # Keep a copy before blending for smoother pole fixes
                    orig_frame = frame.copy()
                    
                    # Get all target faces in this frame
                    target_faces = get_faces(frame)
                    if target_faces is None:
                        print("[DEBUG] get_faces returned None")
                        target_faces = []
                    
                    if target_faces:
                        for target_face in target_faces:
                            try:
                                best_source = select_best_source(target_face)
                                if best_source is None:
                                    continue
                                source_face = best_source['data']
                                if source_face is None:
                                    print("[DEBUG] source_face is None for target", target_face)
                                    continue

                                # For extreme poles/close-ups, optionally try perspective swap to reduce distortion
                                if core.globals.perspective_poles and needs_pole_stabilization(target_face.bbox, frame.shape):
                                    perspective_frame = swap_in_perspective(frame, target_face, source_face)
                                    if perspective_frame is not None:
                                        frame = perspective_frame
                                        continue
                                
                                # Handle 3D faces with negative coordinates by extending frame and shifting landmarks
                                bbox = target_face.bbox
                                h, w = frame.shape[:2]
                                x1, y1, x2, y2 = bbox

                                # Skip faces completely outside frame
                                if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
                                    continue

                                # Skip faces that are way too far out
                                if y2 < -200 or x2 < -200 or x1 > w + 200 or y1 > h + 200:
                                    print(f"[DEBUG] Skipping far out-of-bounds face (bbox={bbox}, frame_shape={frame.shape[:2]})")
                                    continue

                                # Skip too small faces
                                if (x2 - x1) <= 10 or (y2 - y1) <= 10:
                                    print(f"[DEBUG] Skipping too small bbox (bbox={bbox})")
                                    continue

                                # Handle faces with negative coordinates by padding frame
                                pad_top = max(0, int(-y1) + 10)
                                pad_left = max(0, int(-x1) + 10)
                                pad_bottom = max(0, int(y2 - h) + 10)
                                pad_right = max(0, int(x2 - w) + 10)

                                if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
                                    # Extend frame with padding
                                    padded_frame = cv2.copyMakeBorder(
                                        frame, pad_top, pad_bottom, pad_left, pad_right,
                                        cv2.BORDER_REPLICATE
                                    )

                                    # Adjust target face bbox and landmarks for padded frame
                                    adjusted_face = shifted_face(target_face, pad_left, pad_top)

                                    # Swap on padded frame
                                    try:
                                        swapped_padded = safe_swap_call(padded_frame, adjusted_face, source_face, paste_back=True)
                                        # Extract result back to original frame size
                                        frame = swapped_padded[pad_top:pad_top+h, pad_left:pad_left+w]
                                    except Exception as swap_err:
                                        print(f"[DEBUG] Padded swap failed (bbox={bbox}, pads={(pad_top, pad_bottom, pad_left, pad_right)}, adjusted_bbox={adjusted_face.bbox}): {swap_err}")
                                        continue
                                else:
                                    # Normal swap for faces within bounds
                                    try:
                                        frame = safe_swap_call(frame, target_face, source_face, paste_back=True)
                                    except Exception as swap_err:
                                        print(f"[DEBUG] Normal swap failed (bbox={bbox}): {swap_err}")
                                        continue

                                # Post color + expression-preserving blend in bbox region
                                x1c, y1c, x2c, y2c = map(int, [x1, y1, x2, y2])
                                x1c = max(0, x1c); y1c = max(0, y1c)
                                x2c = min(frame.shape[1], x2c); y2c = min(frame.shape[0], y2c)
                                if x2c > x1c and y2c > y1c:
                                    swap_patch = frame[y1c:y2c, x1c:x2c]
                                    orig_patch = orig_frame[y1c:y2c, x1c:x2c]
                                    try:
                                        colored = adaptive_color_blend(orig_patch, swap_patch, mask=None, method="auto")
                                        # Add a small amount of original detail to keep expressions
                                        detail = orig_patch.astype(np.float32) - cv2.GaussianBlur(orig_patch, (0, 0), 1.2)
                                        restored = np.clip(colored.astype(np.float32) + 0.18 * detail, 0, 255).astype(np.uint8)
                                        band = max(6, min(24, min(swap_patch.shape[0], swap_patch.shape[1]) // 5))
                                        mask = soft_bbox_mask(swap_patch.shape[0], swap_patch.shape[1], band)
                                        mask3 = mask[:, :, None]
                                        blended = swap_patch.astype(np.float32) * (1 - mask3) + restored.astype(np.float32) * mask3
                                        frame[y1c:y2c, x1c:x2c] = np.clip(blended, 0, 255).astype(np.uint8)
                                    except Exception as blend_err:
                                        print(f"[DEBUG] Color/detail blend failed: {blend_err}")
                                
                                # RTX 4060 Ti: GPU-accelerated border blur
                                bbox = target_face.bbox
                                if use_gpu_blur:
                                    frame = apply_border_blur_gpu(frame, bbox, blur_strength=15, device='cuda')
                                else:
                                    frame = apply_border_blur(frame, bbox, blur_strength=15)

                                # Stabilize extreme close-ups near poles with seamless cloning
                                if needs_pole_stabilization(bbox, frame.shape):
                                    x1, y1, x2, y2 = map(int, bbox)
                                    x1 = max(0, x1); y1 = max(0, y1)
                                    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
                                    if x2 > x1 and y2 > y1:
                                        patch = frame[y1:y2, x1:x2]
                                        mask = np.ones(patch.shape[:2], dtype=np.uint8) * 255
                                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                                        try:
                                            frame = cv2.seamlessClone(patch, frame, mask, center, cv2.MIXED_CLONE)
                                        except cv2.error:
                                            pass
                                
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
                    import traceback
                    traceback.print_exc()
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

    parser.add_argument("--detector", choices=['auto', 'l', 'm', 's'], default='auto',
                       help="Face detector size: auto (default), l=buffalo_l, m=buffalo_m, s=buffalo_s")

    parser.add_argument("--perspective-poles", action="store_true", default=False,
                       help="Try perspective reproject swap for extreme pole closeups (experimental)")
    
    parser.add_argument("--execution-provider", choices=['cuda', 'tensorrt', 'cpu'], default='cuda',
                       help="Execution provider (default: cuda)")
    
    args = parser.parse_args()
    
    # Initialize GPU settings
    if not args.gpu or args.execution_provider == 'cpu':
        args.gpu = False
        core.globals.providers = ['CPUExecutionProvider']
        core.globals.provider_options = [{}]
        print("[INFO] Using CPU mode")
    elif args.execution_provider == 'tensorrt':
        core.globals.enable_tensorrt(fp16=True)
        print("[INFO] Using GPU mode (TensorRT) with engine caching")
    else:
        core.globals.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        core.globals.provider_options = [
            {
                'device_id': '0',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': '1',
            },
            {}
        ]
        print("[INFO] Using GPU mode (CUDA)")
    
    # Propagate detector choice
    core.globals.detector_override = args.detector
    core.globals.perspective_poles = args.perspective_poles

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
