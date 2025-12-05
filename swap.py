import os
import time
import glob
import subprocess
import argparse
import sys
from PIL import Image as PILImage
import shutil
import torch
import insightface
import core.globals
from pathlib import Path
from core.swapper import get_face_swapper
from core.analyser import get_face, get_faces, get_face_analyser
from threading import Thread
import threading
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from core.lib import Equirec2Perspec as E2P, Perspec2Equirec as P2E
from math import pi
import numpy as np
import json
from core.advanced_blending import AdvancedFaceBlender

# Optional imports s fallback
try:
    import cupy as cp
    from numba import cuda
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

if 'ROCMExecutionProvider' in core.globals.providers:
    del torch

# Initialize argument parser
parser = argparse.ArgumentParser(description="VRSwap - Equirectangular Face Swap")
parser.add_argument("--frames_folder", help="Frames folder", required=True)
parser.add_argument("--face", help="Source Face", required=True)
parser.add_argument("--gpu_threads", help="Processing threads", default=5, type=int)
parser.add_argument('--gpu', help='use GPU acceleration', dest='gpu', action='store_true', default=True)
parser.add_argument('--cpu', help='force CPU mode', dest='gpu', action='store_false')
parser.add_argument('--batch_size', help='batch size for processing', default=4, type=int)
parser.add_argument('--tile_size', help='tile size for 8K (0=disable)', default=512, type=int)
parser.add_argument('--fast', help='Fast mode - skip color matching', dest='fast_mode', action='store_true', default=False)
args = parser.parse_args()

framesFolder = args.frames_folder
sourceFace = args.face
gpuThreads = args.gpu_threads
batchSize = args.batch_size
tileSize = args.tile_size
fastMode = args.fast_mode

# Create a lock for thread-safe file writing
lock = threading.Lock()

# Windows + Linux path compatibility
sep = "\\" if os.name == "nt" else "/"


def process_tile(tile_img, source_face, swapper):
    """Procesuje jeden tile s FP16 optimalizací"""
    if core.globals.use_fp16 and core.globals.device == 'cuda':
        with torch.autocast('cuda'):
            target_faces = get_faces(tile_img)
            if target_faces:
                for target_face in target_faces:
                    tile_img = swapper.get(tile_img, target_face, source_face, paste_back=True)
    else:
        target_faces = get_faces(tile_img)
        if target_faces:
            for target_face in target_faces:
                tile_img = swapper.get(tile_img, target_face, source_face, paste_back=True)
    
    return tile_img


def split_into_tiles(frame, tile_size=512):
    """Rozdělí frame na tiles s overlapem pro měkký blend"""
    h, w = frame.shape[:2]
    tiles = []
    positions = []
    
    overlap = 50  # Overlap pro blending
    step = tile_size - overlap
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = frame[y:y_end, x:x_end]
            tiles.append(tile)
            positions.append((y, x, y_end, x_end))
    
    return tiles, positions


def merge_tiles(tiles, positions, original_shape):
    """Sloučí zpracované tiles s overlap blending - OPRAVENO"""
    result = np.zeros(original_shape, dtype=np.float32)
    weight_sum = np.zeros(original_shape[:2], dtype=np.float32)
    
    overlap = 50
    
    for tile, (y, x, y_end, x_end) in zip(tiles, positions):
        h = y_end - y
        w = x_end - x
        
        # Vytvořit feathering masku pro měkké přechody
        feather_mask = np.ones((h, w), dtype=np.float32)
        
        # Měkké hrany - distance map
        feather_width = min(overlap, h // 4, w // 4)
        if feather_width > 0:
            dist_y = np.minimum(np.arange(h), h - np.arange(h))
            dist_x = np.minimum(np.arange(w), w - np.arange(w))
            
            dist_field = np.minimum(
                np.minimum(dist_y[:, np.newaxis], dist_x[np.newaxis, :]),
                feather_width
            )
            
            feather_mask = dist_field / (feather_width + 1e-6)
            feather_mask = np.clip(feather_mask, 0, 1)
        
        # Aplikovat masku
        tile_masked = tile.astype(np.float32) * feather_mask[:,:,np.newaxis]
        feather_mask_3d = np.stack([feather_mask] * 3, axis=-1)
        
        result[y:y_end, x:x_end] += tile_masked
        weight_sum[y:y_end, x:x_end] += feather_mask_3d[:,:,0]
    
    # Normalizovat
    weight_sum = np.maximum(weight_sum, 1e-6)
    for c in range(result.shape[2]):
        result[:, :, c] /= weight_sum
    
    return np.clip(result, 0, 255).astype(np.uint8)


def pre_check():
    """Kontrola předpokladů pro spuštění na Windows 11 + Python 3.12"""
    if sys.version_info < (3, 8):
        quit('Python version is not supported - please upgrade to 3.8 or higher')
    
    if not shutil.which('ffmpeg'):
        quit('ffmpeg is not installed!')
    
    # Check model
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'inswapper_128_fp16.onnx')
    if not os.path.isfile(model_path):
        quit('File "inswapper_128_fp16.onnx" does not exist!')
    
    # GPU check - simplified pro Windows 11 stability
    if args.gpu:
        try:
            if not torch.cuda.is_available():
                print("[WARNING] CUDA not available, falling back to CPU")
                core.globals.providers = ['CPUExecutionProvider']
                args.gpu = False
            else:
                cuda_version = torch.version.cuda
                print(f"[INFO] Using CUDA {cuda_version}")
                # Windows 11 CUDA 11.8 je optimální
                if cuda_version and ('11.8' not in cuda_version and '12.' not in cuda_version):
                    print(f"[WARNING] CUDA {cuda_version} detected, 11.8 or 12.x recommended")
        except Exception as e:
            print(f"[WARNING] GPU check failed: {e}, using CPU")
            core.globals.providers = ['CPUExecutionProvider']
            args.gpu = False
    else:
        core.globals.providers = ['CPUExecutionProvider']


#creates a thread and returns value when joined
class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def face_analyser_thread(frame_path, source_face, vr = True):
    yes_face = True
    result = None  # Initialize result
    
    # Load the frame
    frame = cv2.imread(frame_path)
    
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]  # 000001
    frame_folder = os.path.splitext(frame_path)[0]                  # D:/test/000001
    output_folder = os.path.dirname(frame_path)                     # D:/test
    processing_folder = output_folder + "/processing"
    
    result = equir2pers(frame_path, processing_folder)

    left = f"{processing_folder}/{frame_name}_L.jpg"
    right = f"{processing_folder}/{frame_name}_R.jpg"

    img1 = perform_face_swap(left, source_face)
    img2 = perform_face_swap(right, source_face)

    return yes_face, result


def process_frames(source_img, frame_paths):
    """
    Procesuje framy s maximální rychlostí - optimalizovaná verze.
    Bez zbytečných threadů, přímé GPU zpracování.
    """
    swapper = get_face_swapper()
    source_frame = cv2.imread(source_img)
    source_face = get_face(source_frame)
    
    if source_face is None:
        print("[ERROR] Couldn't detect source face")
        return

    print("[INFO] Processing frames with GPU acceleration...")
    
    with tqdm(total=len(frame_paths), desc='Processing', unit="frame", 
              dynamic_ncols=True, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as progress:
        for frame_path in frame_paths:
            try:
                # Čti frame
                frame = cv2.imread(frame_path)
                if frame is None:
                    progress.update(1)
                    continue
                
                # Detekuj tváře
                target_faces = get_faces(frame)
                if not target_faces:
                    progress.update(1)
                    continue
                
                # Swap s FP16 supportem
                result = frame.copy()
                for target_face in target_faces:
                    if core.globals.use_fp16 and core.globals.device == 'cuda':
                        with torch.autocast('cuda'):
                            swapped = swapper.get(frame, target_face, source_face, paste_back=False)
                    else:
                        swapped = swapper.get(frame, target_face, source_face, paste_back=False)
                    
                    # Ujisti se že swapped je numpy array
                    if swapped is None:
                        continue
                    if isinstance(swapped, (tuple, list)):
                        # swapper.get() někdy vrací tuple (frame, landmarks)
                        swapped = swapped[0] if isinstance(swapped[0], np.ndarray) else swapped
                    if not isinstance(swapped, np.ndarray):
                        continue
                    
                    # Pokročilý blend - eliminuje artefakty
                    bbox = target_face.bbox
                    result = AdvancedFaceBlender.blend_faces_advanced(result, swapped, bbox, expand_ratio=1.35, use_color_match=(not fastMode))
                
                # Ulož
                cv2.imwrite(frame_path, result)
                
                # GPU cleanup
                if core.globals.device == 'cuda':
                    torch.cuda.empty_cache()
                
                progress.update(1)
                
            except Exception as e:
                print(f"[ERROR] {os.path.basename(frame_path)}: {e}")
                import traceback
                traceback.print_exc()
                progress.update(1)
                continue



def perform_face_swap(frame_path, source_face, swapper, use_tiling=False, tile_size=512):
    """
    Optimalizovaný swap s FP16, pokročilým blendingem a tile supportem pro 8K.
    OPRAVENO: Eliminuje rámečky kolem obličeje, tvrdé přechody, nerealistické barvy.
    """
    frame_path = os.path.normpath(frame_path)
    
    if not os.path.exists(frame_path):
        return None
    
    try:
        frame = cv2.imread(frame_path)
        if frame is None:
            return None
        
        target_faces = get_faces(frame)
        if not target_faces:
            return frame
        
        result = frame.copy()
        
        for target_face in target_faces:
            swapped_temp = swapper.get(frame, target_face, source_face, paste_back=False)
            bbox = target_face.bbox
            result = AdvancedFaceBlender.blend_faces_advanced(result, swapped_temp, bbox, expand_ratio=1.35)
        
        cv2.imwrite(frame_path, result)
        return result
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return None
            
    except Exception as e:
        print(f"[ERROR] Face swap error: {e}")
        return None



def extractFace(frame_name, input_img, face, output_dir, side):
    """Extract obličej s vylepšeným boundingem - OPRAVENO"""
    bbox = face.bbox

    # Load equirectangular image
    equ = E2P.Equirectangular(input_img)   

    # Convert bounding box to ints
    x1, y1, x2, y2 = map(int, bbox)

    # OPRAVENO: Zvětšit bbox o 25% pro měkký blend
    h = y2 - y1
    w = x2 - x1
    
    expand_ratio = 0.25
    y1_expanded = max(0, int(y1 - h * expand_ratio))
    x1_expanded = max(0, int(x1 - w * expand_ratio))
    y2_expanded = min(equ.get_height(), int(y2 + h * expand_ratio))
    x2_expanded = min(equ.get_width(), int(x2 + w * expand_ratio))

    # Determine the center of the bounding box
    x_center = (x1_expanded + x2_expanded) / 2
    y_center = (y1_expanded + y2_expanded) / 2

    # Normalize coordinates to range [-1, 1]
    x_center_normalized = x_center / (equ.get_width() / 2) - 1
    y_center_normalized = y_center / (equ.get_height() / 2) - 1

    # Convert normalized coordinates to spherical (theta, phi)
    theta = x_center_normalized * 180
    phi = -y_center_normalized * 90

    img = equ.GetPerspective(90, theta, phi, 1280, 1280)
    output_path = os.path.join(output_dir, f'{frame_name}_{side}.jpg')
    cv2.imwrite(output_path, img)
    storeInfo(frame_name, side, output_dir, theta, phi)


def storeInfo(frame_name, side, output_dir, theta, phi):
    """Store metadata with Windows path compatibility"""
    exif_data = {
        f'theta{side}': str(theta),
        f'phi{side}': str(phi)
    }

    parent_dir = os.path.dirname(output_dir)
    data_file = os.path.join(parent_dir, '_data.json')

    with lock:
        data = {}
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load data file: {e}")
        
        if frame_name in data:
            data[frame_name].update(exif_data)
        else:
            data[frame_name] = exif_data
        
        try:
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save data file: {e}")


def loadInfo(frame_number, output_dir, side):
    """Load metadata with Windows path compatibility"""
    parent_dir = os.path.dirname(output_dir)
    data_file = os.path.join(parent_dir, '_data.json')

    if not os.path.exists(data_file):
        raise ValueError(f"Data file not found: {data_file}")

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if frame_number in data:
        theta = float(data[frame_number][f'theta{side}'])
        phi = float(data[frame_number][f'phi{side}'])
        return theta, phi
    else:
        raise ValueError(f"Frame number {frame_number} not found in {data_file}")



def equir2pers(input_img, output_dir):
    """Converts equirectangular image to perspective with Windows path support"""
    frame_name = os.path.splitext(os.path.basename(input_img))[0]

    img = cv2.imread(input_img)
    if img is None:
        print(f"[ERROR] Can't read image: {input_img}")
        return
    
    faces = get_faces(img)
    if not faces:
        print(f"[DEBUG] No faces in {frame_name}")
        return

    width = img.shape[1]

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        x_center = (x1 + x2) / 2

        if x_center < width / 2:
            extractFace(frame_name, input_img, face, output_dir, "L") 
        else:
            extractFace(frame_name, input_img, face, output_dir, "R") 


if __name__ == '__main__':
    try:
        pre_check()

        # Windows path normalization
        framesFolder = os.path.normpath(framesFolder)
        sourceFace = os.path.normpath(sourceFace)

        processingPath = os.path.join(framesFolder, "processing")

        framePaths = []
        for framePath in glob.glob(os.path.join(framesFolder, "*.jpg")):
            if not framePath.endswith('_p.jpg'):
                framePaths.append(framePath)

        framePaths = tuple(sorted(framePaths, key=lambda x: int(os.path.basename(x).replace(".jpg", ""))))

        if not framePaths:
            print("[ERROR] No frames found in folder")
            sys.exit(1)

        print(f"[INFO] Found {len(framePaths)} frames")
        print(f"[INFO] Processing {sourceFace}")
        print(f"[INFO] Device: {core.globals.device}")
        print(f"[INFO] FP16 enabled: {core.globals.use_fp16}")
        print(f"[INFO] Fast mode: {fastMode}")
        print("[INFO] Swapping in progress...")
        
        process_frames(sourceFace, framePaths)
        
        print("[INFO] Processing completed!")
        if core.globals.device == 'cuda':
            torch.cuda.empty_cache()
            
    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def enhance_output_quality(frame, denoise=True, sharpen=False):
    """Post-processing pro vyšší kvalitu výstupu"""
    result = frame.copy().astype(np.float32) / 255.0
    
    if denoise:
        frame_uint8 = (result * 255).astype(np.uint8)
        result = cv2.bilateralFilter(frame_uint8, 9, 75, 75).astype(np.float32) / 255.0
    
    if sharpen:
        blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
        result = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)