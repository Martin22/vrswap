import os
import subprocess
import shutil
from tqdm import tqdm
import cv2
import insightface
import core.globals
from core.analyser import get_face
import torch
import numpy as np
import traceback

FACE_SWAPPER = None


def get_face_swapper():
    """Načte face swapper model s optimalizacemi."""
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../inswapper_128_fp16.onnx')

        # Build provider list locally and drop TensorRT for stability (TRT sometimes yields empty outputs)
        providers = list(core.globals.providers or ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        provider_options = core.globals.provider_options
        if 'TensorrtExecutionProvider' in providers:
            trt_idx = providers.index('TensorrtExecutionProvider')
            providers = [p for p in providers if p != 'TensorrtExecutionProvider']
            if isinstance(provider_options, list) and len(provider_options) > trt_idx:
                provider_options = [opt for i, opt in enumerate(provider_options) if i != trt_idx]
            print("[INFO] TensorRT disabled for face swapper (stability fix); using", providers)

        # Load model using insightface (pass provider_options if available)
        try:
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=providers,
                provider_options=provider_options
            )
            # Verify model loaded correctly and has .get method
            if FACE_SWAPPER is None:
                print("[ERROR] insightface.model_zoo.get_model returned None")
            elif not hasattr(FACE_SWAPPER, 'get') or not callable(getattr(FACE_SWAPPER, 'get', None)):
                print(f"[ERROR] Loaded model does not have callable .get method. Type: {type(FACE_SWAPPER)}")
            else:
                print(f"[INFO] Face swapper loaded successfully: {type(FACE_SWAPPER).__name__}")
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise e

    return FACE_SWAPPER


def get_swapped_face(frame, target_face, source_face):
    """Perform a single face swap using the global swapper."""
    swapper = get_face_swapper()

    try:
        if core.globals.use_fp16 and core.globals.device == 'cuda':
            with torch.autocast('cuda'):
                result = swapper.get(frame, target_face, source_face, paste_back=True)
        else:
            result = swapper.get(frame, target_face, source_face, paste_back=True)

        if core.globals.device == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return result
    except Exception as e:
        print(f"Face swap error: {e}")
        traceback.print_exc()
        return frame