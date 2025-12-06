import os
import subprocess
import shutil
from tqdm import tqdm
import cv2
import insightface
import core.globals
from core.analyser import get_face
import torch

FACE_SWAPPER = None


def get_face_swapper():
    """Načte face swapper model s optimalizacemi."""
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../inswapper_128_fp16.onnx')
        
        # Load model using insightface (pass provider_options if available)
        try:
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=core.globals.providers,
                provider_options=core.globals.provider_options
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    return FACE_SWAPPER


def get_swapped_face(frame, target_face, source_face):
    """
    Perform face swap s optimalizacemi.
    
    Args:
        frame: Source frame
        target_face: Target face to replace
        source_face: Source face to insert
        
    Returns:
        Swapped frame
    """
    swapper = get_face_swapper()
    
    try:
        # FP16 optimization pokud je dostupná
        if core.globals.use_fp16 and core.globals.device == 'cuda':
            with torch.autocast('cuda'):
                result = swapper.get(frame, target_face, source_face, paste_back=True)
        else:
            result = swapper.get(frame, target_face, source_face, paste_back=True)
        
        # Clear cache
        if core.globals.device == 'cuda':
            try:
                torch.cuda.empty_cache()
            except:
                pass
        
        return result
    except Exception as e:
        print(f"Face swap error: {e}")
        return frame