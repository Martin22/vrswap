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

            # Try to enable ONNX Runtime IO binding for faster GPU path (skip when TensorRT is active)
            try:
                # If TensorRT EP is present, avoid monkey-patching
                if 'TensorrtExecutionProvider' not in (core.globals.providers or []):
                    sess = getattr(FACE_SWAPPER, 'model', None)
                    if sess is not None and hasattr(sess, 'io_binding') and hasattr(sess, 'run_with_iobinding'):
                        inputs = sess.get_inputs()
                        outputs = sess.get_outputs()
                        if len(inputs) >= 2 and len(outputs) >= 1:
                            input_names = [inp.name for inp in inputs[:2]]
                            output_name = outputs[0].name
                            device_type = 'cuda' if core.globals.device == 'cuda' else 'cpu'

                            def _run_with_iobinding(*args, **kwargs):
                                # ORT run signature: run(output_names=None, input_feed=None, ...)
                                try:
                                    input_feed = None
                                    if len(args) >= 2 and isinstance(args[1], dict):
                                        input_feed = args[1]
                                    if input_feed is None:
                                        input_feed = kwargs.get('input_feed') or (args[0] if args and isinstance(args[0], dict) else None)
                                    if input_feed is None:
                                        # Fallback to original run
                                        return sess._orig_run(*args, **kwargs)

                                    # Prepare IO binding
                                    ib = sess.io_binding()
                                    for name in input_names:
                                        if name not in input_feed:
                                            return sess._orig_run(*args, **kwargs)
                                        arr = np.ascontiguousarray(input_feed[name])
                                        ib.bind_input(
                                            name=name,
                                            device_type=device_type,
                                            device_id=0,
                                            element_type=np.float32,
                                            shape=arr.shape,
                                            buffer_ptr=arr.ctypes.data
                                        )

                                    # Output buffer shape is fixed for inswapper_128
                                    out_arr = np.empty((1, 3, 128, 128), dtype=np.float32)
                                    ib.bind_output(
                                        name=output_name,
                                        device_type=device_type,
                                        device_id=0,
                                        element_type=np.float32,
                                        shape=out_arr.shape,
                                        buffer_ptr=out_arr.ctypes.data
                                    )

                                    sess.run_with_iobinding(ib)
                                    return [out_arr]
                                except Exception:
                                    # On any failure, revert to original run and fall back
                                    sess.run = sess._orig_run
                                    return sess._orig_run(*args, **kwargs)

                            # Monkey-patch run while keeping original as backup
                            if not hasattr(sess, '_orig_run'):
                                sess._orig_run = sess.run
                            sess.run = _run_with_iobinding
                            print("[INFO] IO binding enabled for inswapper session")
            except Exception as io_err:
                print(f"[DEBUG] IO binding not applied (fallback to default run): {io_err}")
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