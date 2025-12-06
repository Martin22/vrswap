import insightface
import core.globals
import torch

FACE_ANALYSER = None


def get_face_analyser():
    """
    Vrací optimalizovaný face analyser.
    RTX 4060 Ti OPTIMIZED: buffalo_l model s balanced det_size pro rychlost.
    """
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        try:
            # RTX 4060 Ti (16GB) - použij buffalo_l pro nejlepší kvalitu
            if core.globals.device == 'cuda':
                try:
                    device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    if device_memory >= 12:
                        model_name = 'buffalo_l'  # RTX 4060 Ti má 16GB → best quality
                        det_size = (640, 640)  # stabilní pro buffalo_l
                    elif device_memory >= 8:
                        model_name = 'buffalo_m'
                        det_size = (512, 512)
                    else:
                        model_name = 'buffalo_s'
                        det_size = (512, 512)
                except:
                    model_name = 'buffalo_l'
                    det_size = (640, 640)
            else:
                model_name = 'buffalo_s'
                det_size = (512, 512)
            
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name=model_name,
                providers=core.globals.providers,
                provider_options=core.globals.provider_options
            )
            
            # RTX 4060 Ti: 640x640 je sweet spot - dobré detection + rychlost
            ctx_id = 0 if core.globals.device == 'cuda' else -1
            FACE_ANALYSER.prepare(ctx_id=ctx_id, det_size=det_size)
            
            print(f"[INFO] Face analyzer: {model_name}, det_size={det_size}")
            
        except Exception as e:
            print(f"Face analyzer init error: {e}")
            FACE_ANALYSER = None
    
    return FACE_ANALYSER


def get_face(img_data):
    """Detekuje jednu tvář (nejlevější)."""
    analyser = get_face_analyser()
    if analyser is None:
        return None
    
    try:
        # FP16 support když je dostupná
        if core.globals.use_fp16 and core.globals.device == 'cuda':
            with torch.autocast('cuda'):
                faces = analyser.get(img_data)
        else:
            faces = analyser.get(img_data)
        
        if faces:
            return sorted(faces, key=lambda x: x.bbox[0])[0]
    except Exception as e:
        print(f"Face detection error: {e}")
    
    return None


def get_faces(img_data):
    """Detekuje všechny tváře."""
    analyser = get_face_analyser()
    if analyser is None:
        return None
    
    try:
        # FP16 support když je dostupná
        if core.globals.use_fp16 and core.globals.device == 'cuda':
            with torch.autocast('cuda'):
                faces = analyser.get(img_data)
        else:
            faces = analyser.get(img_data)
        
        if faces:
            return sorted(faces, key=lambda x: x.bbox[0])
    except Exception as e:
        print(f"Face detection error: {e}")
    
    return None