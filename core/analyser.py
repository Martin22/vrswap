import insightface
import core.globals
import torch

FACE_ANALYSER = None


def get_face_analyser():
    """
    Vrací optimalizovaný face analyser.
    Windows 11 + Python 3.12 compatible.
    """
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        try:
            # Auto-select model podle dostupné paměti
            if core.globals.device == 'cuda':
                try:
                    device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    if device_memory >= 16:
                        model_name = 'buffalo_l'  # Best quality
                    elif device_memory >= 8:
                        model_name = 'buffalo_m'  # Balanced
                    else:
                        model_name = 'buffalo_s'  # Fast
                except:
                    model_name = 'buffalo_l'
            else:
                model_name = 'buffalo_s'  # CPU mode - use small model
            
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name=model_name,
                providers=core.globals.providers
            )
            
            # Prepare with appropriate context
            ctx_id = 0 if core.globals.device == 'cuda' else -1
            FACE_ANALYSER.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
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
            with torch.cuda.amp.autocast():
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
            with torch.cuda.amp.autocast():
                faces = analyser.get(img_data)
        else:
            faces = analyser.get(img_data)
        
        if faces:
            return sorted(faces, key=lambda x: x.bbox[0])
    except Exception as e:
        print(f"Face detection error: {e}")
    
    return None