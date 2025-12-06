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
            # Allow user override via CLI (core.globals.detector_override)
            override = getattr(core.globals, 'detector_override', 'auto')

            def pick_by_override(mem_gb):
                if override == 'l':
                    return 'buffalo_l', (608, 608)
                if override == 'm':
                    return 'buffalo_m', (512, 512)
                if override == 's':
                    return 'buffalo_s', (512, 512)
                # auto
                if mem_gb >= 12:
                    return 'buffalo_l', (608, 608)
                if mem_gb >= 8:
                    return 'buffalo_m', (512, 512)
                return 'buffalo_s', (512, 512)

            if core.globals.device == 'cuda':
                try:
                    device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    model_name, det_size = pick_by_override(device_memory)
                except:
                    model_name, det_size = pick_by_override(12)
            else:
                model_name = 'buffalo_s'
                det_size = (512, 512)
            
            # Use CUDA for analyser even when swapper runs TensorRT (TRT can be slower/unstable on analyser)
            analyser_providers = core.globals.providers
            analyser_opts = core.globals.provider_options
            if 'TensorrtExecutionProvider' in analyser_providers:
                # Drop TRT for analyser to improve stability
                filtered = []
                filtered_opts = []
                for p, o in zip(analyser_providers, analyser_opts if analyser_opts else []):
                    if p == 'TensorrtExecutionProvider':
                        continue
                    filtered.append(p)
                    filtered_opts.append(o)
                if filtered:
                    analyser_providers = filtered
                    analyser_opts = filtered_opts if filtered_opts else None

            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name=model_name,
                providers=analyser_providers,
                provider_options=analyser_opts
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