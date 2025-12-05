import onnxruntime
import platform

# Windows 11 + Python 3.12 compatible settings
use_gpu = False
providers = onnxruntime.get_available_providers()

# Remove suboptimal providers for Windows stability
if 'TensorrtExecutionProvider' in providers:
    providers.remove('TensorrtExecutionProvider')

# Přidat CPU executor na konec pro fallback (důležité pro Windows)
if 'CPUExecutionProvider' not in providers:
    providers.append('CPUExecutionProvider')

# GPU optimization (Windows compatible)
try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_fp16 = torch.cuda.is_available()  # FP16 jen na CUDA
    
    # GPU Settings (auto-detect)
    if torch.cuda.is_available():
        # TensorFloat-32 optimization (safe on all GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Memory settings
        try:
            torch.cuda.set_per_process_memory_fraction(0.9)
        except:
            pass  # Windows sometimes has issues with this
            
except ImportError:
    device = 'cpu'
    use_fp16 = False
