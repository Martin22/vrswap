import onnxruntime
import platform

# Windows 11 + Python 3.12 compatible settings
use_gpu = False
providers = onnxruntime.get_available_providers()

# Přidat CPU executor na konec pro fallback
if 'CPUExecutionProvider' not in providers:
    providers.append('CPUExecutionProvider')

# GPU optimization (Windows compatible)
device = 'cpu'
use_fp16 = False

try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # OPRAVENO: FP16 má fallback na FP32 pokud se nezdaří
    if torch.cuda.is_available():
        try:
            # Test FP16 compatibility
            test_tensor = torch.randn(1, 3, 256, 256, device='cuda', dtype=torch.float16)
            del test_tensor
            use_fp16 = True
        except Exception as e:
            print(f"[WARNING] FP16 not supported on this GPU, falling back to FP32: {e}")
            use_fp16 = False
        
        # GPU Settings (auto-detect)
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

