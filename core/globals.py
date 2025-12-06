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
    
    # RTX 4060 Ti OPTIMALIZACE - Aggressive FP16 + Memory Management
    if torch.cuda.is_available():
        try:
            # Test FP16 compatibility
            test_tensor = torch.randn(1, 3, 256, 256, device='cuda', dtype=torch.float16)
            del test_tensor
            use_fp16 = True
            print("[INFO] FP16 enabled - RTX 4060 Ti optimized mode")
        except Exception as e:
            print(f"[WARNING] FP16 not supported on this GPU, falling back to FP32: {e}")
            use_fp16 = False
        
        # RTX 4060 Ti GPU Settings - Maximální výkon
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Auto-tune kernels
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False  # Rychlejší, ne deterministické
        
        # Memory settings pro RTX 4060 Ti (16GB)
        # Ponechat 2GB pro systém → využít 14GB = 0.875
        try:
            torch.cuda.set_per_process_memory_fraction(0.875)  # 14GB z 16GB
            torch.cuda.empty_cache()
        except:
            pass
        
        # Povolit memory growth (TensorFlow style)
        try:
            torch.cuda.set_sync_debug_mode(0)  # Disable sync pro rychlost
        except:
            pass
            
except ImportError:
    device = 'cpu'
    use_fp16 = False

