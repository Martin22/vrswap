import os
from pathlib import Path
import onnxruntime
import platform

# Providers and options (can be overridden by caller)
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
provider_options = None

def _ensure_cache_dir():
    cache_dir = Path.cwd() / '.trt_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def enable_tensorrt(fp16=True):
    """Configure global env + provider options for TensorRT with caching.
    Uses ORT engine cache + timing cache to avoid rebuild every run.
    """
    cache_dir = _ensure_cache_dir()
    timing_cache = cache_dir / 'trt_timing.cache'

    os.environ.setdefault('ORT_TENSORRT_ENGINE_CACHE_ENABLE', '1')
    os.environ.setdefault('ORT_TENSORRT_ENGINE_CACHE_PATH', str(cache_dir))
    os.environ.setdefault('ORT_TENSORRT_TIMING_CACHE_ENABLE', '1')
    os.environ.setdefault('ORT_TENSORRT_TIMING_CACHE_PATH', str(timing_cache))
    if fp16:
        os.environ.setdefault('ORT_TENSORRT_FP16_ENABLE', '1')

    # Keep only ONNX Runtime 1.17-compatible TensorRT options (booleans as True/False strings)
    trt_opts = {
        'device_id': '0',
        'trt_engine_cache_enable': 'True',
        'trt_engine_cache_path': str(cache_dir),
        'trt_timing_cache_enable': 'True',
        'trt_timing_cache_path': str(timing_cache),
        'trt_fp16_enable': 'True' if fp16 else 'False',
        'trt_builder_optimization_level': '4',  # aggressive but stable on 4060 Ti
        'trt_max_workspace_size': str(1 << 31),  # 2GB workspace cap
        'trt_cuda_graph_enable': 'True',         # reduce launch overhead
        'trt_sparsity_enable': 'True',           # allow sparse kernels if supported
        'trt_dla_enable': 'False',               # no DLA on 4060 Ti
        # Removed experimental/unsupported flags (e.g., trt_ep_context_enable)
    }

    cuda_opts = {
        'device_id': '0',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': '1',
    }

    # Expose to global vars
    global providers, provider_options
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    provider_options = [trt_opts, cuda_opts, {}]


# Windows 11 + Python 3.12 compatible settings
use_gpu = False
available_providers = onnxruntime.get_available_providers()
if 'CUDAExecutionProvider' in available_providers:
    use_gpu = True

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

