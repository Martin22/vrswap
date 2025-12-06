"""Border Blur Module - Post-processing pro paste_back=True
Implementuje VisoMaster border blending strategie
RTX 4060 Ti OPTIMIZED - GPU accelerated kernels
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def apply_border_blur_gpu(frame, bbox, blur_strength=12, device='cuda'):
    """GPU-accelerated border blur pro RTX 4060 Ti - 3-5x rychlejší
    
    Args:
        frame: Frame s již aplikovaným swapem (paste_back=True)
        bbox: Bounding box tváře (x1, y1, x2, y2)
        blur_strength: Síla bluuru (default 15)
        device: 'cuda' nebo 'cpu'
    
    Returns:
        Frame s vyhladenými okraji
    """
    if not isinstance(frame, np.ndarray) or device != 'cuda':
        return apply_border_blur(frame, bbox, blur_strength)  # Fallback CPU
    
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = y2 - y1, x2 - x1
        
        if h <= 0 or w <= 0 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return frame
        
        # Full-frame mask to avoid visible ROI box
        frame_h, frame_w = frame.shape[:2]
        frame_tensor = torch.from_numpy(frame).to(device).float()
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

        mask = torch.zeros((1, 1, frame_h, frame_w), device=device)
        mask[0, 0, y1:y2, x1:x2] = 1.0
        
        # Erosion using max_pool2d (inverse)
        feather_radius = max(2, min(6, min(h, w) // 10))
        erosion_kernel_size = 2 * feather_radius + 1
        # Erosion = 1 - max_pool(-mask)
        mask_inv = 1.0 - mask
        mask_inv_eroded = F.max_pool2d(mask_inv, kernel_size=erosion_kernel_size, 
                                       stride=1, padding=erosion_kernel_size//2)
        mask_eroded = 1.0 - mask_inv_eroded
        
        # Gaussian blur (Rope-next style)
        kernel_size = max(3, feather_radius * 2 + 1)
        sigma = max(0.8, feather_radius * 0.4 + 0.8)
        
        # Create Gaussian kernel
        x = torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1, device=device, dtype=torch.float32)
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_kernel = gaussian_1d.view(1, -1) * gaussian_1d.view(-1, 1)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        
        # Apply Gaussian blur to mask and frame
        mask_soft = F.conv2d(mask_eroded, gaussian_kernel, padding=kernel_size//2)
        mask_soft = torch.clamp(mask_soft, 0, 1)
        mask_soft = mask_soft / (mask_soft.amax() + 1e-6)
        # Edge band (outer minus eroded inner)
        band_kernel = max(3, min(7, feather_radius * 2 - 1))
        inner = 1.0 - F.max_pool2d(1.0 - mask_soft, kernel_size=band_kernel, stride=1, padding=band_kernel//2)
        edge = torch.clamp(mask_soft - inner, 0, 1)

        # Blur whole frame lightly
        gaussian_kernel_3ch = gaussian_kernel.expand(3, 1, -1, -1)
        frame_blurred = F.conv2d(frame_tensor, gaussian_kernel_3ch, padding=kernel_size//2, groups=3)

        # Blend only on edge band
        edge_3ch = edge.expand(1, 3, -1, -1)
        result = frame_tensor + edge_3ch * (frame_blurred - frame_tensor)
        
        # Convert back
        result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result
    
    except Exception as e:
        print(f"[DEBUG] GPU border blur failed, fallback to CPU: {e}")
        return apply_border_blur(frame, bbox, blur_strength)


def apply_border_blur(frame, bbox, blur_strength=12):
    """Aplikuj měkké rozmazání na hrany swapped tváře - dle VisoMaster VR180
    
    Args:
        frame: Frame s již aplikovaným swapem (paste_back=True)
        bbox: Bounding box tváře (x1, y1, x2, y2)
        blur_strength: Síla bluuru (default 15 dle VisoMaster, max 100)
    
    Returns:
        Frame s vyhladeneými okraji
    """
    if not isinstance(frame, np.ndarray):
        return frame
    
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = y2 - y1, x2 - x1
        
        if h <= 0 or w <= 0 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return frame
        
        frame_h, frame_w = frame.shape[:2]
        mask = np.zeros((frame_h, frame_w), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        
        # === Feathering dle Rope-next: úzký pás ===
        feather_radius = max(2, min(6, min(h, w) // 10))
        erosion_kernel_size = 2 * feather_radius + 1
        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (erosion_kernel_size, erosion_kernel_size)
        )
        mask_eroded = cv2.erode(mask, erosion_kernel, iterations=1)
        
        # 2. Gaussian blur pro feathering (malý kernel)
        kernel_size = max(3, feather_radius * 2 + 1)
        sigma = max(0.8, feather_radius * 0.4 + 0.8)
        
        # Kernel size musí být lichý a minimálně 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 1:
            kernel_size = 1
        
        # Aplikuj blur na erodovanou masku
        mask_soft = cv2.GaussianBlur(mask_eroded, (kernel_size, kernel_size), sigma)
        mask_soft = np.clip(mask_soft, 0, 1)
        if mask_soft.max() > 1e-6:
            mask_soft = mask_soft / mask_soft.max()

        # Edge band: mask_soft - eroded(mask_soft)
        band_kernel = max(3, min(7, feather_radius * 2 - 1))
        band = cv2.erode(mask_soft, np.ones((band_kernel, band_kernel), np.uint8), iterations=1)
        edge = np.clip(mask_soft - band, 0, 1)

        # Blur whole frame lightly
        frame_blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)

        # Blend only on edge band
        edge_3ch = np.repeat(edge[:, :, None], 3, axis=2)
        result = frame.astype(np.float32) + edge_3ch * (frame_blurred.astype(np.float32) - frame.astype(np.float32))
        return np.clip(result, 0, 255).astype(np.uint8)
    
    except Exception as e:
        print(f"[DEBUG] Border blur error: {e}")
        return frame
