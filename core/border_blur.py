"""Border Blur Module - Post-processing pro paste_back=True
Implementuje VisoMaster border blending strategie
RTX 4060 Ti OPTIMIZED - GPU accelerated kernels
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def apply_border_blur_gpu(frame, bbox, blur_strength=15, device='cuda'):
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
        
        # Pracuj pouze v ROI okolo bbox, aby se nerozmazal celý snímek
        pad = max(blur_strength * 2, 32)
        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(frame.shape[1], x2 + pad)
        ry2 = min(frame.shape[0], y2 + pad)

        frame_roi = frame[ry1:ry2, rx1:rx2]
        roi_h, roi_w = frame_roi.shape[:2]

        # Convert to torch tensor (ROI only)
        frame_tensor = torch.from_numpy(frame_roi).to(device).float()
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        # Create mask in ROI coords
        mask = torch.zeros((1, 1, roi_h, roi_w), device=device)
        mask[0, 0, y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = 1.0
        
        # Erosion using max_pool2d (inverse)
        feather_radius = max(2, min(12, min(h, w) // 6))  # adapt na velikost bbox, aby nezmizela maska
        erosion_kernel_size = 2 * feather_radius + 1
        # Erosion = 1 - max_pool(-mask)
        mask_inv = 1.0 - mask
        mask_inv_eroded = F.max_pool2d(mask_inv, kernel_size=erosion_kernel_size, 
                                       stride=1, padding=erosion_kernel_size//2)
        mask_eroded = 1.0 - mask_inv_eroded
        
        # Gaussian blur
        kernel_size = blur_strength * 2 + 1
        sigma = (blur_strength + 1) * 0.2
        
        # Create Gaussian kernel
        x = torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1, device=device, dtype=torch.float32)
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_kernel = gaussian_1d.view(1, -1) * gaussian_1d.view(-1, 1)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        
        # Apply Gaussian blur to mask and frame
        mask_soft = F.conv2d(mask_eroded, gaussian_kernel, padding=kernel_size//2)
        mask_soft = torch.clamp(mask_soft, 0, 1)
        # Normalizace, aby centrum zůstalo ~1 i u malých bbox
        mask_soft = mask_soft / (mask_soft.amax() + 1e-6)

        # Blur the frame for edge feathering
        gaussian_kernel_3ch = gaussian_kernel.expand(3, 1, -1, -1)
        frame_blurred = F.conv2d(frame_tensor, gaussian_kernel_3ch, padding=kernel_size//2, groups=3)
        
        # Blend blurred + sharp frame using soft mask (eliminates rectangle edges)
        mask_3ch = mask_soft.expand(1, 3, -1, -1)
        result = frame_tensor * mask_3ch + frame_blurred * (1 - mask_3ch)
        
        # Convert back
        result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Vlož zpět jen ROI, zbytek snímku zůstává ostrý
        output = frame.copy()
        output[ry1:ry2, rx1:rx2] = result
        return output
    
    except Exception as e:
        print(f"[DEBUG] GPU border blur failed, fallback to CPU: {e}")
        return apply_border_blur(frame, bbox, blur_strength)


def apply_border_blur(frame, bbox, blur_strength=15):
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
        
        # ROI pouze kolem bbox, aby se nerozmazal celý frame
        pad = max(blur_strength * 2, 32)
        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(frame.shape[1], x2 + pad)
        ry2 = min(frame.shape[0], y2 + pad)

        frame_roi = frame[ry1:ry2, rx1:rx2]
        roi_h, roi_w = frame_roi.shape[:2]

        # Vytvoř binární masku s 1 v bbox regionu (v ROI souřadnicích)
        mask = np.zeros((roi_h, roi_w), dtype=np.float32)
        mask[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = 1.0
        
        # === OPRAVENO: Balanced feathering jako VisoMaster ===
        # 1. Erosion pro stažení okrajů dovnitř
        feather_radius = 12  # VisoMaster balanced value
        erosion_kernel_size = 2 * feather_radius + 1
        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (erosion_kernel_size, erosion_kernel_size)
        )
        mask_eroded = cv2.erode(mask, erosion_kernel, iterations=1)
        
        # 2. Gaussian blur pro feathering - dle Rope-next formula
        kernel_size = blur_strength * 2 + 1
        sigma = (blur_strength + 1) * 0.2
        
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
        
        # Připrav rozmazaný snímek pro měkké hrany (jen ROI)
        frame_blurred = cv2.GaussianBlur(frame_roi, (kernel_size, kernel_size), sigma)
        
        # Blend: ostrý swap uvnitř, rozmazané okraje venku
        mask_3ch = np.stack([mask_soft] * 3, axis=-1)
        result = frame_roi.astype(np.float32) * mask_3ch + frame_blurred.astype(np.float32) * (1 - mask_3ch)
        result = np.clip(result, 0, 255).astype(np.uint8)

        output = frame.copy()
        output[ry1:ry2, rx1:rx2] = result
        return output
    
    except Exception as e:
        print(f"[DEBUG] Border blur error: {e}")
        return frame
