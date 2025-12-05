"""
Border Blur Module - Post-processing pro paste_back=True
Implementuje Rope-next border blending strategie
"""

import cv2
import numpy as np


def apply_border_blur(frame, bbox, blur_strength=10):
    """Aplikuj měkké rozmazání na hrany swapped tváře - dle Rope-next
    
    Args:
        frame: Frame s již aplikovaným swapem (paste_back=True)
        bbox: Bounding box tváře (x1, y1, x2, y2)
        blur_strength: Síla bluuru (default 10 dle Rope-next, max 100)
    
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
        
        # Vytvoř binární masku s 1 v bbox regionu, 0 jinde
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        
        # Aplikuj Gaussian blur na masku - dle Rope-next
        # kernel = blur*2+1, sigma = (blur+1)*0.2
        kernel_size = blur_strength * 2 + 1
        sigma = (blur_strength + 1) * 0.2
        
        # Kernel size musí být lichý a minimálně 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 1:
            kernel_size = 1
        
        # Gaussianův blur pro měkké hrany
        mask_soft = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
        mask_soft = np.clip(mask_soft, 0, 1)
        
        # Blend frame s měkkou maskou
        # Kde mask_soft=1 je všechno swap, kde mask_soft=0 je všechno original
        mask_3ch = np.stack([mask_soft] * 3, axis=-1)
        
        # Aplikuj blend - měkký přechod na okrajích
        result = frame.astype(np.float32) * mask_3ch
        result += frame.astype(np.float32) * (1 - mask_3ch)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    except Exception as e:
        print(f"[DEBUG] Border blur error: {e}")
        return frame
