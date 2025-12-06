"""
Color Transfer Module - Inspirováno VisoMaster histogram matching
Řeší problém barevných rozdílů mezi swapped tváří a původním framem
"""

import cv2
import numpy as np
import torch


def histogram_matching_DFL_Orig(
    original_face: np.ndarray,
    swapped_face: np.ndarray,
    mask: np.ndarray = None,
    blend_amount: float = 100.0
) -> np.ndarray:
    """DFL-style histogram matching - inspirováno DeepFaceLab
    
    Args:
        original_face: Původní tvář (H, W, C) BGR
        swapped_face: Swappovaná tvář (H, W, C) BGR
        mask: Optional maska (H, W) float 0-1
        blend_amount: Síla efektu (0-100)
    
    Returns:
        Color-matched swapped face
    """
    if blend_amount <= 0:
        return swapped_face
    
    result = swapped_face.copy().astype(np.float32)
    original = original_face.astype(np.float32)
    
    # Pokud máme masku, použij ji
    if mask is not None:
        mask_3ch = np.stack([mask] * 3, axis=-1) if mask.ndim == 2 else mask
    else:
        mask_3ch = np.ones_like(result)
    
    # Pro každý kanál
    for c in range(3):
        # Vypočítat histogram pro oblast s maskou
        orig_channel = original[:, :, c]
        swap_channel = result[:, :, c]
        
        # Histogram matching
        hist_orig, bins_orig = np.histogram(orig_channel.flatten(), 256, [0, 256])
        hist_swap, bins_swap = np.histogram(swap_channel.flatten(), 256, [0, 256])
        
        # CDF
        cdf_orig = hist_orig.cumsum()
        cdf_orig = cdf_orig / cdf_orig[-1]  # Normalize
        
        cdf_swap = hist_swap.cumsum()
        cdf_swap = cdf_swap / cdf_swap[-1]
        
        # Lookup table
        lut = np.interp(cdf_swap, cdf_orig, np.arange(256))
        
        # Aplikovat LUT
        matched = np.interp(swap_channel.flatten(), np.arange(256), lut)
        matched = matched.reshape(swap_channel.shape)
        
        # Blend s original
        alpha = blend_amount / 100.0
        result[:, :, c] = swap_channel * (1 - alpha) + matched * alpha
    
    # Aplikovat masku
    result = result * mask_3ch + swapped_face.astype(np.float32) * (1 - mask_3ch)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def color_transfer_mean_std(
    source: np.ndarray,
    target: np.ndarray,
    blend_strength: float = 0.7
) -> np.ndarray:
    """Color transfer pomocí mean/std matching
    
    Args:
        source: Source image (referenční barvy)
        target: Target image (upravovaný)
        blend_strength: Síla efektu (0-1)
    
    Returns:
        Color-transferred target
    """
    source_float = source.astype(np.float32)
    target_float = target.astype(np.float32)
    
    # Convert to LAB
    source_lab = cv2.cvtColor(source_float / 255.0, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_float / 255.0, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l_src, a_src, b_src = cv2.split(source_lab)
    l_tgt, a_tgt, b_tgt = cv2.split(target_lab)
    
    # Match mean and std
    l_out = match_channel(l_tgt, l_src)
    a_out = match_channel(a_tgt, a_src)
    b_out = match_channel(b_tgt, b_src)
    
    # Merge
    result_lab = cv2.merge([l_out, a_out, b_out])
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    result_bgr = np.clip(result_bgr * 255.0, 0, 255)
    
    # Blend
    result = target_float * (1 - blend_strength) + result_bgr * blend_strength
    
    return np.clip(result, 0, 255).astype(np.uint8)


def match_channel(target_channel, source_channel):
    """Match mean and std of target channel to source"""
    target_mean = target_channel.mean()
    target_std = target_channel.std()
    
    source_mean = source_channel.mean()
    source_std = source_channel.std()
    
    # Normalize target
    normalized = (target_channel - target_mean) / (target_std + 1e-6)
    
    # Scale to source
    matched = normalized * source_std + source_mean
    
    return matched


def adaptive_color_blend(
    original_face: np.ndarray,
    swapped_face: np.ndarray,
    mask: np.ndarray = None,
    method: str = "histogram"
) -> np.ndarray:
    """Adaptivní color blending s automatickou detekcí nejlepší metody
    
    Args:
        original_face: Original face region
        swapped_face: Swapped face region
        mask: Optional mask
        method: "histogram", "mean_std", or "auto"
    
    Returns:
        Color-blended face
    """
    if method == "auto":
        # Automaticky zvol metodu podle variance
        orig_var = np.var(original_face)
        swap_var = np.var(swapped_face)
        
        # Pokud je velký rozdíl ve variance, použij mean_std
        if abs(orig_var - swap_var) / (swap_var + 1e-6) > 0.3:
            method = "mean_std"
        else:
            method = "histogram"
    
    if method == "histogram":
        return histogram_matching_DFL_Orig(original_face, swapped_face, mask, 80.0)
    elif method == "mean_std":
        return color_transfer_mean_std(original_face, swapped_face, 0.7)
    else:
        return swapped_face


# PyTorch versions pro GPU acceleration
def torch_histogram_matching(
    original: torch.Tensor,  # (C, H, W) float 0-1
    swapped: torch.Tensor,   # (C, H, W) float 0-1
    mask: torch.Tensor = None,
    blend_amount: float = 80.0
) -> torch.Tensor:
    """GPU-accelerated histogram matching"""
    if blend_amount <= 0:
        return swapped
    
    result = swapped.clone()
    
    for c in range(3):
        orig_channel = original[c]
        swap_channel = result[c]
        
        # Compute histograms
        hist_orig = torch.histc(orig_channel, bins=256, min=0, max=1)
        hist_swap = torch.histc(swap_channel, bins=256, min=0, max=1)
        
        # CDF
        cdf_orig = torch.cumsum(hist_orig, dim=0)
        cdf_orig = cdf_orig / cdf_orig[-1]
        
        cdf_swap = torch.cumsum(hist_swap, dim=0)
        cdf_swap = cdf_swap / cdf_swap[-1]
        
        # Interpolate lookup table
        # (Simplified - pro přesné LUT use torch.searchsorted)
        bins = torch.linspace(0, 1, 256, device=swapped.device)
        
        # Apply matching (simplified version)
        alpha = blend_amount / 100.0
        mean_orig = orig_channel.mean()
        mean_swap = swap_channel.mean()
        std_orig = orig_channel.std()
        std_swap = swap_channel.std()
        
        matched = (swap_channel - mean_swap) / (std_swap + 1e-6) * std_orig + mean_orig
        
        result[c] = swap_channel * (1 - alpha) + matched * alpha
    
    if mask is not None:
        mask_3ch = mask.unsqueeze(0).expand(3, -1, -1)
        result = result * mask_3ch + swapped * (1 - mask_3ch)
    
    return torch.clamp(result, 0, 1)
