"""
VR180 Stitching Module - Inspirováno VisoMaster-Experimental
Řeší problém s viditelným švem v equirectangular videu po perspective crop processing
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class VRStitcher:
    """VR180 perspective to equirectangular stitching s feathering"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.sobel_x_kernel = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], 
            dtype=torch.float32
        ).reshape(1, 1, 3, 3).to(device)
        self.sobel_y_kernel = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], 
            dtype=torch.float32
        ).reshape(1, 1, 3, 3).to(device)
    
    def apply_feathering(
        self,
        mask_torch: torch.Tensor,
        feather_radius: int = 15,
        blur_sigma_factor: float = 0.5,
        erosion_kernel_size: int = 5,
    ) -> torch.Tensor:
        """Aplikuje erosion + Gaussian feathering na masku - dle VisoMaster
        
        Args:
            mask_torch: Boolean mask tensor (1, H, W) nebo (H, W)
            feather_radius: Radius featheringu
            blur_sigma_factor: Faktor pro sigma Gaussian blur
            erosion_kernel_size: Velikost erosion kernelu
        
        Returns:
            Float mask s feathering (0-1)
        """
        if mask_torch.dim() == 2:
            mask_torch = mask_torch.unsqueeze(0)
        
        # Convert boolean to float
        mask_float = mask_torch.float()
        
        # 1. Erosion - zmenší oblast
        erosion_kernel = torch.ones(
            1, 1, erosion_kernel_size, erosion_kernel_size,
            device=self.device
        )
        mask_eroded = F.conv2d(
            mask_float.unsqueeze(0),
            erosion_kernel,
            padding=erosion_kernel_size // 2
        )
        # Threshold - zachovat pouze kde všechny pixely byly 1
        threshold = erosion_kernel_size * erosion_kernel_size
        mask_eroded = (mask_eroded >= threshold).float()
        
        # 2. Gaussian blur pro feathering
        sigma = feather_radius * blur_sigma_factor
        kernel_size = 2 * feather_radius + 1
        
        # Vytvořit Gaussian kernel
        x = torch.arange(-feather_radius, feather_radius + 1, device=self.device, dtype=torch.float32)
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # 2D kernel
        gaussian_kernel = gaussian_1d.view(1, -1) * gaussian_1d.view(-1, 1)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        
        # Aplikovat blur
        mask_feathered = F.conv2d(
            mask_eroded,
            gaussian_kernel,
            padding=kernel_size // 2
        )
        
        return torch.clamp(mask_feathered.squeeze(0), 0, 1)
    
    def stitch_perspective_to_equirect(
        self,
        target_equirect: torch.Tensor,  # (C, H, W) RGB uint8
        processed_crop: torch.Tensor,   # (C, H, W) RGB uint8
        mask: torch.Tensor,             # (H, W) boolean - kde paste processed_crop
        is_left_eye: bool = True,
        feather_radius: int = 12
    ) -> torch.Tensor:
        """Stitchuje processed perspective crop zpět do equirect s feathering
        
        Args:
            target_equirect: Target equirectangular image
            processed_crop: Processed perspective crop (swapped)
            mask: Maska kde aplikovat crop
            is_left_eye: Zda je to levé oko (pro eye-specific masking)
            feather_radius: Radius featheringu
        
        Returns:
            Updated equirectangular image
        """
        C, H, W = target_equirect.shape
        
        # Eye-specific masking - pouze pro příslušnou polovinu
        eye_region_mask = torch.zeros_like(mask, dtype=torch.bool)
        half_width = W // 2
        if is_left_eye:
            eye_region_mask[:, :half_width] = True
        else:
            eye_region_mask[:, half_width:] = True
        
        # Kombinuj masky
        eye_specific_mask = mask & eye_region_mask
        
        # Aplikuj feathering
        feathered_mask = self.apply_feathering(
            eye_specific_mask,
            feather_radius=feather_radius,
            blur_sigma_factor=0.5,
            erosion_kernel_size=(2 * feather_radius + 1)
        )
        
        # Memory-efficient blending (in-place operations)
        # result = target + (processed - target) * mask
        
        target_float = target_equirect.float()
        processed_float = processed_crop.float()
        
        # Expand mask pro 3 kanály
        feathered_mask_3ch = feathered_mask.unsqueeze(0).expand(C, -1, -1)
        
        # Blend
        # difference = processed - target
        difference = processed_float - target_float
        
        # Apply mask
        difference *= feathered_mask_3ch
        
        # Add to target
        result = target_float + difference
        
        # Clamp a convert zpět
        result = torch.clamp(result, 0, 255).byte()
        
        return result
    
    def compute_edge_weights(self, image: torch.Tensor) -> torch.Tensor:
        """Vypočítá edge weights pomocí Sobel operátoru - pro gradient-based blending
        
        Args:
            image: RGB image tensor (C, H, W)
        
        Returns:
            Edge magnitude map (H, W)
        """
        # Convert to grayscale
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        gray = gray.unsqueeze(0).unsqueeze(0).float()
        
        # Sobel gradients
        grad_x = F.conv2d(gray, self.sobel_x_kernel, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y_kernel, padding=1)
        
        # Magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        return magnitude.squeeze()


def numpy_stitch_with_feathering(
    target_eq: np.ndarray,      # (H, W, C) BGR uint8
    processed_crop: np.ndarray, # (H_crop, W_crop, C) BGR uint8
    mask: np.ndarray,           # (H, W) boolean
    is_left_eye: bool = True,
    feather_radius: int = 12
) -> np.ndarray:
    """NumPy verze stitching s feathering - pro CPU fallback
    
    Args:
        target_eq: Target equirectangular image
        processed_crop: Processed crop (musí být už transformován do eq coords)
        mask: Boolean maska kde paste
        is_left_eye: Eye selection
        feather_radius: Feathering radius
    
    Returns:
        Stitched equirectangular image
    """
    H, W, C = target_eq.shape
    
    # Eye-specific mask
    eye_region_mask = np.zeros((H, W), dtype=bool)
    half_width = W // 2
    if is_left_eye:
        eye_region_mask[:, :half_width] = True
    else:
        eye_region_mask[:, half_width:] = True
    
    # Combine masks
    eye_specific_mask = mask & eye_region_mask
    
    # Feathering
    # 1. Erosion
    erosion_kernel_size = 2 * feather_radius + 1
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (erosion_kernel_size, erosion_kernel_size)
    )
    mask_eroded = cv2.erode(
        eye_specific_mask.astype(np.uint8),
        erosion_kernel,
        iterations=1
    )
    
    # 2. Gaussian blur
    sigma = feather_radius * 0.5
    kernel_size = 2 * feather_radius + 1
    mask_feathered = cv2.GaussianBlur(
        mask_eroded.astype(np.float32),
        (kernel_size, kernel_size),
        sigma
    )
    mask_feathered = np.clip(mask_feathered, 0, 1)
    
    # 3. Blend
    mask_3ch = np.stack([mask_feathered] * C, axis=-1)
    
    result = target_eq.astype(np.float32) * (1 - mask_3ch)
    result += processed_crop.astype(np.float32) * mask_3ch
    
    return np.clip(result, 0, 255).astype(np.uint8)
