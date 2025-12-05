"""
Face Warping Module - Dle Rope-next
Warpuje tvář podle 5-ti bodů landmarks pro lepší swap alignment
"""

import cv2
import numpy as np


def get_affine_matrix_from_landmarks(landmarks, target_size=512, scale=1.5, vy_ratio=-0.125):
    """Vytvoří affine transformační matici z landmarks
    
    Args:
        landmarks: 5 bodů (shape: (5, 2)) - levé oko, pravé oko, nos, levý roh úst, pravý roh úst
        target_size: Cílová velikost (512, 224, atd.)
        scale: Faktor zoomování (1.5-2.5)
        vy_ratio: Vertikální offset poměr (-0.125 je dobrý)
    
    Returns:
        M_o2c: Matice z originálu do cropu
        M_c2o: Matice z cropu zpět do originálu
    """
    if landmarks.shape[0] < 5:
        return None, None
    
    # Standardní pozice 5 bodů v 512x512 frameu
    # Levé oko, pravé oko, nos, levý roh úst, pravý roh úst
    dst_pts = np.array([
        [192, 240],  # levé oko
        [320, 240],  # pravé oko
        [256, 340],  # nos
        [192, 430],  # levý roh úst
        [320, 430],  # pravý roh úst
    ], dtype=np.float32)
    
    src_pts = landmarks.astype(np.float32)
    
    # Aplikuj scale a vy_ratio
    dst_pts = dst_pts * (target_size / 512.0)
    
    # Vypočítej podobnostní transformaci
    M_o2c, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    if M_o2c is None:
        return None, None
    
    # Inverzní matice
    M_c2o = cv2.invertAffineTransform(M_o2c)
    
    return M_o2c, M_c2o


def warp_face(frame, landmarks, target_size=512, scale=1.5, vy_ratio=-0.125):
    """Warpuje tvář podle landmarks
    
    Args:
        frame: Originální frame
        landmarks: 5 bodů landmarks
        target_size: Cílová velikost
        scale: Zoom faktor
        vy_ratio: Vertikální offset
    
    Returns:
        warped_face: Warpovaná tvář
        M_o2c: Forward transformační matice
        M_c2o: Inverse transformační matice
    """
    M_o2c, M_c2o = get_affine_matrix_from_landmarks(landmarks, target_size, scale, vy_ratio)
    
    if M_o2c is None:
        return None, None, None
    
    # Warp frame
    warped = cv2.warpAffine(
        frame, 
        M_o2c, 
        (target_size, target_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return warped, M_o2c, M_c2o


def unwarp_face(warped_face, M_c2o, original_shape):
    """Unwarpuje tvář zpět do originálu
    
    Args:
        warped_face: Warpovaná tvář
        M_c2o: Inverse transformační matice
        original_shape: Původní tvar (h, w, c)
    
    Returns:
        unwarped_face: Tvář ve původní poloze
    """
    h, w = original_shape[0], original_shape[1]
    
    unwarped = cv2.warpAffine(
        warped_face,
        M_c2o,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return unwarped


def create_soft_mask(target_size=512, border_width=50):
    """Vytvoří měkkou masku pro blend okrajů
    
    Args:
        target_size: Velikost masky
        border_width: Šířka měkkého okraje
    
    Returns:
        mask: Soft maska (0-1)
    """
    mask = np.ones((target_size, target_size), dtype=np.float32)
    
    # Vytvoř gradient na okrajích
    for i in range(border_width):
        alpha = i / border_width
        mask[i, :] *= alpha
        mask[-(i+1), :] *= alpha
        mask[:, i] *= alpha
        mask[:, -(i+1)] *= alpha
    
    # Gaussian blur pro měkčí přechod
    mask = cv2.GaussianBlur(mask, (2*border_width+1, 2*border_width+1), border_width/3.0)
    
    return np.clip(mask, 0, 1)
