"""
Advanced Face Blending Module
Řeší problém rámečků kolem obličeje - implementuje pokročilé blending strategie
"""

import cv2
import numpy as np
from scipy import ndimage


class AdvancedFaceBlender:
    """Pokročilé blending techniky pro eliminaci rámečků"""
    
    @staticmethod
    def create_feather_mask(shape, expand_pixels=50, feather_pixels=30):
        """Vytvoří měkkou masku s graduelním přechodem
        
        Args:
            shape: Tvar masky (height, width)
            expand_pixels: Počet pixelů pro rozšíření
            feather_pixels: Počet pixelů pro měkké přechody
        
        Returns:
            Maska s hodnotami 0-1
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Vnitřní obdélník
        y1 = expand_pixels
        y2 = h - expand_pixels
        x1 = expand_pixels
        x2 = w - expand_pixels
        
        mask[y1:y2, x1:x2] = 1.0
        
        # Gaussian feathering pro měkké okraje
        mask = cv2.GaussianBlur(mask, (2*feather_pixels+1, 2*feather_pixels+1), feather_pixels)
        
        return mask
    
    @staticmethod
    def create_skin_mask(face_img, dilate_kernel=11):
        """Vytvoří masku kůže pomocí HSV segmentace
        
        Args:
            face_img: BGR obrázek tváře
            dilate_kernel: Velikost kernel pro morfologické operace
        
        Returns:
            Float maska [0, 1]
        """
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        
        # HSV rozsahy pro lidskou kůži (empiricky testováno)
        # Hue: 0-20, Saturation: 10-40, Value: 50-255
        lower_skin = np.array([0, 10, 50], dtype=np.uint8)
        upper_skin = np.array([20, 40, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morfologické operace
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Gaussian blur pro měkké hrany
        mask = cv2.GaussianBlur(mask, (21, 21), 7)
        
        return mask.astype(np.float32) / 255.0
    
    @staticmethod
    def blend_faces_advanced(frame, swapped_face, bbox, expand_ratio=1.35, use_color_match=False):
        """Pokročilý blend swappované tváře s rámem - inspirován Rope-next
        
        Args:
            frame: Původní frame
            swapped_face: Swappovaná tvář (paste_back=False output - ořezaná tvář)
            bbox: Bounding box (x1, y1, x2, y2)
            expand_ratio: Kolik pixelů rozšířit bbox pro měkký blend
            use_color_match: Použít color matching (pomalejší, ale lepší kvalita)
        
        Returns:
            Blended frame
        """
        # Ověřit že frame a swapped_face jsou numpy arrays
        if not isinstance(frame, np.ndarray):
            return frame
        if not isinstance(swapped_face, np.ndarray):
            return frame
        
        # Zkontroluj validní velikost
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            return frame
        if swapped_face.shape[0] <= 0 or swapped_face.shape[1] <= 0:
            return frame
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = y2 - y1, x2 - x1
        
        # Zkontroluj bbox validitu
        if h <= 0 or w <= 0:
            return frame
        
        # Ořízni frame region
        frame_region = frame[y1:y2, x1:x2].copy()
        
        # Resize swapped_face aby seděl
        try:
            swapped_resized = cv2.resize(swapped_face, (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"[DEBUG] Resize error: {e}")
            return frame
        
        # Vytvořit měkkou masku s Gaussianem - inspirován Rope-next
        # Jednoduchá čtvercová maska
        mask = np.ones((h, w), dtype=np.float32)
        
        # Aplikuj Gaussianův blur pro měkké okraje
        # Parametry dle Rope-next: kernel = blur*2+1, sigma = (blur+1)*0.2, blur=15
        kernel_size = 15 * 2 + 1  # 31
        sigma = (15 + 1) * 0.2      # 3.2
        
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
        mask = np.clip(mask, 0, 1)
        
        # Vrstvu swapped a frame_region dohromady s maskou
        # result = frame_region * (1 - mask) + swapped_resized * mask
        result_region = (frame_region.astype(np.float32) * (1 - mask[:, :, np.newaxis]) +
                        swapped_resized.astype(np.float32) * mask[:, :, np.newaxis])
        
        # Umísti výsledek zpět do frameu
        frame_out = frame.copy()
        frame_out[y1:y2, x1:x2] = np.clip(result_region, 0, 255).astype(np.uint8)
        
        return frame_out
    
    @staticmethod
    def color_match_faces(swapped_face, original_face_region, blend_strength=0.7):
        """Slaďuje barvu swappované tváře s původní
        
        Args:
            swapped_face: Swappovaná tvář
            original_face_region: Původní region tváře
            blend_strength: Síla color matchingu (0-1)
        
        Returns:
            Swappovaná tvář s sladěnou barvou
        """
        # Vypočítat průměrné barvy
        original_mean = original_face_region.mean(axis=(0, 1))
        swapped_mean = swapped_face.mean(axis=(0, 1))
        
        # Color shift
        color_shift = (original_mean - swapped_mean) * blend_strength
        
        # Aplikovat shift
        result = swapped_face.astype(np.float32) + color_shift
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def histogram_match(swapped_face, reference_face):
        """Matchuje histogram swappované tváře s referenční
        
        Args:
            swapped_face: Swappovaná tvář
            reference_face: Referenční tvář pro matching
        
        Returns:
            Swappovaná tvář s matchovaným histogramem
        """
        result = swapped_face.copy()
        
        # Histogram matching pro každý kanál
        for i in range(3):  # BGR
            # Vypočítat CDF
            hist_ref = cv2.calcHist([reference_face], [i], None, [256], [0, 256])
            hist_ref = hist_ref.flatten() / hist_ref.sum()
            cdf_ref = hist_ref.cumsum()
            
            hist_swap = cv2.calcHist([swapped_face], [i], None, [256], [0, 256])
            hist_swap = hist_swap.flatten() / hist_swap.sum()
            cdf_swap = hist_swap.cumsum()
            
            # Vytvořit lookup tabulku
            lut = np.zeros(256, dtype=np.uint8)
            for j in range(256):
                # Najít nejbližší hodnotu v referenční CDF
                diff = np.abs(cdf_ref - cdf_swap[j])
                lut[j] = np.argmin(diff)
            
            # Aplikovat LUT
            result[:, :, i] = cv2.LUT(result[:, :, i], lut)
        
        return result
    
    @staticmethod
    def seamless_clone_advanced(frame, swapped_face, bbox, method=cv2.MIXED_CLONE):
        """Seamless cloning s pokročilými parametry
        
        Args:
            frame: Původní frame
            swapped_face: Swappovaná tvář
            bbox: Bounding box
            method: Metoda cloningu (NORMAL_CLONE, MIXED_CLONE, MONOCHROME_TRANSFER)
        
        Returns:
            Frame se seamless cloned obličejem
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Vytvořit masku
        mask = 255 * np.ones((y2-y1, x2-x1, 3), dtype=np.uint8)
        
        # Seamless clone
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        swapped_resized = cv2.resize(swapped_face, (x2-x1, y2-y1))
        
        result = cv2.seamlessClone(swapped_resized, frame, mask[:,:,0], center, method)
        
        return result
    
    @staticmethod
    def blur_face_edges(frame, bbox, blur_strength=15):
        """Post-processing: blur hrany swapped tváře pro eliminaci tvrdých okrajů
        
        Args:
            frame: Frame s aplikovaným swapem (paste_back=True)
            bbox: Bounding box tváře (x1, y1, x2, y2)
            blur_strength: Síla bluuru (10-20 dobrý rozsah)
        
        Returns:
            Frame s rozmazanými okraji
        """
        if not isinstance(frame, np.ndarray):
            return frame
        
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = y2 - y1, x2 - x1
            
            if h <= 0 or w <= 0 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                return frame
            
            # Vytvoř masku s Gaussianem - maska pro okraje
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0
            
            # Aplikuj Gaussian blur - parametry dle Rope-next
            kernel_size = blur_strength * 2 + 1
            sigma = (blur_strength + 1) * 0.2
            
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Rozmazej masku na okrajích
            mask_soft = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
            mask_soft = np.clip(mask_soft, 0, 1)
            
            # Vytvoř blurred verzi framu (rozmazaný frame)
            frame_blurred = cv2.GaussianBlur(frame, (5, 5), 1.5)
            
            # Blend - u okrajů použij rozmazanou verzi, uprostřed swap
            mask_3ch = np.stack([mask_soft] * 3, axis=-1)
            result = frame.astype(np.float32) * mask_3ch + frame_blurred.astype(np.float32) * (1 - mask_3ch)
            
            return np.clip(result, 0, 255).astype(np.uint8)
        
        except Exception as e:
            print(f"[DEBUG] Blur edges error: {e}")
            return frame

class TileBlender:
    """Blending pro tile-based processing (8K video)"""
    
    @staticmethod
    def merge_tiles_with_blend(tiles, positions, original_shape, overlap=50):
        """Sloučí tiles s overlap blending
        
        Args:
            tiles: List zpracovaných tilů
            positions: List (y, x, y_end, x_end) pro každý tile
            original_shape: Tvar původního framu (H, W, C)
            overlap: Počet pixelů pro overlap blending
        
        Returns:
            Sloučený frame
        """
        result = np.zeros(original_shape, dtype=np.float32)
        weight_sum = np.zeros(original_shape[:2], dtype=np.float32)
        
        for tile, (y, x, y_end, x_end) in zip(tiles, positions):
            h = y_end - y
            w = x_end - x
            
            # Vytvořit feathering masku
            feather_mask = np.ones((h, w), dtype=np.float32)
            
            # Měkké hrany - lineární gradient
            feather_width = min(overlap, h // 4, w // 4)
            if feather_width > 0:
                # Create 2D distance map od okrajů
                dist_y = np.minimum(np.arange(h), h - np.arange(h))
                dist_x = np.minimum(np.arange(w), w - np.arange(w))
                
                dist_field = np.minimum(
                    np.minimum(dist_y[:, np.newaxis], dist_x[np.newaxis, :]),
                    feather_width
                )
                
                feather_mask = dist_field / (feather_width + 1e-6)
                feather_mask = np.clip(feather_mask, 0, 1)
            
            # Aplikovat masku
            tile_masked = tile.astype(np.float32) * feather_mask[:,:,np.newaxis]
            feather_mask_3d = np.stack([feather_mask] * 3, axis=-1)
            
            result[y:y_end, x:x_end] += tile_masked
            weight_sum[y:y_end, x:x_end] += feather_mask_3d[:,:,0]
        
        # Normalizovat - zabránit dělení nulou
        weight_sum = np.maximum(weight_sum, 1e-6)
        for c in range(result.shape[2]):
            result[:, :, c] /= weight_sum
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def split_tiles_with_overlap(frame, tile_size=512, overlap=50):
        """Rozdělí frame na tiles s overlapem
        
        Args:
            frame: Input frame
            tile_size: Velikost jednoho tile
            overlap: Overlap mezi tiles
        
        Returns:
            (tiles, positions) - tiles a jejich pozice
        """
        h, w = frame.shape[:2]
        tiles = []
        positions = []
        
        step = tile_size - overlap
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                # Zajistit přesně tile_size velikost (padding pokud potřeba)
                tile_h = y_end - y
                tile_w = x_end - x
                
                if tile_h < tile_size or tile_w < tile_size:
                    # Pad tile
                    tile = np.pad(
                        frame[y:y_end, x:x_end],
                        ((0, tile_size - tile_h), (0, tile_size - tile_w), (0, 0)),
                        mode='edge'
                    )
                else:
                    tile = frame[y:y_end, x:x_end]
                
                tiles.append(tile)
                positions.append((y, x, y_end, x_end))
        
        return tiles, positions
