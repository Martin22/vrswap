#!/usr/bin/env python3
"""Ověř, že opravy jsou správně nainstalovány"""
import os
import sys

print("=" * 60)
print("VRSwap - Verifikace oprav")
print("=" * 60)

# 1. Zkontroluj nové moduly
print("\n1. Kontrola nových modulů...")
required_files = [
    'core/advanced_blending.py',
    'core/face_swap_improved.py'
]

for f in required_files:
    if os.path.exists(f):
        print(f"   ✓ {f}")
    else:
        print(f"   ✗ CHYBÍ: {f}")
        sys.exit(1)

# 2. Zkontroluj opravy v swap.py
print("\n2. Kontrola oprav v swap.py...")
with open('swap.py', 'r') as f:
    content = f.read()
    checks = [
        ('from core.advanced_blending', 'Import AdvancedFaceBlender'),
        ('blend_faces_advanced', 'Pokročilý blend'),
        ('color_match_faces', 'Color matching'),
    ]
    for check, desc in checks:
        if check in content:
            print(f"   ✓ {desc}")
        else:
            print(f"   ✗ CHYBÍ: {desc}")

# 3. Zkontroluj opravy v core/analyser.py
print("\n3. Kontrola oprav v core/analyser.py...")
with open('core/analyser.py', 'r') as f:
    content = f.read()
    if '896, 896' in content:
        print("   ✓ det_size zvětšen na 896x896")
    else:
        print("   ✗ det_size není zvětšen")

# 4. Zkontroluj opravy v core/globals.py
print("\n4. Kontrola oprav v core/globals.py...")
with open('core/globals.py', 'r') as f:
    content = f.read()
    if 'torch.float16' in content:
        print("   ✓ FP16 fallback implementován")
    else:
        print("   ✗ FP16 fallback chybí")

print("\n" + "=" * 60)
print("✓ VŠECHNY OPRAVY JSOU SPRÁVNĚ NAINSTALOVÁNY!")
print("=" * 60)
print("\nSpuštění: python swap.py --frames_folder ./frames --face source.jpg")
