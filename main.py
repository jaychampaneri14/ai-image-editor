"""
AI Image Editor
PIL-based image editing with filters, crop, enhance, and AI-style effects.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageOps
import os
import warnings
warnings.filterwarnings('ignore')


class ImageEditor:
    """Comprehensive PIL-based image editor."""

    def __init__(self, image: Image.Image):
        self.original = image.copy()
        self.current  = image.copy()
        self.history  = [image.copy()]

    def reset(self):
        self.current = self.original.copy()
        self.history = [self.original.copy()]
        return self

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.current = self.history[-1].copy()
        return self

    def _save(self):
        self.history.append(self.current.copy())

    # ─── CROPS & TRANSFORMS ──────────────────────────────────────────────────
    def crop(self, left, top, right, bottom):
        self.current = self.current.crop((left, top, right, bottom))
        self._save(); return self

    def resize(self, width, height, resample=Image.LANCZOS):
        self.current = self.current.resize((width, height), resample)
        self._save(); return self

    def rotate(self, angle, expand=True):
        self.current = self.current.rotate(angle, expand=expand)
        self._save(); return self

    def flip_horizontal(self):
        self.current = self.current.transpose(Image.FLIP_LEFT_RIGHT)
        self._save(); return self

    def flip_vertical(self):
        self.current = self.current.transpose(Image.FLIP_TOP_BOTTOM)
        self._save(); return self

    def center_crop(self, size):
        w, h = self.current.size
        left = (w - size) // 2; top = (h - size) // 2
        return self.crop(left, top, left + size, top + size)

    # ─── ADJUSTMENTS ─────────────────────────────────────────────────────────
    def brightness(self, factor=1.0):
        self.current = ImageEnhance.Brightness(self.current).enhance(factor)
        self._save(); return self

    def contrast(self, factor=1.0):
        self.current = ImageEnhance.Contrast(self.current).enhance(factor)
        self._save(); return self

    def saturation(self, factor=1.0):
        self.current = ImageEnhance.Color(self.current).enhance(factor)
        self._save(); return self

    def sharpness(self, factor=1.0):
        self.current = ImageEnhance.Sharpness(self.current).enhance(factor)
        self._save(); return self

    def auto_enhance(self):
        """Auto-balance brightness and contrast."""
        self.current = ImageOps.autocontrast(self.current)
        self._save(); return self

    def equalize(self):
        self.current = ImageOps.equalize(self.current.convert('RGB'))
        self._save(); return self

    def gamma_correction(self, gamma=1.0):
        arr = np.array(self.current, dtype=np.float32) / 255.0
        arr = np.power(arr, 1.0 / gamma)
        self.current = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        self._save(); return self

    # ─── FILTERS ─────────────────────────────────────────────────────────────
    def blur(self, radius=2):
        self.current = self.current.filter(ImageFilter.GaussianBlur(radius))
        self._save(); return self

    def sharpen(self):
        self.current = self.current.filter(ImageFilter.SHARPEN)
        self._save(); return self

    def edge_enhance(self):
        self.current = self.current.filter(ImageFilter.EDGE_ENHANCE_MORE)
        self._save(); return self

    def emboss(self):
        self.current = self.current.filter(ImageFilter.EMBOSS)
        self._save(); return self

    def find_edges(self):
        self.current = self.current.filter(ImageFilter.FIND_EDGES)
        self._save(); return self

    def unsharp_mask(self, radius=2, percent=150, threshold=3):
        self.current = self.current.filter(ImageFilter.UnsharpMask(radius, percent, threshold))
        self._save(); return self

    # ─── ARTISTIC EFFECTS ────────────────────────────────────────────────────
    def grayscale(self):
        self.current = self.current.convert('L').convert('RGB')
        self._save(); return self

    def sepia(self):
        img = self.current.convert('RGB')
        arr = np.array(img, dtype=np.float32)
        r = arr[:,:,0]*0.393 + arr[:,:,1]*0.769 + arr[:,:,2]*0.189
        g = arr[:,:,0]*0.349 + arr[:,:,1]*0.686 + arr[:,:,2]*0.168
        b = arr[:,:,0]*0.272 + arr[:,:,1]*0.534 + arr[:,:,2]*0.131
        sepia = np.stack([r.clip(0,255), g.clip(0,255), b.clip(0,255)], axis=2).astype(np.uint8)
        self.current = Image.fromarray(sepia)
        self._save(); return self

    def vignette(self, strength=0.5):
        """Dark vignette effect around edges."""
        img   = self.current.convert('RGB')
        w, h  = img.size
        arr   = np.array(img, dtype=np.float32)
        Y, X  = np.ogrid[:h, :w]
        cx, cy = w/2, h/2
        dist   = np.sqrt(((X-cx)/cx)**2 + ((Y-cy)/cy)**2)
        mask   = 1 - strength * dist**1.5
        mask   = mask.clip(0, 1)
        arr   *= mask[:,:,np.newaxis]
        self.current = Image.fromarray(arr.clip(0,255).astype(np.uint8))
        self._save(); return self

    def color_filter(self, r_mult=1.0, g_mult=1.0, b_mult=1.0):
        """Apply per-channel color filter."""
        img = self.current.convert('RGB')
        arr = np.array(img, dtype=np.float32)
        arr[:,:,0] *= r_mult
        arr[:,:,1] *= g_mult
        arr[:,:,2] *= b_mult
        self.current = Image.fromarray(arr.clip(0,255).astype(np.uint8))
        self._save(); return self

    def cartoon_effect(self):
        """Cartoonize via edge detection overlay."""
        img     = self.current.convert('RGB')
        # Bilateral-like smoothing
        smooth  = img.filter(ImageFilter.MedianFilter(size=3))
        smooth  = smooth.filter(ImageFilter.SMOOTH_MORE)
        # Edges
        edges   = img.convert('L').filter(ImageFilter.FIND_EDGES)
        edges   = ImageOps.invert(edges).convert('RGB')
        # Multiply
        arr_s = np.array(smooth, dtype=np.float32) / 255.0
        arr_e = np.array(edges,  dtype=np.float32) / 255.0
        cartoon = (arr_s * arr_e * 255).clip(0,255).astype(np.uint8)
        # Boost saturation
        self.current = Image.fromarray(cartoon)
        self.saturation(2.0)
        return self

    def pixelate(self, pixel_size=10):
        """Pixelate effect."""
        w, h = self.current.size
        small = self.current.resize((w // pixel_size, h // pixel_size), Image.NEAREST)
        self.current = small.resize((w, h), Image.NEAREST)
        self._save(); return self

    def add_text_watermark(self, text='SAMPLE', opacity=80):
        """Add text watermark."""
        img  = self.current.convert('RGBA')
        w, h = img.size
        watermark = Image.new('RGBA', (w, h), (0,0,0,0))
        draw = ImageDraw.Draw(watermark)
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
        except:
            font = None
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        pos = ((w-tw)//2, (h-th)//2)
        draw.text(pos, text, fill=(255,255,255,opacity), font=font)
        merged = Image.alpha_composite(img, watermark)
        self.current = merged.convert('RGB')
        self._save(); return self

    def get_image(self):
        return self.current

    def save(self, path):
        self.current.save(path)
        print(f"Saved to {path}")


def create_test_image(width=400, height=300):
    """Create a colorful test image."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Gradient background
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = int(255 * x / width)
            arr[y, x, 1] = int(255 * y / height)
            arr[y, x, 2] = int(255 * (1 - x/width) * (1 - y/height))
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    # Add shapes
    draw.ellipse([50, 50, 150, 150], fill=(255, 255, 0, 200), outline=(255, 100, 0), width=3)
    draw.rectangle([200, 80, 320, 180], fill=(100, 200, 255), outline=(0, 0, 200), width=3)
    draw.polygon([(180, 250), (230, 180), (280, 250)], fill=(200, 100, 200), outline=(100, 0, 100), width=2)
    return img


def demo_all_effects(img_path=None):
    """Demo all image effects and save comparison grid."""
    if img_path and os.path.exists(img_path):
        base = Image.open(img_path).resize((400, 300))
    else:
        base = create_test_image()

    effects = {
        'Original':     lambda e: e,
        'Grayscale':    lambda e: e.grayscale(),
        'Sepia':        lambda e: e.sepia(),
        'Blur':         lambda e: e.blur(3),
        'Sharpen':      lambda e: e.sharpen(),
        'Edge Enhance': lambda e: e.edge_enhance(),
        'Emboss':       lambda e: e.emboss(),
        'Find Edges':   lambda e: e.find_edges(),
        'High Contrast':lambda e: e.contrast(2.5),
        'Low Bright':   lambda e: e.brightness(0.4),
        'Saturation 3x':lambda e: e.saturation(3.0),
        'Vignette':     lambda e: e.vignette(0.7),
        'Cartoon':      lambda e: e.cartoon_effect(),
        'Pixelate':     lambda e: e.pixelate(15),
        'Color Filter': lambda e: e.color_filter(r_mult=1.5, g_mult=0.8, b_mult=0.7),
        'Watermark':    lambda e: e.add_text_watermark('DEMO', opacity=100),
    }

    n_cols = 4
    n_rows = (len(effects) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    for i, (name, fn) in enumerate(effects.items()):
        editor = ImageEditor(base)
        fn(editor)
        result = editor.get_image()
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(result)
        ax.set_title(name, fontsize=10)
        ax.axis('off')
    # Hide unused axes
    for j in range(len(effects), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')
    plt.suptitle('AI Image Editor — Filter Gallery', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('effects_gallery.png', dpi=150)
    plt.close()
    print("Effects gallery saved to effects_gallery.png")


def demo_editing_pipeline(base_img):
    """Apply a complete editing pipeline."""
    print("\n--- Editing Pipeline Demo ---")
    editor = ImageEditor(base_img)
    result = (editor
              .auto_enhance()
              .brightness(1.1)
              .contrast(1.2)
              .saturation(1.3)
              .sharpen()
              .vignette(0.4)
              .add_text_watermark('EDITED')
              .get_image())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(base_img);  ax1.set_title('Original'); ax1.axis('off')
    ax2.imshow(result);    ax2.set_title('After Pipeline'); ax2.axis('off')
    plt.tight_layout()
    plt.savefig('pipeline_demo.png', dpi=150)
    plt.close()
    print("Pipeline demo saved to pipeline_demo.png")
    editor.save('edited_output.png')
    return result


def main():
    print("=" * 60)
    print("AI IMAGE EDITOR")
    print("=" * 60)

    base = create_test_image(400, 300)
    print(f"Test image: {base.size} pixels, mode: {base.mode}")

    demo_all_effects()
    demo_editing_pipeline(base)

    # Available operations summary
    print("\n--- Available Operations ---")
    ops = [
        "Crop, Resize, Rotate, Flip",
        "Brightness, Contrast, Saturation, Sharpness",
        "Auto-enhance, Equalize, Gamma correction",
        "Blur, Sharpen, Edge enhance, Emboss",
        "Grayscale, Sepia, Vignette, Color filter",
        "Cartoon effect, Pixelate, Watermark"
    ]
    for op in ops:
        print(f"  {op}")

    print("\n✓ AI Image Editor complete!")


if __name__ == '__main__':
    main()
