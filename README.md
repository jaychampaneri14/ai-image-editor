# AI Image Editor

Comprehensive PIL-based image editing toolkit with 20+ filters, adjustments, and artistic effects.

## Features
### Transforms
- Crop, resize, rotate, flip

### Adjustments
- Brightness, contrast, saturation, sharpness
- Auto-enhance, histogram equalization, gamma correction

### Filters
- Gaussian blur, sharpen, edge enhance, emboss, find edges
- Unsharp mask

### Artistic Effects
- Grayscale, sepia tone, vignette
- Color channel filter (warm/cool tones)
- Cartoon effect, pixelate
- Text watermark

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Usage
```python
from main import ImageEditor
from PIL import Image

editor = ImageEditor(Image.open('photo.jpg'))
result = (editor
    .auto_enhance()
    .contrast(1.3)
    .vignette(0.5)
    .sepia()
    .get_image())
result.save('output.jpg')
```

## Output
- `effects_gallery.png` — 16-panel filter comparison
- `pipeline_demo.png` — before/after editing pipeline
- `edited_output.png` — pipeline result
