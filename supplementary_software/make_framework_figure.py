"""Export publication assets from the approved PowerPoint-exported vector PDF."""
from pathlib import Path
import argparse
import pymupdf
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--dpi', type=int, default=1200)
parser.add_argument('--width-mm', type=float, default=180)
args = parser.parse_args()
fig = args.package_root / 'figures' / 'figure1_framework'
if args.dpi <= 0 or args.width_mm <= 0:
    parser.error('dpi and width-mm must be positive')
with pymupdf.open(fig.with_suffix('.pdf')) as doc:
    if len(doc) != 1:
        raise ValueError('The framework PDF must contain exactly one slide.')
    page = doc[0]
    target_width = round(args.width_mm / 25.4 * args.dpi)
    scale = target_width / page.rect.width
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    image.save(fig.with_suffix('.png'), dpi=(args.dpi,args.dpi), optimize=True)
    image.save(fig.with_suffix('.tiff'), dpi=(args.dpi,args.dpi), compression='tiff_lzw')
    fig.with_suffix('.svg').write_text(page.get_svg_image(text_as_path=True), encoding='utf-8')
    print(f'Exported {pix.width} x {pix.height} PNG/TIFF and vector SVG. PPTX/PDF preserved.')
