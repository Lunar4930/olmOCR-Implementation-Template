"""Command-line image preprocessing utilities for olmOCR inputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import fitz
import img2pdf
import numpy as np
import pytesseract
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def image_paths(directory: Path) -> list[Path]:
    """Return supported images in a directory in filename order."""
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

# ---------------------------------------------------------
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ---------------------------------------------------------
def apply_clahe(
    image_path: Path,
    output_path: Path,
    apply_blur: bool = True,
    blur_kernel_size: tuple[int, int] = (3, 3),
    blur_sigma: float = 0,
    apply_clahe_step: bool = True,
    clahe_clip_limit: float = 8.0,
    clahe_tile_grid_size: tuple[int, int] = (32, 32),
) -> None:
    """Convert an image to grayscale and optionally apply blur and CLAHE."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Read the image in grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Optional blur
    if apply_blur:
        if any(size <= 0 or size % 2 == 0 for size in blur_kernel_size):
            raise ValueError("Blur kernel dimensions must be positive odd integers.")
        image = cv2.GaussianBlur(image, blur_kernel_size, blur_sigma)

    # Optional CLAHE
    if apply_clahe_step:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size,
        )
        image = clahe.apply(image)

    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write image: {output_path}")

def process_clahe(args: argparse.Namespace) -> None:
    input_directory = args.input_directory
    output_directory = args.output_directory or input_directory.with_name(
        f"{input_directory.name}_clahe"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = image_paths(input_directory)

    for path in paths:
        output_path = output_directory / f"{path.stem}_clahe{path.suffix.lower()}"
        apply_clahe(
            path,
            output_path,
            apply_blur=not args.no_blur,
            blur_kernel_size=tuple(args.blur_kernel),
            blur_sigma=args.blur_sigma,
            apply_clahe_step=not args.no_clahe,
            clahe_clip_limit=args.clip_limit,
            clahe_tile_grid_size=tuple(args.tile_grid),
        )

    print(f"Processed {len(paths)} image(s) into {output_directory}")

# ---------------------------------------------------------
# Functions for detecting and correcting image orientation and mirroring
# ---------------------------------------------------------
def fix_rotation_by_text_confidence(image: Image.Image) -> Image.Image:
    """Select the right-angle rotation that produces the most OCR text."""
    best_text_length = -1
    best_rotation = 0

    for angle in (0, 90, 180, 270):
        rotated = image.rotate(angle, expand=True)
        text = pytesseract.image_to_string(rotated, config="--psm 6")
        if len(text.strip()) > best_text_length:
            best_text_length = len(text.strip())
            best_rotation = angle

    print(f"Best rotation by OCR text length: {best_rotation} degrees")
    return image.rotate(best_rotation, expand=True)


def compare_ocr_confidence_mirrored(image: Image.Image, confidence_margin: float = 5.0) -> bool:
    """Compare normal and horizontally flipped OCR confidence. Return True if the image is likely mirrored (flipped reads better)."""
    def score(img: Image.Image) -> tuple[float, int]:
        """Return the average OCR confidence and text length for an image."""
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [float(c) for c in data["conf"] if float(c) >= 0]
        text = "".join(data["text"]).strip()
        avg_conf = float(np.median(confs)) if confs else 0.0
        return avg_conf, len(text)

    normal_conf, normal_len = score(image)
    flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    flipped_conf, flipped_len = score(flipped)

    print(
        f"Normal: conf={normal_conf:.1f}, chars={normal_len} | "
        f"Flipped: conf={flipped_conf:.1f}, chars={flipped_len}"
    )

    # Flipped is better if it reads better on BOTH axes, or much better on one.
    return (flipped_conf > normal_conf + confidence_margin) or (
        flipped_len > normal_len * 1.2 and flipped_conf >= normal_conf
    )


def detect_and_correct_orientation(
    image_path: Path, mirror_confidence_margin: float = 5.0
) -> Image.Image:
    """Correct right-angle rotation and a likely mirrored scan."""
    with Image.open(image_path) as source:
        image = source.copy()

    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotation = int(osd["rotate"])
        confidence = float(osd["orientation_conf"])
        print(f"Detected rotation: {rotation} degrees, confidence: {confidence:.1f}")
        if rotation:
            image = image.rotate(rotation, expand=True)
    except (pytesseract.TesseractError, KeyError, TypeError, ValueError):
        print("OSD failed; trying OCR text-length comparison")
        image = fix_rotation_by_text_confidence(image)

    if compare_ocr_confidence_mirrored(image, mirror_confidence_margin): # compare_ocr_confidence_mirrored runs the check and returns True or False. If True, perform the flip.
        print("Document appears mirrored; flipping horizontally")
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    return image

def process_orientation(args: argparse.Namespace) -> None:
    input_directory = args.input_directory
    output_directory = args.output_directory or input_directory.with_name(
        f"{input_directory.name}_corrected"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = image_paths(input_directory)

    for path in paths:
        print(f"Processing {path.name}")
        corrected = detect_and_correct_orientation(path, args.mirror_confidence_margin)
        corrected.save(output_directory / f"{path.stem}_corrected{path.suffix.lower()}")

    print(f"Processed {len(paths)} image(s) into {output_directory}")

# ---------------------------------------------------------
# Convert PDF pages to images and vice versa
# ---------------------------------------------------------
def pdf_to_images(pdf_path: Path, output_directory: Path, scale: float = 2.0) -> int:
    """Render every PDF page as a PNG image."""
    output_directory.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            pixmap.save(
                output_directory / f"{pdf_path.stem}-page-{page_number:04d}.png"
            )
        return len(document)


def process_pdf_to_images(args: argparse.Namespace) -> None:
    """Render one PDF or every PDF in a directory to images."""
    input_path = args.input_path

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        page_count = pdf_to_images(input_path, args.output_directory, args.scale)
        print(f"Rendered {page_count} page(s) into {args.output_directory}")
        return

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    pdf_paths = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise ValueError(f"No PDF files found in directory: {input_path}")

    total_pages = 0
    for pdf_path in pdf_paths:
        page_count = pdf_to_images(pdf_path, args.output_directory, args.scale)
        total_pages += page_count
        print(f"Rendered {pdf_path.name}: {page_count} page(s)")

    print(
        f"Rendered {total_pages} page(s) from {len(pdf_paths)} PDF(s) "
        f"into {args.output_directory}"
    )


def images_to_pdf(image_directory: Path, output_directory: Path) -> int:
    """Create one PDF per image in a directory."""
    paths = image_paths(image_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for path in paths:
        output_path = output_directory / f"{path.stem}.pdf"
        output_path.write_bytes(img2pdf.convert(str(path)))
    return len(paths)

# ---------------------------------------------------------
# Other
# ---------------------------------------------------------
def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed

# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess images and PDFs for use with olmOCR."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clahe = subparsers.add_parser(
        "clahe",
        help="Apply grayscale, blur, and CLAHE to an image directory.",
        description=(
            "Apply optional Gaussian blur and CLAHE to every supported image "
            "in a directory. Processed images are always written to the output "
            "directory."
        ),
    )
    clahe.add_argument(
        "input_directory",
        type=Path,
        help="Directory containing the input image files (image_path).",
    )
    clahe.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        help=(
            "Destination directory for processed images (output_path). "
            "Defaults to <input_directory>_clahe."
        ),
    )
    clahe.add_argument(
        "--no-blur",
        action="store_true",
        help=(
            "Disable Gaussian blur before CLAHE (apply_blur defaults to True). "
            "Blur can reduce noise."
        ),
    )
    clahe.add_argument(
        "--blur-kernel",
        type=int,
        nargs=2,
        default=(3, 3),
        metavar=("KX", "KY"),
        help=(
            "Gaussian kernel size (blur_kernel_size; default: 3 3). Both values "
            "must be odd; larger kernels apply stronger smoothing."
        ),
    )
    clahe.add_argument(
        "--blur-sigma",
        type=float,
        default=0.0,
        help=(
            "Gaussian sigma (default: 0). When 0, OpenCV derives sigma from "
            "the kernel size."
        ),
    )
    clahe.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE after blur (apply_clahe_step defaults to True).",
    )
    clahe.add_argument(
        "--clip-limit",
        type=positive_float,
        default=8.0,
        help=(
            "CLAHE clipping threshold (clahe_clip_limit; default: 8.0). Higher "
            "values increase local contrast."
        ),
    )
    clahe.add_argument(
        "--tile-grid",
        type=int,
        nargs=2,
        default=(32, 32),
        metavar=("COLS", "ROWS"),
        help=(
            "CLAHE tile grid size (clahe_tile_grid_size; default: 32 32). "
            "Smaller tiles increase local contrast effects."
        ),
    )
    clahe.set_defaults(handler=process_clahe)

    orientation = subparsers.add_parser(
        "correct-orientation", help="Correct rotated or mirrored images with Tesseract."
    )
    orientation.add_argument("input_directory", type=Path)
    orientation.add_argument("-o", "--output-directory", type=Path)
    orientation.add_argument(
        "--mirror-confidence-margin",
        type=float,
        default=5.0,
        help=(
            "Minimum amount by which the horizontally flipped image's OCR "
            "confidence must exceed the original image's confidence for it to "
            "be detected as mirrored (default: 5.0)."
        ),
    )
    orientation.set_defaults(handler=process_orientation)

    render = subparsers.add_parser(
        "pdf-to-image", help="Render a PDF or a directory of PDFs to PNG files."
    )
    render.add_argument("input_path", type=Path, help="PDF file or directory of PDFs.")
    render.add_argument("output_directory", type=Path)
    render.add_argument("--scale", type=positive_float, default=2.0)
    render.set_defaults(handler=process_pdf_to_images)

    convert = subparsers.add_parser(
        "image-to-pdf", help="Convert each image in a directory to a PDF."
    )
    convert.add_argument("input_directory", type=Path)
    convert.add_argument("output_directory", type=Path)
    convert.set_defaults(
        handler=lambda args: print(
            f"Converted {images_to_pdf(args.input_directory, args.output_directory)} "
            f"image(s) into {args.output_directory}"
        )
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, fitz.FileDataError) as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
