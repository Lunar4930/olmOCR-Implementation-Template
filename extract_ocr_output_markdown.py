"""Archive the Markdown files from a directory into a ZIP file."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def archive_markdown_files(source_directory: Path, output_path: Path) -> int:
    """Add every Markdown file in a directory to a ZIP archive."""
    if not source_directory.is_dir():
        raise ValueError(f"Source directory does not exist: {source_directory}")

    markdown_files = sorted(
        path
        for path in source_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".markdown"}
    )
    if not markdown_files:
        raise ValueError(f"No Markdown files found in: {source_directory}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for markdown_file in markdown_files:
            archive.write(markdown_file, arcname=markdown_file.name)

    return len(markdown_files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect all .md and .markdown files directly inside a directory "
            "and store them in a compressed ZIP archive."
        ),
        epilog=(
            "Example: python extract_ocr_output_markdown.py "
            "output/markdown/ocr_ready --output ocr_markdown.zip"
        ),
    )
    parser.add_argument(
        "source_directory",
        type=Path,
        help="Directory containing the Markdown files to archive.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Name or path of the ZIP file. Defaults to "
            "<source-directory-name>_markdown.zip in the current working directory."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = args.output or Path(
        f"{args.source_directory.resolve().name}_markdown.zip"
    )
    if output_path.suffix.lower() != ".zip":
        output_path = output_path.with_suffix(".zip")

    try:
        file_count = archive_markdown_files(args.source_directory, output_path)
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 1

    print(f"Archived {file_count} Markdown file(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
