# About

olmOCR (https://github.com/allenai/olmocr) is a tool for turning PDFs and image-based documents into text that computers can work with. It is designed to handle difficult page layouts, handwriting, tables, equations, and other challenging document formats. This repository contains a preconfigured environment, supporting scripts, and instructions for installing and using the olmOCR. Please follow the instructions in this README file to get started.

# Prerequisites

### Nvidia Hardware

You’ll need a Nvidia graphics card with at least 16 GB of VRAM to run this code.

### Tesseract
Some image pre-processing steps use tesseract. It must be installed on the system.

**Linux**

Ubuntu or Debian
`sudo apt update`
`sudo apt install tesseract-ocr`

Fedora
`sudo dnf install tesseract`

Arch Linux
`sudo pacman -S tesseract`

**macOS**
First, install Homebrew (https://brew.sh/) if it is not already installed. 

Then run:
`brew install tesseract`

**Windows**
Open PowerShell and run:
`winget install --id UB-Mannheim.TesseractOCR`

Alternatively, if you use Chocolatey:
`choco install tesseract`

Check the installation by open a new Terminal or PowerShell window and run:
`tesseract --version`

If installation was successful, the command will display the installed Tesseract version.

### UV

You must have UV (https://docs.astral.sh/uv/getting-started/installation/) installed on the system

### Git
You must have Git (https://git-scm.com/install/) installed on the system

Then, use UV to configure the virtual environment:
- Run `uv sync` in the project directory
- You may need to manually enter the virtual environment by running `source .venv/bin/activate`

# Installation

1. Open a terminal and download the project:
`git clone {REPOSITORY_URL}`

2. Move into the project folder:
`cd {PROJECT_FOLDER}`

3. Install the required Python packages:
`uv sync`

4. Activate the project’s virtual environment.
Linux or macOS:
`source .venv/bin/activate`

Windows PowerShell:
`.venv\Scripts\Activate.ps1`

Keep this terminal open while using the project. You must activate the virtual environment again whenever you open a new terminal.

# Running from the Terminal

### Preprocessing

If necessary, run `image_preprocessing.py` from the project directory while the virtual environment is active. The script can improve image contrast, correct rotated or mirrored images, convert PDFs to images, and convert images to PDFs. Start by viewing the available commands:
`python image_preprocessing.py --help`

Each command also has its own options; for example:
`python image_preprocessing.py clahe --help`

Follow the help instructions to provide the input and output folders for the task you want to perform.

### OCR
Execute the OCR from the terminal by calling the `olmocr` command with the required options. OlmOCR works with PDF, PNG, and JPEG files.

Example for a single file:
`olmocr {OUTPUT_DIRECTORY} --markdown --pdfs {INPUT_FILES}/{FILE_NAME}.{FILE_EXTENSION} --gpu-memory-utilization .85`

Example for multiple files:
`olmocr {OUTPUT_DIRECTORY} --markdown --pdfs {INPUT_FILES}/*.{FILE_EXTENSION}--gpu-memory-utilization .85 --workers 2 --pages_per_group 3`

### Breakdown of the the CLI command

- `olmocr`: The name of the program you are telling the computer to run.
- `OUTPUT_DIRECTORY`: The folder where the finished files will be saved.
- `--markdown`: An instruction telling the program to save the OCR output in Markdown format.
- `--pdfs {INPUT_FILES}/*.{FILE_EXTENSION}`: The location of your source files. The `*` is a wildcard that tells the program to get every file with the following extension (such as PDF, JPEG, etc.) in that folder.
- `--gpu-memory-utilization .85`: A limit that tells the program it can use up to 85% of your graphics card's memory, leaving some room for other tasks.
- `--workers 2`: Tells the program to use 2 "workers" (simultaneous processes) at once to make the job go faster. How high you can set this will depend on the capabilities of your hardware.
- `--pages_per_group 3`: Tells the program to process the pages in batches of 3 for better efficiency. How high you can set this will depend on the capabilities of your hardware.

### After running
- Check the `FINAL METRICS SUMMARY` output in the terminal
- Optionally, run `extract_ocr_output_markdown.py`, which collects all Markdown files directly inside a specified folder and places them in a single ZIP file. To see all available options and an example, run:
`python extract_ocr_output_markdown.py --help`

### Troubleshooting
- Ensure Nvidia drivers, PyTorch, and CUDA are correctly configured on your system. Run `python test_cuda.py` to test the functionality of your configuration.
- Adjust the number of workers and the page group size to suit your system’s resources. Setting these values too high may use more VRAM or RAM than is available, causing the program to crash.
- Ensure the virtual environment is activated when executing olmOCR commands.