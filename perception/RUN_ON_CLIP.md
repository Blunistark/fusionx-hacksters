# Run multi-model detection on clip1.mp4

Follow these steps from the repository root: `C:\Users\ayush\OneDrive\Desktop\Hacsters_project\fusionx-hacksters`.

1) Open PowerShell and change to the repo root

```powershell
cd C:\Users\ayush\OneDrive\Desktop\Hacsters_project\fusionx-hacksters
```

2) Create and activate a Python virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate
```

3) Install dependencies

```powershell
pip install --upgrade pip
pip install -r perception\requirements.txt
```

4) Run the multi-model detector on your clip

```powershell
python perception\multi_detect.py --input "C:\Users\ayush\OneDrive\Desktop\Hacsters_project\fusionx-hacksters\assets\data\clip1.mp4" --output-dir perception\output --device cpu
```

Options:
- Use `--device cuda` if you have a suitable GPU and CUDA-enabled PyTorch installed.
- Change backends or model files with `--person-backend`, `--obj-backend`, `--person-model`, `--obj-model`.

5) Check outputs

The script writes three files to `perception/output`:

- `<basename>_person.mp4`
- `<basename>_objects.mp4`
- `<basename>_combined.mp4`

Troubleshooting
- First run may download model weights — allow time and ensure internet access.
- If `torch` installation fails, install the CUDA wheel from https://pytorch.org then re-run `pip install -r perception\requirements.txt`.
- If a backend errors (ultralytics/yolov5), ensure its dependencies are installed and compatible with your Python version.

Optional: run the tracker for live/demo

```powershell
python perception\tracker.py
```
