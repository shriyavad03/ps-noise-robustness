"""One-time setup for the GPU extraction pipeline: installs MediaPipe /
RTMPose / onnxruntime-gpu, points onnxruntime at the pip-installed CUDA
libraries, and downloads the hand-landmarker model. Not needed if your
environment already has these packages and CUDA libraries set up."""
import os
import site
import subprocess
import sysconfig
import urllib.request
from glob import glob
from pathlib import Path

PACKAGES = [
    "mediapipe",
    "rtmlib",
    "onnxruntime-gpu==1.19.2",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12==9.*",
    "nvidia-cublas-cu12",
]


def install_packages():
    for pkg in PACKAGES:
        subprocess.run(["pip", "install", pkg], check=True)


def _site_packages_dirs():
    """Every site-packages dir this Python could have installed into
    (works the same in a venv, conda env, or system Python)."""
    dirs = set(site.getsitepackages()) if hasattr(site, "getsitepackages") else set()
    dirs.add(sysconfig.get_paths()["purelib"])
    return [d for d in dirs if d]


def configure_cuda_ld_library_path():
    """Adds every pip-installed nvidia-*-cuXX package's lib/ dir to
    LD_LIBRARY_PATH, so onnxruntime-gpu can find CUDA at import time."""
    cuda_lib_paths = []
    for site_dir in _site_packages_dirs():
        cuda_lib_paths += glob(f"{site_dir}/nvidia/*/lib")
    if cuda_lib_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_lib_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")


def download_hand_landmarker_model(model_url, model_cache: Path):
    if model_cache.exists():
        return
    model_cache.parent.mkdir(parents=True, exist_ok=True)
    print("[INFO] Downloading hand-landmarker model ...")
    urllib.request.urlretrieve(model_url, model_cache)


def run_full_setup():
    install_packages()
    configure_cuda_ld_library_path()

    import onnxruntime as ort
    print(f"[INFO] onnxruntime {ort.__version__}, providers: {ort.get_available_providers()}")

    from . import config
    download_hand_landmarker_model(config.MODEL_URL, config.MODEL_CACHE)
