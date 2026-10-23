"""
Author: Mohammad Khan
mkhan2024@fau.edu
"""
######################################### import start #########################################

from hashlib import sha256
from pathlib import Path
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import torch
except ImportError:
    sys.exit(
        "Error: PyTorch is not installed. Run "
        "'pip install -r requirements.txt' first."
    )
######################################### import end #########################################


WEIGHT_NAME = "inceptionresnetv2-520b38e4.pth"

DOWNLOAD_URL = (
    "https://github.com/md-bd/SHADES/releases/download/"
    f"v1.0.0/{WEIGHT_NAME}"
)

EXPECTED_SHA256 = (
    "520b38e4fe58682abd64eeeb2057c71d0d3a50a95b9bce9dabbaae2be8d5d632"
)

# This file is located in src/, so its parent directory is the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPOSITORY_ROOT / "models"
MODEL_PATH = MODELS_DIR / WEIGHT_NAME
TEMPORARY_PATH = MODELS_DIR / f"{WEIGHT_NAME}.part"

# pretrainedmodels uses the standard PyTorch checkpoint cache.
TORCH_CHECKPOINT_DIR = Path(torch.hub.get_dir()) / "checkpoints"
TORCH_CHECKPOINT_PATH = TORCH_CHECKPOINT_DIR / WEIGHT_NAME


def calculate_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def verify_file(path: Path) -> bool:
    """Return True when a file exists and matches the expected SHA-256."""
    if not path.is_file():
        return False

    print(f"Verifying: {path}")
    actual_sha256 = calculate_sha256(path)

    if actual_sha256 != EXPECTED_SHA256:
        print("SHA-256 verification failed.")
        print(f"Expected: {EXPECTED_SHA256}")
        print(f"Found:    {actual_sha256}")
        return False

    print("SHA-256 verification succeeded.")
    return True


def download_weight() -> None:
    """Download the checkpoint into models/ using a temporary file."""
    TEMPORARY_PATH.unlink(missing_ok=True)

    print(f"Downloading {WEIGHT_NAME}...")
    print(f"Source: {DOWNLOAD_URL}")

    request = Request(
        DOWNLOAD_URL,
        headers={"User-Agent": "SHADES-weight-setup"},
    )

    try:
        with urlopen(request) as response:
            with TEMPORARY_PATH.open("wb") as output:
                shutil.copyfileobj(response, output)
    except (HTTPError, URLError, OSError) as error:
        TEMPORARY_PATH.unlink(missing_ok=True)
        sys.exit(f"Error downloading checkpoint: {error}")

    if not verify_file(TEMPORARY_PATH):
        TEMPORARY_PATH.unlink(missing_ok=True)
        sys.exit(
            "Error: The downloaded checkpoint is incomplete, corrupted, "
            "or different from the expected file."
        )

    TEMPORARY_PATH.replace(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


def install_in_torch_cache() -> None:
    """Copy the verified model to the cache expected by pretrainedmodels."""
    TORCH_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if verify_file(TORCH_CHECKPOINT_PATH):
        print(
            "PyTorch checkpoint is already installed: "
            f"{TORCH_CHECKPOINT_PATH}"
        )
        return

    if TORCH_CHECKPOINT_PATH.exists():
        print("Removing invalid checkpoint from the PyTorch cache.")
        TORCH_CHECKPOINT_PATH.unlink()

    shutil.copy2(MODEL_PATH, TORCH_CHECKPOINT_PATH)

    if not verify_file(TORCH_CHECKPOINT_PATH):
        TORCH_CHECKPOINT_PATH.unlink(missing_ok=True)
        sys.exit("Error: Checkpoint copy verification failed.")

    print(f"PyTorch checkpoint installed at: {TORCH_CHECKPOINT_PATH}")


def main() -> None:
    # Create models/ automatically if it does not exist.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if verify_file(MODEL_PATH):
        print(f"Using existing model: {MODEL_PATH}")
    else:
        if MODEL_PATH.exists():
            print("Removing invalid model from models/.")
            MODEL_PATH.unlink()

        download_weight()

    install_in_torch_cache()
    print("Weight setup completed successfully.")


if __name__ == "__main__":
    main()

