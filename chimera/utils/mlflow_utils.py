import hashlib
import os
import platform
import socket
import subprocess
from datetime import datetime

import mlflow
import pynvml
import timm
import torch
import transformers


def get_git_commit_hash():
    """
    Retrieve the latest Git commit hash of the current repository.

    This function attempts to run the git command to obtain the latest commit hash
    of the current Git repository. If the command fails (e.g., if the directory is
    not a Git repository), it returns "N/A".

    :raises subprocess.CalledProcessError: If the git command fails to execute properly.

    :return: The Git commit hash as a string if successful, otherwise "N/A".
    :rtype: str
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except subprocess.CalledProcessError:
        return "N/A"


def get_git_info():
    """Get git repository information."""
    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        return {
            "git_branch": git_branch,
            "git_commit": git_commit,
            "git_dirty": bool(git_status),
        }
    except Exception as e:
        return {"git_error": str(e)}


def get_python_packages():
    """Get key installed Python packages and their versions."""
    try:
        import pkg_resources

        packages = {
            pkg.key: pkg.version
            for pkg in pkg_resources.working_set
            if pkg.key
            in [
                "torch",
                "torchvision",
                "numpy",
                "transformers",
                "pandas",
                "mlflow",
                "pillow",
                "timm",
            ]
        }
        return packages
    except Exception as e:
        return {"package_error": str(e)}


def get_conda_lock_hash(lockfile_path: str = "conda-lock.yml"):
    """
    Computes and returns the SHA-256 hash of the specified conda lock file. If the lock file
    does not exist, it returns "N/A".

    :param lockfile_path: The path to the conda lock file. Defaults to "conda-lock.yml".
    :type lockfile_path: str
    :return: The SHA-256 hash of the lock file content, or "N/A" if the file is not found.
    :rtype: str
    """
    try:
        with open(lockfile_path, "rb") as f:
            lockfile_content = f.read()
            return hashlib.sha256(lockfile_content).hexdigest()
    except FileNotFoundError:
        return "N/A"


def _safe_decode(value: str | bytes) -> str:
    """
    Safely decodes a given value into a UTF-8 string. If the input value is
    a byte object, it decodes it. If the input is already a string, it
    converts it to a standard string object.

    :param value: The value to decode. This can be either a string or bytes.
    :type value: str | bytes
    :return: A UTF-8 decoded string from the input value.
    :rtype: str
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def log_reproducibility_passport():
    """
    Logs detailed hardware, software, and code environment information to MLflow tags to ensure reproducibility
    of the experiment. This information serves as a reproducibility passport, capturing the system's key
    characteristics such as GPU details, operating system, Python and library versions, and versioning details
    of the code and environment.

    The function logs:
    - Hardware information: GPU details (via NVML), CPU architecture, and OS
    - Software information: Python, PyTorch, CUDA, cuDNN, Transformers, and TIMM versions
    - Code information: Git commit hash and environment configuration checksum

    :raises pynvml.NVMLError: If any error occurs during interaction with NVIDIA Management Library (NVML).
    """
    # --- Hardware Information ---
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = _safe_decode(pynvml.nvmlDeviceGetName(handle))
        gpu_driver = _safe_decode(pynvml.nvmlSystemGetDriverVersion())
        total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
        mlflow.set_tag("hardware.gpu.name", gpu_name)
        mlflow.set_tag("hardware.gpu.driver", gpu_driver)
        mlflow.set_tag("hardware.gpu.vram_gb", f"{total_memory:.2f}")
    except pynvml.NVMLError:
        mlflow.set_tag("hardware.gpu", "N/A")
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    # Additional hardware information
    mlflow.set_tag("hardware.hostname", socket.gethostname())
    mlflow.set_tag("hardware.cpu.arch", platform.processor())
    mlflow.set_tag("hardware.cpu.count", os.cpu_count())
    mlflow.set_tag("hardware.os", f"{platform.system()} {platform.release()}")
    mlflow.set_tag("hardware.timestamp", datetime.now().isoformat())

    # --- Software Information ---
    mlflow.set_tag("software.python.version", platform.python_version())
    mlflow.set_tag("software.torch.version", torch.__version__)
    mlflow.set_tag("software.cuda.version", torch.version.cuda)
    mlflow.set_tag("software.cudnn.version", torch.backends.cudnn.version())
    mlflow.set_tag("software.transformers.version", transformers.__version__)
    mlflow.set_tag("software.timm.version", timm.__version__)

    # --- CUDA Detailed Information (if available) ---
    if torch.cuda.is_available():
        mlflow.set_tag("cuda.arch_list", torch.cuda.get_arch_list())
        mlflow.set_tag("cuda.device_count", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            mlflow.set_tag(f"cuda.device{i}.name", device_name)
            props = torch.cuda.get_device_properties(i)
            mlflow.set_tag(f"cuda.device{i}.capability", f"{props.major}.{props.minor}")
            mlflow.set_tag(
                f"cuda.device{i}.total_memory_gb",
                round(props.total_memory / (1024**3), 2),
            )

    # --- Python Key Packages ---
    packages = get_python_packages()
    for key, value in packages.items():
        mlflow.set_tag(f"package.{key}", value)

    # --- Code and Environment Information ---
    # Git information
    git_info = get_git_info()
    for key, value in git_info.items():
        mlflow.set_tag(f"git.{key}", value)

    mlflow.set_tag("code.git_commit_sha", get_git_commit_hash())
    mlflow.set_tag("code.conda_lock_hash", get_conda_lock_hash())

    print("Reproducibility passport logged successfully.")
