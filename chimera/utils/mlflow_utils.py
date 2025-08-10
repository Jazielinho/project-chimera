import hashlib
import platform
import subprocess

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

    :raises pynvml.NVMLError: If any error occurs during interaction with NVIDIA Management Library (NVML).
    """
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

        mlflow.set_tag("hardware.cpu.arch", platform.processor())
        mlflow.set_tag("hardware.os", f"{platform.system()} {platform.release()}")

        # --- Pasaporte de Software ---
        mlflow.set_tag("software.python.version", platform.python_version())
        mlflow.set_tag("software.torch.version", torch.__version__)
        mlflow.set_tag("software.cuda.version", torch.version.cuda)
        mlflow.set_tag("software.cudnn.version", torch.backends.cudnn.version())
        mlflow.set_tag("software.transformers.version", transformers.__version__)
        mlflow.set_tag("software.timm.version", timm.__version__)

        # --- Pasaporte de Código y Entorno ---
        mlflow.set_tag("code.git_commit_sha", get_git_commit_hash())
        mlflow.set_tag("code.conda_lock_hash", get_conda_lock_hash())

        print("Passport logged successfully.")
