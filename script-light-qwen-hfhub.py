import os
import requests
from urllib.parse import urlparse
import multiprocessing
import time
import subprocess
import urllib.request
import sys

# Install pyngrok if not already installed
try:
    from pyngrok import ngrok, conf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
    from pyngrok import ngrok, conf

# CONFIGURATION
USE_GOOGLE_DRIVE = False
UPDATE_FORGE = True
INSTALL_DEPS = True
ALLOW_EXTENSION_INSTALLATION = True
USE_USERNAME_AND_PASSWORD = False
USERNAME = ""
PASSWORD = ""

WORKSPACE = 'stable-diffusion-webui-forge'

BRANCH = "neo"

# Skip Google Drive setup for local script

# Clone repo if not exists
os.environ.pop("HF_HUB_DISABLE_XET", None)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if not os.path.exists(WORKSPACE):
    print("-= Initial setup SDForge =-")
    subprocess.run([
        "git", "clone",
        "--config", "core.filemode=false",
        "-b", BRANCH,   # <-- branch argument
        "https://github.com/Haoming02/sd-webui-forge-classic",
        WORKSPACE
    ])

os.chdir(WORKSPACE)

if UPDATE_FORGE:
    print("-= Updating SDForge =-")
    subprocess.run(["git", "pull"])

if INSTALL_DEPS:
    print("-= Installing dependencies =-")
    subprocess.run(["sudo", "apt-get", "update", "-y"])
    subprocess.run(["sudo", "apt", "--fix-broken", "install", "-y"])
    subprocess.run(["sudo", "apt", "-y", "install", "-qq", "aria2", "ffmpeg"])
    subprocess.run(["pip", "install", "setuptools==80", "pip==25"])
    subprocess.run(["pip", "uninstall", "-y", "numpy", "pandas", "scikit-learn"])
    subprocess.run(["pip", "cache", "purge"])
    subprocess.run(["pip", "install", "numpy==1.26.4", "pandas==2.2.2", "scikit-learn==1.5.1", "--no-cache-dir"])
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    subprocess.run(
        ["uv", "pip", "install", "--system", "-q", "-U", "huggingface_hub[hf_xet]", "hf_xet"],
        check=False,
    )
# Fallback to pip
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub[hf_xet]", "hf_xet"],
        check=False,
    )


from concurrent.futures import ThreadPoolExecutor, as_completed


def download_hf_model(repo_id: str, filename: str, local_dir: str):
    from huggingface_hub import hf_hub_download

    os.makedirs(local_dir, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )
    print(f"Downloaded✅: {os.path.basename(path)}")
    return path


hf_download_jobs = [
    (
        "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
        "./models/Stable-diffusion",
    ),
    (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "./models/text_encoder",
    ),
    (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/vae/qwen_image_vae.safetensors",
        "./models/VAE",
    ),
]

print(f"Using {len(hf_download_jobs)} parallel workers for Hugging Face downloads")
with ThreadPoolExecutor(max_workers=len(hf_download_jobs)) as executor:
    futures = [
        executor.submit(download_hf_model, repo_id, filename, local_dir)
        for repo_id, filename, local_dir in hf_download_jobs
    ]
    for future in as_completed(futures):
        future.result()

# Refresh config file
try:
    os.remove("config.json")
except FileNotFoundError:
    pass

urllib.request.urlretrieve(
    "https://huggingface.co/phoenix-1708/DR_NED/resolve/main/config.json",
    "config.json"
)

# Refresh ui-config file
try:
    os.remove("ui-config.json")
except FileNotFoundError:
    pass

urllib.request.urlretrieve(
    "https://huggingface.co/phoenix-1708/DR_NED/resolve/main/ui-config.json",
    "ui-config.json"
)


# NGROK CONFIG
COMIC = True #hk1708
MIDGE = False #vk872

if COMIC:
    NGROK_AUTHTOKEN = "1ec4Az5OT9kYyABKxKnhLUl5jpH_7h6HjLDYkybGRS93M5aqG"
    Ngrok_domain = "comic-caribou-frankly.ngrok-free.app"
elif MIDGE:
    NGROK_AUTHTOKEN = "2SVfPekz6a7YTrM0V6FoZBeQblN_WV4nWZsvjRRAjSMubeS"
    Ngrok_domain = "midge-major-falcon.ngrok-free.app"
else:
    NGROK_AUTHTOKEN = None
    Ngrok_domain = None

User, Password = "", ""
auth = f"--gradio-auth {User}:{Password}" if User and Password else ""

# NGROK Tunnel
share = ""
if NGROK_AUTHTOKEN and Ngrok_domain:
    try:
        ngrok.kill()
        srv = ngrok.connect(
            7860,
            pyngrok_config=conf.PyngrokConfig(auth_token=NGROK_AUTHTOKEN),
            bind_tls=True,
            domain=Ngrok_domain
        ).public_url
        print(f"NGROK Link ✅: https://{Ngrok_domain}")
    except Exception as e:
        print(f"NGROK Error: {e}")
        share = "--share"
else:
    share = "--share"

# Launch WebUI
#LAUNCH_CMD = f"python launch.py --xformers-flash-attention --cuda-stream --always-high-vram --pin-shared-memory --api --listen --enable-insecure-extension-access --disable-console-progressbars --no-hashing --precision autocast --upcast-sampling"
#LAUNCH_CMD = f"python launch.py {share} --api --listen --cuda-stream --enable-insecure-extension-access {auth} --disable-console-progressbars"
LAUNCH_CMD = f"python launch.py {share} --api --listen --cuda-malloc --cuda-stream --pin-shared-memory --enable-insecure-extension-access {auth} --disable-console-progressbars"
print(f"Starting WebUI with command:\n{LAUNCH_CMD}")
os.system(LAUNCH_CMD)
