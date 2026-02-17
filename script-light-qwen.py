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

    #subprocess.run(["pip", "cache", "purge"])
    subprocess.run(["pip", "uninstall", "-y", "numpy", "pandas", "scikit-learn"])
    subprocess.run(["pip", "cache", "purge"])
    subprocess.run(["pip", "install", "numpy==1.26.4", "pandas==2.2.2", "scikit-learn==1.5.1", "--no-cache-dir"])
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    #subprocess.run(["pip", "uninstall", "-y", "numpy", "pandas", "scikit-learn"])
    #subprocess.run(["pip", "cache", "purge"])
    #subprocess.run(["pip", "install", "numpy==1.26.4", "pandas==2.2.2", "scikit-learn==1.5.1", "--no-cache-dir"])
    #subprocess.run(["pip", "install", "pickleshare"])
    #subprocess.run(["pip", "install", "basicsr"])
    #subprocess.run(["pip", "install", "insightface"])
    #subprocess.run(["pip", "uninstall", "-y", "xformers", "torch", "torchvision", "torchaudio"])

subprocess.run([
    "aria2c",
    "--console-log-level=error",
    "-c",
    "-x", "16",
    "-s", "16",
    "-k", "1M",
    "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
    "-d", "./models/Stable-diffusion",
    "-o", "qwen_image_edit.safetensors"
])

subprocess.run([
    "aria2c",
    "--console-log-level=error",
    "-c",
    "-x", "16",
    "-s", "16",
    "-k", "1M",
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "-d", "./models/text_encoder",
    "-o", "qwen_2.5_vl_7b_fp8_scaled.safetensors"
])

subprocess.run([
    "aria2c",
    "--console-log-level=error",
    "-c",
    "-x", "16",
    "-s", "16",
    "-k", "1M",
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
    "-d", "./models/VAE",
    "-o", "qwen_image_vae.safetensors"
])


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
LAUNCH_CMD = f"python launch.py {share} --api --listen --fast-fp8 --enable-insecure-extension-access {auth} --disable-console-progressbars"
print(f"Starting WebUI with command:\n{LAUNCH_CMD}")
os.system(LAUNCH_CMD)
