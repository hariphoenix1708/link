import os
import subprocess
import fileinput
import urllib.request
from urllib.parse import urlparse
import sys
import requests
import multiprocessing
import time


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

WORKSPACE = 'stable-diffusion-webui-forge'

# Clone repo if not exists
if not os.path.exists(WORKSPACE):
    print("-= Initial setup SDForge =-")
    subprocess.run(["git", "clone", "--config", "core.filemode=false",
                    "https://github.com/lllyasviel/stable-diffusion-webui-forge.git", WORKSPACE])

os.chdir(WORKSPACE)

if UPDATE_FORGE:
    print("-= Updating SDForge =-")
    subprocess.run(["git", "pull"])

if INSTALL_DEPS:
    print("-= Installing dependencies =-")
    subprocess.run(["sudo", "apt-get", "update", "-y"])
    subprocess.run(["sudo", "apt", "--fix-broken", "install", "-y"])
    subprocess.run(["sudo", "apt", "-y", "install", "-qq", "aria2"])
    subprocess.run(["pip", "cache", "purge"])
    subprocess.run(["pip", "uninstall", "-y", "xformers", "torch", "torchvision", "torchaudio"])

# --- Choose only ONE NGROK configuration ---
ORIENTED = False
ADEQUATE = True

#@markdown # ORIENTED
if ORIENTED:
    NGROK_AUTHTOKEN = "2uWVKLlphDhoyBtn4wlGPFlDumV_L7iq1bQBBRcmtFBc8oJQ" # @param {type:"string"}
    Ngrok_domain = "oriented-definitely-shepherd.ngrok-free.app" # @param {type:"string"}
#@markdown # ADEQUATE
elif ADEQUATE:
    NGROK_AUTHTOKEN = "2uze4kgXnRaAOX4za98T3CPThn5_6Po5HS2TbQTfMKSMLnmT4" # @param {type:"string"}
    Ngrok_domain = "adequate-globally-herring.ngrok-free.app" # @param {type:"string"}
else:
    NGROK_AUTHTOKEN = None
    Ngrok_domain = None

# --- Optional Authentication ---
User, Password = "", ""  # Set credentials if needed
auth = f"--gradio-auth {User}:{Password}" if User and Password else ""

# --- Stable Diffusion Path ---
SD_PATH = "/teamspace/studios/this_studio/stable-diffusion-webui-forge"
if not os.path.exists(SD_PATH):
    print(f"Error: {SD_PATH} does not exist!")
    sys.exit(1)

os.chdir(SD_PATH)

# --- NGROK Setup ---
share = ''
if NGROK_AUTHTOKEN and Ngrok_domain:
    try:
        ngrok.kill()  # Ensure no existing tunnels
        srv = ngrok.connect(
            7860,
            pyngrok_config=conf.PyngrokConfig(auth_token=NGROK_AUTHTOKEN),
            bind_tls=True,
            domain=Ngrok_domain
        ).public_url
        print(f"NGROK Link ✅: https://{Ngrok_domain}")
    except Exception as e:
        print(f"NGROK Error: {e}")
        share = '--share'  # Fallback to Gradio share
else:
    share = '--share'

# --- Launch Stable Diffusion ---

print(f"Link✅: https://{Ngrok_domain}")

!python {SD_PATH}/launch.py --api --listen --enable-insecure-extension-access --disable-console-progressbars --no-hashing --precision autocast --upcast-sampling
