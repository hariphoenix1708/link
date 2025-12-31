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

# Skip Google Drive setup for local script

# Clone repo if not exists
if not os.path.exists(WORKSPACE):
    print("-= Initial setup SDForge =-")
    subprocess.run(["git", "clone", "--config", "core.filemode=false",
                    "https://github.com/Panchovix/stable-diffusion-webui-reForge.git", WORKSPACE])

os.chdir(WORKSPACE)

if UPDATE_FORGE:
    print("-= Updating SDForge =-")
    subprocess.run(["git", "pull"])

if INSTALL_DEPS:
    print("-= Installing dependencies =-")
    subprocess.run(["apt-get", "update", "-y"])
    subprocess.run(["apt", "--fix-broken", "install", "-y"])
    subprocess.run(["apt", "-y", "install", "-qq", "aria2"])
    urllib.request.urlretrieve(
        "https://github.com/hariphoenix1708/link/raw/refs/heads/main/requirements.txt",
        "requirements.txt"
    )
    subprocess.run(["uv", "cache", "clean"])
    subprocess.run(["uv", "pip", "install", "--system", "-q", "-r", "requirements.txt"])


def get_filename_from_url(url):
    return os.path.basename(urlparse(url).path)

def get_latest_file(directory):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return max(files, key=lambda f: os.path.getctime(os.path.join(directory, f))) if files else None
    except Exception as e:
        print(f"Error finding latest file in {directory}: {e}")
        return None

def download_file(command, category, target_dir=None):
    os.system(command)
    if category == "civitai" and target_dir:
        time.sleep(2)
        latest_file = get_latest_file(target_dir)
        if latest_file:
            print(f"Downloaded✅: {latest_file}")
    elif category == "huggingface":
        filename = command.split('-o "')[1].split('"')[0] if '-o "' in command else "Unknown"
        print(f"Downloaded✅: {filename}")
    elif category == "git":
        repo_name = command.split('/')[-1].replace('"', '')
        print(f"Downloaded✅: {repo_name}")

def get_optimal_workers():
    cpu_count = multiprocessing.cpu_count()
    return max(4, min(cpu_count * 2, 8))

def clean_line(line):
    return line.split('**')[0].strip()

def download_from_txt(txt_url, token="a2fab2bb78fa416e631f84f42595e62e"):
    from concurrent.futures import ThreadPoolExecutor

    model_dirs = {
        "[]MODELS": "./models/Stable-diffusion",
        "[]VAE": "./models/VAE",
        "[]EMBEDDINGS": "./embeddings",
        "[]EMBEDDINGS-NEGATIVE": "./embeddings/Negative",
        "[]LORA": "./models/Lora",
        "[]EXTENSIONS": "./extensions"
    }

    current_dir = None
    civitai_commands = []
    huggingface_commands = []
    extension_commands = []

    response = requests.get(txt_url)
    if response.status_code != 200:
        print("Failed to fetch the file.")
        return

    lines = response.text.strip().split("\n")

    for line in lines:
        line = clean_line(line)

        if line.startswith("[]") and line in model_dirs:
            current_dir = model_dirs[line]
            os.makedirs(current_dir, exist_ok=True)
            print(f"Using directory: {current_dir}")

        elif line.startswith("$") and current_dir:
            link = line[1:].strip()
            if "civitai.com" in link:
                if "token=" not in link:
                    link += f"&token={token}" if "?" in link else f"?token={token}"
                command = f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{link}" -d "{current_dir}"'
                civitai_commands.append((command, "civitai", current_dir))
            else:
                filename = get_filename_from_url(link)
                command = f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{link}" -d "{current_dir}" -o "{filename}"'
                huggingface_commands.append((command, "huggingface", None))

        elif line.startswith("@") and model_dirs["[]EXTENSIONS"]:
            repo = line[1:].strip()
            ext_path = f"{model_dirs['[]EXTENSIONS']}/{repo.split('/')[-1]}"
            command = f'git clone "{repo}" "{ext_path}"'
            extension_commands.append((command, "git", None))

    max_workers = get_optimal_workers()
    print(f"Using {max_workers} parallel workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda args: download_file(*args), civitai_commands)
        executor.map(lambda args: download_file(*args), huggingface_commands)
        executor.map(lambda args: download_file(*args), extension_commands)


# Execute download step
txt_file_url = "https://raw.githubusercontent.com/hariphoenix1708/link/refs/heads/main/batch.txt"
download_from_txt(txt_file_url)

# Refresh config file
try:
    os.remove("ui-config.json")
except FileNotFoundError:
    pass

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/hariphoenix1708/link/refs/heads/main/ui-config.json",
    "ui-config.json"
)

import zipfile

# Extract Lora zip if it exists
lora_zip_path = "./models/Lora/details.zip"
lora_extract_dir = "./models/Lora"

if os.path.exists(lora_zip_path):
    try:
        print(f"Extracting Lora models from {lora_zip_path}...")
        with zipfile.ZipFile(lora_zip_path, 'r') as zip_ref:
            zip_ref.extractall(lora_extract_dir)
        print("Extraction complete ✅")
        os.remove(lora_zip_path)
        print("Cleaned up zip file 🗑️")
    except zipfile.BadZipFile:
        print("Error: Bad ZIP file!")
else:
    print(f"No zip file found at {lora_zip_path}")


# NGROK CONFIG
ORIENTED = False
ADEQUATE = True

if ORIENTED:
    NGROK_AUTHTOKEN = "2uWVKLlphDhoyBtn4wlGPFlDumV_L7iq1bQBBRcmtFBc8oJQ"
    Ngrok_domain = "oriented-definitely-shepherd.ngrok-free.app"
elif ADEQUATE:
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
LAUNCH_CMD = f"python launch.py {share} --gpu-device-id 1 --xformers --cuda-stream --always-gpu --pin-shared-memory --api --listen --enable-insecure-extension-access {auth} --disable-console-progressbars --no-hashing --precision autocast --upcast-sampling"
print(f"Starting WebUI with command:\n{LAUNCH_CMD}")
os.system(LAUNCH_CMD)
