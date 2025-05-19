import os
import subprocess
import sys
import time
import multiprocessing
import requests
from urllib.parse import urlparse

# Ensure pyngrok is installed
try:
    from pyngrok import ngrok, conf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
    from pyngrok import ngrok, conf

# ------------------- CONFIG -------------------
USE_GOOGLE_DRIVE = False
UPDATE_FORGE = True
INSTALL_DEPS = True
ALLOW_EXTENSION_INSTALLATION = True
USE_USERNAME_AND_PASSWORD = False
USERNAME = ""
PASSWORD = ""

WORKSPACE = '/kaggle/working/stable-diffusion-webui-forge'

# ------------------- CLONE REPO -------------------
if not os.path.exists(WORKSPACE):
    subprocess.run(["git", "clone", "--config", "core.filemode=false", "https://github.com/lllyasviel/stable-diffusion-webui-forge.git", WORKSPACE])
os.chdir(WORKSPACE)

# ------------------- UPDATE & INSTALL -------------------
if UPDATE_FORGE:
    subprocess.run(["git", "pull"])

if INSTALL_DEPS:
    subprocess.run(["apt-get", "update", "-y"])
    subprocess.run(["apt", "--fix-broken", "install", "-y"])
    subprocess.run(["apt", "-y", "install", "-qq", "aria2"])
    subprocess.run(["wget", "https://github.com/hariphoenix1708/link/raw/refs/heads/main/requirements.txt"])
    subprocess.run(["uv", "cache", "clean"])
    subprocess.run(["uv", "pip", "install", "--system", "-q", "-r", "requirements.txt"])

# ------------------- DOWNLOAD HELPERS -------------------
def get_filename_from_url(url):
    return os.path.basename(urlparse(url).path)

def get_latest_file(directory):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return max(files, key=lambda f: os.path.getctime(os.path.join(directory, f))) if files else None
    except Exception as e:
        print(f"Error: {e}")
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
    return max(4, min(multiprocessing.cpu_count() * 2, 8))

def clean_line(line):
    return line.split('**')[0].strip()

def download_from_txt(txt_url, token="a2fab2bb78fa416e631f84f42595e62e"):
    model_dirs = {
        "[]MODELS": f"{WORKSPACE}/models/Stable-diffusion",
        "[]VAE": f"{WORKSPACE}/models/VAE",
        "[]EMBEDDINGS": f"{WORKSPACE}/embeddings",
        "[]EMBEDDINGS-NEGATIVE": f"{WORKSPACE}/embeddings/Negative",
        "[]LORA": f"{WORKSPACE}/models/Lora",
        "[]EXTENSIONS": f"{WORKSPACE}/extensions"
    }

    current_dir = None
    civitai_commands = []
    huggingface_commands = []
    extension_commands = []

    response = requests.get(txt_url)
    if response.status_code != 200:
        print("Failed to fetch batch.txt")
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
        elif line.startswith("@"):
            repo = line[1:].strip()
            command = f'git clone "{repo}" "{model_dirs['[]EXTENSIONS']}/{repo.split('/')[-1]}"'
            extension_commands.append((command, "git", None))

    from concurrent.futures import ThreadPoolExecutor
    max_workers = get_optimal_workers()
    print(f"Using {max_workers} parallel workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda args: download_file(*args), civitai_commands)
        executor.map(lambda args: download_file(*args), huggingface_commands)
        executor.map(lambda args: download_file(*args), extension_commands)

# Download models
batch_url = "https://raw.githubusercontent.com/hariphoenix1708/link/refs/heads/main/batch.txt"
download_from_txt(batch_url)

# ------------------- CLEANUP AND CONFIG -------------------
config_path = os.path.join(WORKSPACE, "ui-config.json")
if os.path.exists(config_path):
    os.remove(config_path)
os.chdir(WORKSPACE)
subprocess.run(["wget", "-q", "https://raw.githubusercontent.com/hariphoenix1708/link/refs/heads/main/ui-config.json"])

# ------------------- NGROK SETUP -------------------
ORIENTED = False
ADEQUATE = True

if ORIENTED:
    NGROK_AUTHTOKEN = "2uWVKLlphDhoyBtn4wlGPFlDumV_L7iq1bQBBRcmtFBc8oJQ"
    Ngrok_domain = "oriented-definitely-shepherd.ngrok-free.app"
elif ADEQUATE:
    NGROK_AUTHTOKEN = "2uze4kgXnRaAOX4za98T3CPThn5_6Po5HS2TbQTfMKSMLnmT4"
    Ngrok_domain = "adequate-globally-herring.ngrok-free.app"
else:
    NGROK_AUTHTOKEN = None
    Ngrok_domain = None

User, Password = "", ""
auth = f"--gradio-auth {User}:{Password}" if User and Password else ""

if not os.path.exists(WORKSPACE):
    print(f"Error: {WORKSPACE} does not exist!")
    sys.exit(1)

os.chdir(WORKSPACE)
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

print(f"Launching Stable Diffusion at: https://{Ngrok_domain if Ngrok_domain else 'gradio.live'}")
subprocess.run([
    "python", "launch.py",
    share,
    "--gpu-device-id", "1",
    "--xformers",
    "--cuda-stream",
    "--always-gpu",
    "--pin-shared-memory",
    "--api",
    "--listen",
    "--enable-insecure-extension-access",
    auth,
    "--disable-console-progressbars",
    "--no-hashing",
    "--precision", "autocast",
    "--upcast-sampling"
])
