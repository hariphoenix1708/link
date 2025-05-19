import os
import sys
import time
import requests
import subprocess
import multiprocessing
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from pyngrok import ngrok, conf

# --- CONFIGURATION ---
USE_GOOGLE_DRIVE = False
UPDATE_FORGE = True
INSTALL_DEPS = True
ALLOW_EXTENSION_INSTALLATION = True
USE_USERNAME_AND_PASSWORD = False
USERNAME = ""
PASSWORD = ""

WORKSPACE = 'stable-diffusion-webui-forge'
SD_PATH = f"/kaggle/working/{WORKSPACE}"

# --- Helpers ---
def run_cmd(command, shell=True):
    """Runs a shell command with optional error output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=shell)
    if result.returncode != 0:
        print(f"Command failed: {command}")
    return result

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
    run_cmd(command)
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
    model_dirs = {
        "[]MODELS": f"{SD_PATH}/models/Stable-diffusion",
        "[]VAE": f"{SD_PATH}/models/VAE",
        "[]EMBEDDINGS": f"{SD_PATH}/embeddings",
        "[]EMBEDDINGS-NEGATIVE": f"{SD_PATH}/embeddings/Negative",
        "[]LORA": f"{SD_PATH}/models/Lora",
        "[]EXTENSIONS": f"{SD_PATH}/extensions"
    }

    current_dir = None
    civitai_commands, huggingface_commands, extension_commands = [], [], []

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
        elif line.startswith("@"):
            repo = line[1:].strip()
            command = f'git clone "{repo}" "{model_dirs["[]EXTENSIONS"]}/{repo.split("/")[-1]}"'
            extension_commands.append((command, "git", None))

    max_workers = get_optimal_workers()
    print(f"Using {max_workers} parallel workers")
    for commands in [civitai_commands, huggingface_commands, extension_commands]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(lambda args: download_file(*args), commands)

def setup_ngrok():
    ORIENTED = False
    ADEQUATE = True

    if ORIENTED:
        authtoken = "2uWVKLlphDhoyBtn4wlGPFlDumV_L7iq1bQBBRcmtFBc8oJQ"
        domain = "oriented-definitely-shepherd.ngrok-free.app"
    elif ADEQUATE:
        authtoken = "2uze4kgXnRaAOX4za98T3CPThn5_6Po5HS2TbQTfMKSMLnmT4"
        domain = "adequate-globally-herring.ngrok-free.app"
    else:
        return "--share"

    try:
        ngrok.kill()
        ngrok.connect(
            7860,
            pyngrok_config=conf.PyngrokConfig(auth_token=authtoken),
            bind_tls=True,
            domain=domain
        )
        print(f"NGROK Link ✅: https://{domain}")
        return ""
    except Exception as e:
        print(f"NGROK Error: {e}")
        return "--share"

def main():
    os.makedirs(SD_PATH, exist_ok=True)
    os.chdir(SD_PATH)

    if UPDATE_FORGE:
        if not os.path.exists(SD_PATH):
            run_cmd(f'git clone --config core.filemode=false https://github.com/lllyasviel/stable-diffusion-webui-forge.git "{SD_PATH}"')
        os.chdir(SD_PATH)
        run_cmd("git pull")

    if INSTALL_DEPS:
        run_cmd("apt-get update -y")
        run_cmd("apt --fix-broken install -y")
        run_cmd("apt -y install -qq aria2")
        run_cmd(f"wget https://github.com/hariphoenix1708/link/raw/refs/heads/main/requirements.txt")
        run_cmd("uv cache clean")
        run_cmd(f"uv pip install --system -q -r {SD_PATH}/requirements.txt")

    txt_file_url = "https://raw.githubusercontent.com/hariphoenix1708/link/refs/heads/main/batch.txt"
    download_from_txt(txt_file_url)

    # Download updated UI config
    ui_config_path = os.path.join(SD_PATH, "ui-config.json")
    if os.path.exists(ui_config_path):
        os.remove(ui_config_path)
    run_cmd(f"wget -q https://raw.githubusercontent.com/hariphoenix1708/link/refs/heads/main/ui-config.json -P {SD_PATH}")

    # NGROK
    auth = f"--gradio-auth {USERNAME}:{PASSWORD}" if USE_USERNAME_AND_PASSWORD else ""
    share = setup_ngrok()

    # Launch app
    launch_cmd = f"python {SD_PATH}/launch.py {share} --gpu-device-id 1 --xformers --cuda-stream --always-gpu --pin-shared-memory --api --listen --enable-insecure-extension-access {auth} --disable-console-progressbars --no-hashing --precision autocast --upcast-sampling"
    run_cmd(launch_cmd)

if __name__ == "__main__":
    main()
