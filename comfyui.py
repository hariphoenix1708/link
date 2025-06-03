import os
import subprocess
import time
import re
import threading
import urllib.parse
import socket
import urllib.request

COMFYUI_DIR = "ComfyUI"
PORT = 8188

MODELS = [
    ("https://civitai.com/api/download/models/1761560?type=Model&format=SafeTensor&size=pruned&fp=fp16&token=a2fab2bb78fa416e631f84f42595e62e", "checkpoints"),
    ("https://civitai.com/api/download/models/1820690?type=Model&format=SafeTensor&size=pruned&fp=fp16&token=a2fab2bb78fa416e631f84f42595e62e", "checkpoints"),
    ("https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors", "vae"),
    ("https://civitai.com/api/download/models/1099059?type=Model&format=SafeTensor&token=a2fab2bb78fa416e631f84f42595e62e", "loras"),
    ("https://civitai.com/api/download/models/530662?type=Model&format=SafeTensor&token=a2fab2bb78fa416e631f84f42595e62e", "loras"),
    ("https://civitai.com/api/download/models/135867?type=Model&format=SafeTensor&token=a2fab2bb78fa416e631f84f42595e62e", "loras"),
    ("https://huggingface.co/phoenix-1708/DR_NED/resolve/main/Expressive_H-000001.safetensors", "loras"),
    ("https://huggingface.co/phoenix-1708/DR_NED/resolve/main/AissistXLv2.safetensors", "embeddings"),
    ("https://civitai.com/api/download/models/720175?type=Model&format=SafeTensor&token=a2fab2bb78fa416e631f84f42595e62e", "embeddings"),
    ("https://civitai.com/api/download/models/1223635?type=Model&format=PickleTensor&token=a2fab2bb78fa416e631f84f42595e62e", "embeddings"),
    ("https://civitai.com/api/download/models/775151?type=Model&format=SafeTensor&token=a2fab2bb78fa416e631f84f42595e62e", "embeddings"),
    ("https://civitai.com/api/download/models/82745?type=Negative&format=Other&token=a2fab2bb78fa416e631f84f42595e62e", "embeddings"),
    ("https://huggingface.co/phoenix-1708/DR_NED/resolve/main/AissistXLv2-neg.pt", "embeddings"),
    ("https://civitai.com/api/download/models/1690589?type=Negative&format=Other&token=a2fab2bb78fa416e631f84f42595e62e", "embeddings"),
    ("https://civitai.com/api/download/models/1223627?type=Negative&format=Other&token=a2fab2bb78fa416e631f84f42595e62e", "embeddings"),
    ("https://civitai.com/api/download/models/772342?type=Negative&format=Other&token=a2fab2bb78fa416e631f84f42595e62e", "embeddings"),
]

def run(cmd, cwd=None):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def stream_logs(process, name):
    for line in iter(process.stdout.readline, b''):
        print(f"[{name}] {line.decode().strip()}")

def get_filename_from_url(url):
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    for key in params:
        if "filename" in key.lower():
            return params[key][0]
    return os.path.basename(parsed.path)

# --- Install system packages ---
run("sudo apt update && sudo apt install -y aria2 git curl python3 python3-pip python3-venv")

# === INSTALL CLOUDFLARED ===
if not os.path.exists("/usr/local/bin/cloudflared"):
    run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb")
    run("sudo dpkg -i cloudflared-linux-amd64.deb || sudo apt install -f -y")
    run("rm cloudflared-linux-amd64.deb")

# --- Clone ComfyUI ---
if not os.path.exists(COMFYUI_DIR):
    run(f"git clone https://github.com/comfyanonymous/ComfyUI.git")

os.chdir(COMFYUI_DIR)

# --- Setup Python environment ---
run("pip install --upgrade pip")
run("pip cache purge")
run("pip uninstall -y torch torchvision xformers")
#run("pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121")
run("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128")
#run("pip install xformers!=0.0.18 --extra-index-url https://download.pytorch.org/whl/cu121")
run("pip install xformers!=0.0.18 --extra-index-url https://download.pytorch.org/whl/nightly/cu128 --no-deps")
run("pip install -r requirements.txt")

# --- Download models ---
for url, category in MODELS:
    folder = f"models/{category}"
    os.makedirs(folder, exist_ok=True)
    print(f"[*] Downloading {url} to {folder}")

    if "huggingface.co" in url:
        filename = get_filename_from_url(url)
        run(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M --allow-overwrite=true -o "{filename}" -d "{folder}" "{url}"')
    else:
        run(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d "{folder}" "{url}"')


# --- Start Cloudflared ---
def iframe_thread(port):
  while True:
      time.sleep(0.5)
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      result = sock.connect_ex(('127.0.0.1', port))
      if result == 0:
        break
      sock.close()
  print("\nComfyUI finished loading, trying to launch cloudflared (if it gets stuck here cloudflared is having issues)\n")

  p = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://127.0.0.1:{}".format(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  for line in p.stderr:
    l = line.decode()
    if "trycloudflare.com " in l:
      print("This is the URL to access ComfyUI:", l[l.find("http"):], end='')

threading.Thread(target=iframe_thread, daemon=True, args=(8188,)).start()


run("python3 main.py --dont-print-server --gpu-only")

'''
# --- Start ComfyUI server ---
print("[*] Starting ComfyUI server...")
comfy_proc = subprocess.Popen(
    "python3 main.py --port 8188",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    shell=True,
    executable="/bin/bash"
)
# threading.Thread(target=stream_logs, args=(comfy_proc, "ComfyUI"), daemon=True).start()
'''
