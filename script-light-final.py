import os
import re
import requests
from urllib.parse import urlparse, parse_qs, unquote
import multiprocessing
import time
import subprocess
import urllib.request
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Civitai API token (set this as an environment variable in Kaggle)
# Example:
os.environ["CIVITAI_TOKEN"] = "a2fab2bb78fa416e631f84f42595e62e"
CIVITAI_TOKEN = os.getenv("CIVITAI_TOKEN", "")

MAX_RETRIES = 3

WORKSPACE = 'stable-diffusion-webui-forge'

# Skip Google Drive setup for local script

# Clone repo if not exists
if not os.path.exists(WORKSPACE):
    print("-= Initial setup SDForge =-")
    subprocess.run([
        "git", "clone", "--config", "core.filemode=false",
        "https://github.com/Panchovix/stable-diffusion-webui-reForge.git", WORKSPACE
    ])

os.chdir(WORKSPACE)

if UPDATE_FORGE:
    print("-= Updating SDForge =-")
    subprocess.run(["git", "pull"])

if INSTALL_DEPS:
    print("-= Installing dependencies =-")
    subprocess.run(["sudo", "apt-get", "update", "-y"])
    subprocess.run(["sudo", "apt", "--fix-broken", "install", "-y"])
    subprocess.run(["sudo", "apt", "-y", "install", "-qq", "aria2"])
    subprocess.run(["pip", "install", "setuptools==80", "pip==25"])
    # urllib.request.urlretrieve(
    #     "https://github.com/hariphoenix1708/link/raw/refs/heads/main/requirements.txt",
    #     "requirements.txt"
    # )
    subprocess.run(["pip", "cache", "purge"])
    subprocess.run(["pip", "install", "pickleshare"])
    subprocess.run(["pip", "install", "basicsr"])
    subprocess.run(["pip", "install", "insightface"])
    subprocess.run(["pip", "uninstall", "-y", "xformers", "torch", "torchvision", "torchaudio"])
    # urllib.request.urlretrieve(
    #     "https://github.com/hariphoenix1708/link/raw/refs/heads/main/requirements.txt",
    #     "requirements.txt"
    # )
    # subprocess.run(["uv", "cache", "clean"])
    # subprocess.run(["uv", "pip", "install", "--system", "-q", "-r", "requirements.txt"])


def get_filename_from_url(url):
    return os.path.basename(urlparse(url).path)


def get_latest_file(directory):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return max(files, key=lambda f: os.path.getctime(os.path.join(directory, f))) if files else None
    except Exception as e:
        print(f"Error finding latest file in {directory}: {e}")
        return None


def clean_line(line):
    # Strip inline annotations used in batch.txt
    return line.split('**')[0].strip()


def extract_civitai_model_id(url):
    """
    Supports:
      https://civitai.com/api/download/models/2837020?...
      https://civitai.com/models/2837020?...
    """
    match = re.search(r'/models/(\d+)', url)
    if match:
        return match.group(1)
    match = re.search(r'models/(\d+)', url)
    if match:
        return match.group(1)
    return None


def run_with_retry(func, *args, label="task"):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args)
        except Exception as e:
            last_error = e
            print(f"[Retry {attempt}/{MAX_RETRIES}] {label}: {e}")
            time.sleep(2 * attempt)
    raise last_error


def civitai_download_to_path(model_id, output_path, token):
    """
    Downloads a Civitai model via the API using Bearer auth and saves it
    to output_path. This is the embedded replacement for the separate
    download.py helper.
    """
    if not token:
        raise Exception("CIVITAI_TOKEN is not set. Please set it in the environment.")

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    }

    class NoRedirection(urllib.request.HTTPErrorProcessor):
        def http_response(self, request, response):
            return response
        https_response = http_response

    url = f"https://civitai.com/api/download/models/{model_id}"
    request = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(NoRedirection)
    response = opener.open(request)

    if response.status in [301, 302, 303, 307, 308]:
        redirect_url = response.getheader("Location")

        if not redirect_url:
            raise Exception("Civitai did not return a redirect URL")

        if redirect_url.startswith("/"):
            base_url = urlparse(url)
            redirect_url = f"{base_url.scheme}://{base_url.netloc}{redirect_url}"

        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        content_disposition = query_params.get("response-content-disposition", [None])[0]
        if content_disposition and "filename=" in content_disposition:
            filename = unquote(content_disposition.split("filename=")[1].strip('"'))
        else:
            filename = os.path.basename(parsed_url.path) or f"{model_id}.safetensors"

        response = urllib.request.urlopen(redirect_url)
    elif response.status == 404:
        raise Exception(f"Civitai file not found for model {model_id}")
    else:
        raise Exception(f"No redirect found for model {model_id}, status={response.status}")

    total_size = response.getheader("Content-Length")
    if total_size is not None:
        total_size = int(total_size)

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, filename)

    with open(output_file, "wb") as f:
        downloaded = 0
        start_time = time.time()
        speed = 0.0

        while True:
            chunk_start_time = time.time()
            buffer = response.read(1638400)
            chunk_end_time = time.time()

            if not buffer:
                break

            downloaded += len(buffer)
            f.write(buffer)
            chunk_time = chunk_end_time - chunk_start_time
            if chunk_time > 0:
                speed = len(buffer) / chunk_time / (1024 ** 2)

            if total_size is not None and total_size > 0:
                progress = downloaded / total_size
                sys.stdout.write(f"\rDownloading: {filename} [{progress*100:.2f}%] - {speed:.2f} MB/s")
                sys.stdout.flush()

    end_time = time.time()
    time_taken = end_time - start_time
    hours, remainder = divmod(time_taken, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        time_str = f"{int(minutes)}m {int(seconds)}s"
    else:
        time_str = f"{int(seconds)}s"

    sys.stdout.write("\n")
    latest_file = get_latest_file(output_path)
    if latest_file:
        print(f"Downloaded✅: {latest_file}")
    else:
        print(f"Downloaded✅: {filename}")
    print(f"Downloaded in {time_str}")

    if output_file.endswith(".zip"):
        print("Note: The downloaded file is a ZIP archive.")
        try:
            with zipfile.ZipFile(output_file, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(output_file))
        except Exception as e:
            print(f"ERROR: Failed to unzip the file. {e}")


def download_huggingface(command, filename):
    result = os.system(command)
    if result != 0:
        raise Exception(f"HuggingFace download failed: {filename}")
    print(f"Downloaded✅: {filename}")


def install_extension(repo, ext_path):
    if os.path.exists(ext_path):
        print(f"Installed✅: {repo.split('/')[-1]} (already exists)")
        return
    result = os.system(f'git clone "{repo}" "{ext_path}"')
    if result != 0:
        raise Exception(f"Extension clone failed: {repo}")
    print(f"Installed✅: {repo.split('/')[-1]}")


def get_optimal_workers():
    cpu_count = multiprocessing.cpu_count()
    return max(4, min(cpu_count * 2, 8))


def download_from_txt(txt_url):
    model_dirs = {
        "[]MODELS": "./models/Stable-diffusion",
        "[]VAE": "./models/VAE",
        "[]EMBEDDINGS": "./embeddings",
        "[]EMBEDDINGS-NEGATIVE": "./embeddings/Negative",
        "[]LORA": "./models/Lora",
        "[]EXTENSIONS": "./extensions"
    }

    current_dir = None
    civitai_jobs = []
    huggingface_jobs = []
    extension_jobs = []

    response = requests.get(txt_url)
    if response.status_code != 200:
        print("Failed to fetch the file.")
        return

    lines = response.text.strip().split("\n")

    for line in lines:
        line = clean_line(line)
        if not line:
            continue

        if line.startswith("[]") and line in model_dirs:
            current_dir = model_dirs[line]
            os.makedirs(current_dir, exist_ok=True)
            print(f"Using directory: {current_dir}")

        elif (line.startswith("$") or line.startswith("&")) and current_dir:
            # Keep the original $ logic, but also accept & because your batch.txt uses both.
            link = line[1:].strip()

            if "civitai.com" in link:
                model_id = extract_civitai_model_id(link)
                if model_id:
                    civitai_jobs.append((model_id, current_dir))
                else:
                    print(f"Skipping invalid Civitai URL: {link}")
            else:
                filename = get_filename_from_url(link)
                command = f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{link}" -d "{current_dir}" -o "{filename}"'
                huggingface_jobs.append((command, filename))

        elif line.startswith("@") and ALLOW_EXTENSION_INSTALLATION:
            repo = line[1:].strip()
            ext_path = f"{model_dirs['[]EXTENSIONS']}/{repo.split('/')[-1]}"
            extension_jobs.append((repo, ext_path))

    # Parallel HuggingFace downloads
    hf_workers = get_optimal_workers()
    print(f"Using {hf_workers} parallel workers for HuggingFace downloads")

    hf_futures = []
    with ThreadPoolExecutor(max_workers=hf_workers) as executor:
        for command, filename in huggingface_jobs:
            hf_futures.append(
                executor.submit(
                    run_with_retry,
                    download_huggingface,
                    command,
                    filename,
                    label=f"HF:{filename}"
                )
            )

        for future in as_completed(hf_futures):
            future.result()

    # Sequential Civitai downloads
    print("Using sequential downloads for Civitai")
    for model_id, target_dir in civitai_jobs:
        run_with_retry(
            civitai_download_to_path,
            model_id,
            target_dir,
            CIVITAI_TOKEN,
            label=f"Civitai:{model_id}"
        )

    # Extensions
    if ALLOW_EXTENSION_INSTALLATION and extension_jobs:
        print("Installing extensions")
        for repo, ext_path in extension_jobs:
            run_with_retry(install_extension, repo, ext_path, label=f"EXT:{repo.split('/')[-1]}")


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
LAUNCH_CMD = f"python launch.py {share} --api --listen --cuda-stream --cuda-malloc --pin-shared-memory --always-gpu --force-channels-last --attention-pytorch --enable-insecure-extension-access {auth} --disable-console-progressbars"
print(f"Starting WebUI with command:\n{LAUNCH_CMD}")
os.system(LAUNCH_CMD)
