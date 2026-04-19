
import os
import re
import time
import sys
import subprocess
import multiprocessing
import urllib.request
import zipfile
from urllib.parse import urlparse, urljoin, unquote, parse_qs

import requests

# Install pyngrok if not already installed
try:
    from pyngrok import ngrok, conf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
    from pyngrok import ngrok, conf

# Kaggle secret (preferred) with environment fallback
CIVITAI_TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    secret_label = "CIVITAI_TOKEN"
    secret_value = UserSecretsClient().get_secret(secret_label)
    if secret_value:
        CIVITAI_TOKEN = secret_value.strip()
except Exception:
    CIVITAI_TOKEN = os.getenv("CIVITAI_TOKEN", "").strip()

# CONFIGURATION
USE_GOOGLE_DRIVE = False
UPDATE_FORGE = True
INSTALL_DEPS = True
ALLOW_EXTENSION_INSTALLATION = True
USE_USERNAME_AND_PASSWORD = False
USERNAME = ""
PASSWORD = ""

WORKSPACE = "stable-diffusion-webui-forge"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
MAX_RETRIES = 3
HF_WORKERS = max(4, min(multiprocessing.cpu_count() * 2, 8))

# Skip Google Drive setup for local script

# Clone repo if not exists
if not os.path.exists(WORKSPACE):
    print("-= Initial setup SDForge =-")
    subprocess.run(
        [
            "git",
            "clone",
            "--config",
            "core.filemode=false",
            "https://github.com/Panchovix/stable-diffusion-webui-reForge.git",
            WORKSPACE,
        ]
    )

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
        "requirements.txt",
    )
    subprocess.run(["uv", "cache", "clean"])
    subprocess.run(["uv", "pip", "install", "--system", "-q", "-r", "requirements.txt"])


def get_filename_from_url(url: str) -> str:
    return os.path.basename(urlparse(url).path)


def get_latest_file(directory: str):
    try:
        files = [
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
        return max(files, key=lambda f: os.path.getctime(os.path.join(directory, f))) if files else None
    except Exception as e:
        print(f"Error finding latest file in {directory}: {e}")
        return None


def clean_line(line: str) -> str:
    return line.split("**")[0].strip()


def extract_civitai_model_id(url: str):
    match = re.search(r"models/(\d+)", url)
    return match.group(1) if match else None


def clean_civitai_download_url(url: str) -> str:
    # Keep the full model download query string (type/format/size/fp),
    # but strip any token that may already be attached.
    url = re.sub(r"([?&])token=[^&]+", r"\1", url)
    url = url.replace("?&", "?").rstrip("&?")
    return url


def run_with_retry(func, *args, label=None, retries=MAX_RETRIES, delay=2):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args)
        except Exception as e:
            last_error = e
            tag = label or func.__name__
            print(f"[Retry {attempt}/{retries}] {tag}: {e}")
            time.sleep(delay * attempt)
    if last_error:
        raise last_error


def parse_filename_from_content_disposition(content_disposition: str):
    if not content_disposition:
        return None

    # filename*=UTF-8''...
    match = re.search(r"filename\*\s*=\s*([^']*)''([^;]+)", content_disposition, flags=re.IGNORECASE)
    if match:
        return unquote(match.group(2).strip().strip('"'))

    # filename="..."
    match = re.search(r'filename\s*=\s*"?([^";]+)"?', content_disposition, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def parse_filename_from_url_or_query(url: str):
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)

    if filename:
        return filename

    qs = parse_qs(parsed.query)
    for key in ("b2ContentDisposition", "response-content-disposition"):
        if key in qs and qs[key]:
            parsed_name = parse_filename_from_content_disposition(qs[key][0])
            if parsed_name:
                return parsed_name

    return "downloaded_file"


def ensure_unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 1
    while True:
        candidate = f"{base} ({counter}){ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def civitai_download_to_path(download_url: str, output_dir: str, token: str):
    if not token:
        raise RuntimeError(
            "CIVITAI_TOKEN is missing. Add it in Kaggle Secrets with the label CIVITAI_TOKEN."
        )

    os.makedirs(output_dir, exist_ok=True)

    clean_url = clean_civitai_download_url(download_url)

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
        "Referer": "https://civitai.com/",
        "Accept": "*/*",
    }

    # First request: obtain redirect URL while preserving the full query string
    resp = requests.get(clean_url, headers=headers, allow_redirects=False, timeout=120)

    if resp.status_code in (301, 302, 303, 307, 308):
        redirect_url = resp.headers.get("Location")
        if not redirect_url:
            raise RuntimeError(f"Civitai redirect missing Location header: {clean_url}")
        redirect_url = urljoin(clean_url, redirect_url)
    elif resp.status_code == 200:
        # Some endpoints may return content directly; use the current URL.
        redirect_url = clean_url
    else:
        raise RuntimeError(
            f"Civitai download request failed ({resp.status_code}): {clean_url}"
        )

    # Second request: signed file download with the same headers
    with requests.get(redirect_url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()

        filename = (
            parse_filename_from_content_disposition(r.headers.get("Content-Disposition"))
            or parse_filename_from_url_or_query(redirect_url)
            or os.path.basename(urlparse(clean_url).path)
            or "downloaded_file"
        )

        output_path = ensure_unique_path(os.path.join(output_dir, filename))
        tmp_path = output_path + ".part"
        total = int(r.headers.get("Content-Length") or 0)
        downloaded = 0
        start_time = time.time()

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                elapsed = max(time.time() - start_time, 1e-6)
                speed = downloaded / elapsed / (1024 ** 2)

                if total > 0:
                    pct = downloaded * 100 / total
                    print(
                        f"\rDownloading: {filename} [{pct:.2f}%] - {speed:.2f} MB/s",
                        end="",
                        flush=True,
                    )

        os.replace(tmp_path, output_path)
        print()

        elapsed_total = time.time() - start_time
        mins, secs = divmod(int(elapsed_total), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            time_str = f"{hours}h {mins}m {secs}s"
        elif mins:
            time_str = f"{mins}m {secs}s"
        else:
            time_str = f"{secs}s"

        print(f"Downloaded✅: {os.path.basename(output_path)}")
        print(f"Downloaded in {time_str}")


def download_huggingface(command: str, filename: str):
    result = os.system(command)
    if result != 0:
        raise RuntimeError(f"HF download failed: {filename}")
    print(f"Downloaded✅: {filename}")


def clone_extension(repo: str, ext_path: str):
    if os.path.exists(ext_path):
        print(f"Skipped✅: {repo.split('/')[-1]} (already exists)")
        return
    result = os.system(f'git clone "{repo}" "{ext_path}"')
    if result != 0:
        raise RuntimeError(f"Extension clone failed: {repo}")
    print(f"Installed✅: {repo.split('/')[-1]}")


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


def download_from_txt(txt_url, token=CIVITAI_TOKEN):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    model_dirs = {
        "[]MODELS": "./models/Stable-diffusion",
        "[]VAE": "./models/VAE",
        "[]EMBEDDINGS": "./embeddings",
        "[]EMBEDDINGS-NEGATIVE": "./embeddings/Negative",
        "[]LORA": "./models/Lora",
        "[]EXTENSIONS": "./extensions",
    }

    current_dir = None
    civitai_jobs = []
    huggingface_jobs = []
    extension_jobs = []

    response = requests.get(txt_url, timeout=120)
    if response.status_code != 200:
        print("Failed to fetch the file.")
        return

    lines = response.text.strip().split("\n")

    for raw_line in lines:
        line = clean_line(raw_line)
        if not line:
            continue

        if line.startswith("[]") and line in model_dirs:
            current_dir = model_dirs[line]
            os.makedirs(current_dir, exist_ok=True)
            print(f"Using directory: {current_dir}")
            continue

        # Only process $ for downloads. Ignore all other prefixes, including &.
        if current_dir and line.startswith("$"):
            link = line[1:].strip()

            if "civitai.com" in link:
                civitai_jobs.append((link, current_dir))
            else:
                filename = get_filename_from_url(link) or "downloaded_file"
                command = (
                    f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M '
                    f'"{link}" -d "{current_dir}" -o "{filename}"'
                )
                huggingface_jobs.append((command, filename))
            continue

        if ALLOW_EXTENSION_INSTALLATION and line.startswith("@"):
            repo = line[1:].strip()
            if repo:
                ext_path = f"{model_dirs['[]EXTENSIONS']}/{repo.split('/')[-1]}"
                extension_jobs.append((repo, ext_path))

    # Parallel HuggingFace downloads
    if huggingface_jobs:
        print(f"Using {HF_WORKERS} parallel workers for HuggingFace downloads")
        with ThreadPoolExecutor(max_workers=HF_WORKERS) as executor:
            futures = [
                executor.submit(run_with_retry, download_huggingface, cmd, fname, label=f"HF:{fname}")
                for cmd, fname in huggingface_jobs
            ]
            for future in as_completed(futures):
                future.result()

    # Sequential Civitai downloads
    if civitai_jobs:
        print("Using sequential downloads for Civitai")
        for download_url, output_dir in civitai_jobs:
            label = f"Civitai:{os.path.basename(urlparse(download_url).path) or 'download'}"
            run_with_retry(
                civitai_download_to_path,
                download_url,
                output_dir,
                token,
                label=label,
            )

    # Extensions
    if ALLOW_EXTENSION_INSTALLATION and extension_jobs:
        for repo, ext_path in extension_jobs:
            run_with_retry(clone_extension, repo, ext_path, label=f"Git:{repo.split('/')[-1]}")


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
    "ui-config.json",
)

# Extract Lora zip if it exists
lora_zip_path = "./models/Lora/details.zip"
lora_extract_dir = "./models/Lora"

if os.path.exists(lora_zip_path):
    try:
        print(f"Extracting Lora models from {lora_zip_path}...")
        with zipfile.ZipFile(lora_zip_path, "r") as zip_ref:
            zip_ref.extractall(lora_extract_dir)
        print("Extraction complete ✅")
        os.remove(lora_zip_path)
        print("Cleaned up zip file 🗑️")
    except zipfile.BadZipFile:
        print("Error: Bad ZIP file!")
else:
    print(f"No zip file found at {lora_zip_path}")


# NGROK CONFIG
COMIC = True  # hk1708
MIDGE = False  # vk872

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
            domain=Ngrok_domain,
        ).public_url
        print(f"NGROK Link ✅: https://{Ngrok_domain}")
    except Exception as e:
        print(f"NGROK Error: {e}")
        share = "--share"
else:
    share = "--share"

# Launch WebUI FOR NORMAL FORGE
# LAUNCH_CMD = f"python launch.py {share} --gpu-device-id 1 --xformers --cuda-stream --always-gpu --pin-shared-memory --api --listen --enable-insecure-extension-access {auth} --disable-console-progressbars --no-hashing --precision autocast --upcast-sampling"
# FOR RE-FROGE
LAUNCH_CMD = f"python launch.py {share} --api --listen --cuda-stream --enable-insecure-extension-access {auth} --disable-console-progressbars --no-hashing"
print(f"Starting WebUI with command:\n{LAUNCH_CMD}")
os.system(LAUNCH_CMD)
