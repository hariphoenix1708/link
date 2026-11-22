
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

# ---- app integration (v9) -------------------------------------------------
import json
import datetime
import threading


def _session_id():
    m = re.search(r"-(\d+)-webtier", os.environ.get("KAGGLE_CONTAINER_NAME", ""))
    return m.group(1) if m else "UNKNOWN"


print("SESSION_ID=" + _session_id(), flush=True)   # the app reads this to be able to Stop

# Interactive Kaggle sessions set Jupyter-only variables that break the separate Python 3.11 environment
os.environ["MPLBACKEND"] = "Agg"          # was 'matplotlib_inline' (Jupyter only) -> matplotlib crashed on import
os.environ.pop("PYTHONPATH", None)        # don't let Python 3.13 packages leak into the 3.11 environment
os.environ["PYTHONNOUSERSITE"] = "1"


def phase(name, pct):
    """Progress line the Android app parses from the live log."""
    print(f"@@PHASE {pct} {name}", flush=True)


phase("Starting", 1)
# ---------------------------------------------------------------------------

# ---- Python 3.11 environment (Kaggle's image moved to Python 3.13, which most of
# ---- the pinned packages in requirements.txt do not support) --------------------
import shutil

PY_VERSION = "3.11"


def _pick_scratch():
    """/kaggle/working is size-limited and holds the models; build the big environment elsewhere."""
    best, best_free = None, -1
    for d in ("/tmp", "/kaggle/temp"):
        try:
            os.makedirs(d + "/sdb", exist_ok=True)
            free = shutil.disk_usage(d).free
            if free > best_free:
                best, best_free = d + "/sdb", free
        except Exception:
            pass
    return best or "/tmp/sdb", max(best_free, 0)


SCRATCH, _SCRATCH_FREE = _pick_scratch()
VENV_DIR = SCRATCH + "/venv311"
VENV_PY = VENV_DIR + "/bin/python"
os.environ["UV_CACHE_DIR"] = SCRATCH + "/uv-cache"   # same disk as the venv -> hardlinks, no doubling
os.environ["UV_LINK_MODE"] = "hardlink"
print(f"@@DISK scratch={SCRATCH} free={_SCRATCH_FREE / 1e9:.0f}GB", flush=True)


def _df(tag):
    print(f"@@DISK {tag}", flush=True)
    subprocess.run(["df", "-h", "/kaggle/working", SCRATCH])


def _crash_hook(exc_type, exc, tb):
    import traceback
    print("@@ERROR crash: " + "".join(traceback.format_exception_only(exc_type, exc)).strip()[:300], flush=True)
    traceback.print_exception(exc_type, exc, tb)
    _df("at crash")


sys.excepthook = _crash_hook


def _abort(msg):
    print(f"@@ERROR {msg}", flush=True)
    sys.exit(1)


def _kaggle_pkg_version(pkg):
    """Base version of a package Kaggle already ships (e.g. torch '2.8.0+cu126' -> '2.8.0')."""
    try:
        import importlib.metadata as md
        return md.version(pkg).split("+")[0]
    except Exception:
        return None


def setup_python_env():
    phase("Creating Python 3.11 environment", 24)
    _df("before environment")
    if _SCRATCH_FREE < 15e9:
        print("@@WARN less than 15 GB free in the scratch area; the install may run out of space", flush=True)
    for cmd in (["uv", "python", "install", PY_VERSION],
                ["uv", "venv", "--seed", "--python", PY_VERSION, "--clear", VENV_DIR]):
        if subprocess.run(cmd).returncode != 0:
            _abort("could not create the Python 3.11 environment: " + " ".join(cmd[:3]))

    lines = [l.strip() for l in open("requirements.txt").read().splitlines() if l.strip()]
    optional = [l for l in lines if l.lower().startswith("insightface")]   # needs compiling; not essential
    core = [l for l in lines if l not in optional]

    torch_v, tv_v = _kaggle_pkg_version("torch"), _kaggle_pkg_version("torchvision")

    def write_req(name, pin_torch):
        out = []
        for l in core:
            if l.lower() == "torch" and pin_torch and torch_v:
                l = f"torch=={torch_v}"      # same torch Kaggle ships = the combination that worked before
            out.append(l)
        if pin_torch and tv_v:
            out.append(f"torchvision=={tv_v}")
        elif "torchvision" not in "".join(out).lower():
            out.append("torchvision")
        open(name, "w").write("\n".join(out) + "\n")

    phase("Installing dependencies", 28)
    ok = False
    for pin in (True, False):                # second try: let uv choose the torch version
        write_req("requirements-venv.txt", pin)
        if subprocess.run(["uv", "pip", "install", "--python", VENV_PY, "-r", "requirements-venv.txt"]).returncode == 0:
            ok = True
            break
        print("@@WARN dependency install failed" + (", retrying without the torch pin" if pin else ""), flush=True)
    if not ok:
        _abort("dependency install failed in the Python 3.11 environment (see log above)")
    open("constraints-venv.txt", "w").write("protobuf==4.25.6\n")     # insightface/onnx otherwise upgrade it, breaking mediapipe
    for pkg in optional:
        r = subprocess.run(["uv", "pip", "install", "--python", VENV_PY, "-c", "constraints-venv.txt", pkg])
        if r.returncode != 0:
            print(f"@@WARN optional package {pkg} failed to install (continuing)", flush=True)
    chk = subprocess.run([VENV_PY, "-c", "import mediapipe, google.protobuf as p; print('mediapipe OK, protobuf', p.__version__)"])
    if chk.returncode != 0:
        print("@@WARN mediapipe does not import in the new environment (ADetailer's mediapipe detectors may fail)", flush=True)
    shutil.rmtree(os.environ["UV_CACHE_DIR"], ignore_errors=True)   # frees several GB; venv files stay (hardlinks)
    _df("after environment")
    # reForge's own pip calls (launch.py) run inside this environment
    subprocess.run([VENV_PY, "-c", "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"])

# Hugging Face large-file downloads are faster when Xet support is enabled.
# Do not disable Xet here; Kaggle can use the native fast path when hf_xet is available.
os.environ.pop("HF_HUB_DISABLE_XET", None)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

def _ensure_hf_stack():
    try:
        import hf_xet  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "hf_xet"])
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[hf_xet]"])
    from huggingface_hub import hf_hub_download
    return hf_hub_download

hf_hub_download = _ensure_hf_stack()
print("Hugging Face Xet backend ready")

# Install pyngrok if not already installed
try:
    from pyngrok import ngrok, conf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
    from pyngrok import ngrok, conf

# Secrets: private dataset (works for notebooks pushed via API) -> Kaggle Secrets -> env.
# Nothing is hardcoded here. Values are never printed.
_SECRET_KEYS = ["CIVITAI_TOKEN", "NGROK_TOKEN", "NGROK_DOMAIN", "APP_USER", "APP_PASS", "IDLE_MINUTES", "MAX_HOURS"]


def _load_secrets():
    vals = {}
    import glob
    # Batch runs mount at /kaggle/input/sd-secrets/, interactive ones at /kaggle/input/datasets/<owner>/sd-secrets/
    cands = (glob.glob("/kaggle/input/**/sd-secrets/secrets.json", recursive=True)
             or glob.glob("/kaggle/input/**/secrets.json", recursive=True))
    f = cands[0] if cands else ""
    if f and os.path.exists(f):
        try:
            vals.update({k: str(v).strip() for k, v in json.load(open(f)).items() if v})
        except Exception as e:
            print("WARN: could not read secrets dataset:", type(e).__name__)
    for k in _SECRET_KEYS:
        if vals.get(k):
            continue
        v = ""
        try:
            from kaggle_secrets import UserSecretsClient
            v = UserSecretsClient().get_secret(k) or ""
        except Exception:
            v = os.getenv(k, "")
        if v:
            vals[k] = v.strip()
    return vals


SECRETS = _load_secrets()
CIVITAI_TOKEN = SECRETS.get("CIVITAI_TOKEN", "")
print("Secrets found:", ", ".join(k for k in _SECRET_KEYS if SECRETS.get(k)) or "none", flush=True)

_missing = [k for k in ("CIVITAI_TOKEN", "NGROK_TOKEN", "NGROK_DOMAIN") if not SECRETS.get(k)]
if _missing:
    try:
        import glob as _g
        _attached = sorted(_g.glob("/kaggle/input/*") + _g.glob("/kaggle/input/*/*") + _g.glob("/kaggle/input/*/*/*"))[:12]
    except Exception:
        _attached = []
    _abort("missing secrets: " + ", ".join(_missing) + ". Datasets attached to this session: "
           + (", ".join(_attached) or "none") + ". Attach the private dataset 'sd-secrets' (editor > Input > Add Input).")

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

phase("Setting up reForge", 8)
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
    phase("Updating reForge", 14)
    print("-= Updating SDForge =-")
    subprocess.run(["git", "pull"])

if INSTALL_DEPS:
    phase("Installing dependencies", 22)
    print("-= Installing dependencies =-")
    subprocess.run(["apt-get", "update", "-y"])
    subprocess.run(["apt", "--fix-broken", "install", "-y"])
    subprocess.run(["apt", "-y", "install", "-qq", "aria2"])
    urllib.request.urlretrieve(
        "https://github.com/hariphoenix1708/link/raw/refs/heads/main/requirements.txt",
        "requirements.txt",
    )
    subprocess.run(["uv", "cache", "clean"])
    setup_python_env()   # replaces the old system-wide requirements install
    # Restore fast Hugging Face transfers for Xet-backed repos.
    # Fast install with uv (preferred)
    subprocess.run(
        ["uv", "pip", "install", "--system", "-q", "-U", "huggingface_hub[hf_xet]", "hf_xet"],
        check=False,
    )

# Fallback to pip
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub[hf_xet]", "hf_xet"],
        check=False,
    )
    #subprocess.run(["uv", "pip", "install", "--system", "-q", "hf_xet"], check=False)
    #subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf_xet"], check=False)


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
        except FileNotFoundError:
            # A 404 will not fix itself on retry.
            raise
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
    def _first_request(u):
        return requests.get(u, headers=headers, allow_redirects=False, timeout=120)

    resp = _first_request(clean_url)

    # If the type/format/size/fp filters don't match any file, Civitai answers 404.
    # Retry with the bare model-version URL (or keep ?fileId= if present).
    if resp.status_code == 404 and "?" in clean_url:
        base_url, query = clean_url.split("?", 1)
        file_id = parse_qs(query).get("fileId")
        bare_url = f"{base_url}?fileId={file_id[0]}" if file_id else base_url
        if bare_url != clean_url:
            print(f"404 with filters, retrying without type/format: {bare_url}")
            retry_resp = _first_request(bare_url)
            if retry_resp.status_code in (200, 301, 302, 303, 307, 308):
                resp, clean_url = retry_resp, bare_url

    if resp.status_code in (301, 302, 303, 307, 308):
        redirect_url = resp.headers.get("Location")
        if not redirect_url:
            raise RuntimeError(f"Civitai redirect missing Location header: {clean_url}")
        redirect_url = urljoin(clean_url, redirect_url)
    elif resp.status_code == 200:
        # Some endpoints may return content directly; use the current URL.
        redirect_url = clean_url
    elif resp.status_code == 404:
        raise FileNotFoundError(f"Civitai 404 (not found / filters don't match): {clean_url}")
    else:
        raise RuntimeError(
            f"Civitai download request failed ({resp.status_code}): {clean_url}"
        )

    # Second request: signed Cloudflare/R2 URL.
    # Do NOT send the Civitai bearer token here because the signed URL
    # already contains authentication query parameters.
    download_headers = {
        "User-Agent": USER_AGENT,
        "Referer": "https://civitai.com/",
        "Accept": "*/*",
    }

    try:
        r = requests.get(
            redirect_url,
            headers=download_headers,
            stream=True,
            timeout=300,
        )
        r.raise_for_status()
    except requests.HTTPError as first_error:
        # Some signed URLs reject extra headers entirely.
        # Retry once with minimal request metadata.
        try:
            r = requests.get(
                redirect_url,
                stream=True,
                timeout=300,
            )
            r.raise_for_status()
        except Exception:
            raise first_error

    with r:

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



def parse_hf_url(url: str):
    """
    Parse a Hugging Face file URL such as:
      https://huggingface.co/org/model/resolve/main/file.safetensors
    Returns (repo_id, revision, filename) or None when the URL does not match.
    """
    match = re.search(
        r"https?://huggingface\.co/(?P<repo_id>.+?)/(?:resolve|blob)/(?P<revision>[^/]+)/(?P<filename>.+)$",
        url,
    )
    if not match:
        return None

    return (
        match.group("repo_id"),
        match.group("revision"),
        unquote(match.group("filename")),
    )


def download_huggingface(source: str, destination: str, token: str = None):
    if source.startswith("http://") or source.startswith("https://"):
        parsed = parse_hf_url(source)
        if not parsed:
            raise RuntimeError(f"Unsupported Hugging Face URL: {source}")

        repo_id, revision, filename = parsed
        os.makedirs(destination, exist_ok=True)

        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=token or None,
                local_dir=destination,
                local_dir_use_symlinks=False,
                resume_download=True,
                etag_timeout=60,
            )
            print(f"Downloaded✅: {os.path.basename(local_path)}")
            return local_path
        except Exception as e:
            raise RuntimeError(
                f"HF hub download failed for {repo_id}/{filename} @ {revision}: {e}"
            ) from e

    result = os.system(source)
    if result != 0:
        raise RuntimeError(f"HF download failed: {destination}")
    print(f"Downloaded✅: {destination}")


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
                if "huggingface.co" in link or "hf.co" in link:
                    huggingface_jobs.append((link, current_dir))
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

    failed = []

    # Parallel HuggingFace downloads
    if huggingface_jobs:
        print(f"Using {HF_WORKERS} parallel workers for HuggingFace downloads")
        with ThreadPoolExecutor(max_workers=HF_WORKERS) as executor:
            future_map = {}
            for item in huggingface_jobs:
                source, destination = item
                label = f"HF:{os.path.basename(urlparse(source).path) or destination}"
                fut = executor.submit(run_with_retry, download_huggingface, source, destination, token, label=label)
                future_map[fut] = label
            for future in as_completed(future_map):
                try:
                    future.result()
                except Exception as e:
                    failed.append((future_map[future], str(e)))
                    print(f"❌ Skipped {future_map[future]}: {e}")

    # Sequential Civitai downloads
    if civitai_jobs:
        print("Using sequential downloads for Civitai")
        for download_url, output_dir in civitai_jobs:
            label = f"Civitai:{os.path.basename(urlparse(download_url).path) or 'download'}"
            try:
                run_with_retry(
                    civitai_download_to_path,
                    download_url,
                    output_dir,
                    token,
                    label=label,
                )
            except Exception as e:
                failed.append((label, str(e)))
                print(f"❌ Skipped {label}: {e}")

    # Extensions
    if ALLOW_EXTENSION_INSTALLATION and extension_jobs:
        for repo, ext_path in extension_jobs:
            try:
                run_with_retry(clone_extension, repo, ext_path, label=f"Git:{repo.split('/')[-1]}")
            except Exception as e:
                failed.append((f"Git:{repo.split('/')[-1]}", str(e)))
                print(f"❌ Skipped Git:{repo.split('/')[-1]}: {e}")

    if failed:
        print("\n===== Failed downloads =====")
        for label, err in failed:
            print(f" - {label}: {err}")
        print("============================\n")


# Execute download step
phase("Downloading models and extensions", 40)
txt_file_url = "https://raw.githubusercontent.com/hariphoenix1708/link/refs/heads/main/batch.txt"
download_from_txt(txt_file_url)

phase("Finishing setup", 88)
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


# ===== NGROK + LAUNCH + WATCHDOG (managed by the Android app) =====
phase("Opening tunnel", 92)

NGROK_AUTHTOKEN = SECRETS.get("NGROK_TOKEN", "")
Ngrok_domain = SECRETS.get("NGROK_DOMAIN", "").replace("https://", "").replace("http://", "").strip("/ ")
APP_USER = SECRETS.get("APP_USER", "")
APP_PASS = SECRETS.get("APP_PASS", "")
IDLE_MINUTES = int(SECRETS.get("IDLE_MINUTES", "30"))     # auto-shutdown after this long with no activity
MAX_HOURS = float(SECRETS.get("MAX_HOURS", "8"))           # hard safety cap on session length
READY_TIMEOUT_MIN = 25                                     # give up if the UI never comes up
ALLOW_UI_EXTENSION_INSTALL = False                         # only matters when a login is set


def _fail(msg):
    print(f"@@ERROR {msg}", flush=True)
    try:
        ngrok.kill()
    except Exception:
        pass
    sys.exit(1)


if not (NGROK_AUTHTOKEN and Ngrok_domain):
    _fail("NGROK_TOKEN / NGROK_DOMAIN missing (secrets dataset not attached?)")

try:
    ngrok.kill()
except Exception:
    pass

pyngrok_cfg = conf.PyngrokConfig(auth_token=NGROK_AUTHTOKEN)
tunnel_started, last_err = False, ""
# Retries matter: right after a Stop, the old session may still hold the free domain for a moment.
for attempt in range(1, 7):
    for kwargs in ({"url": f"https://{Ngrok_domain}"}, {"bind_tls": True, "domain": Ngrok_domain}):
        try:
            ngrok.connect(7860, pyngrok_config=pyngrok_cfg, **kwargs)
            tunnel_started = True
            break
        except Exception as e:
            last_err = str(e).replace(NGROK_AUTHTOKEN, "***")
    if tunnel_started:
        break
    print(f"@@WARN tunnel attempt {attempt}/6 failed: {last_err[:200]}", flush=True)
    time.sleep(10)
if not tunnel_started:
    _fail("ngrok tunnel failed: " + last_err[:200])   # no --share fallback: the app could not find it
print(f"NGROK Link OK: https://{Ngrok_domain}", flush=True)

# ---- launch ----
cmd = [VENV_PY if os.path.exists(VENV_PY) else "python", "launch.py", "--api", "--listen", "--cuda-stream", "--disable-console-progressbars", "--no-hashing"]
if APP_USER and APP_PASS:
    cred = f"{APP_USER}:{APP_PASS}"          # avoid ',' and ':' inside the user name / password
    cmd += ["--api-auth", cred, "--gradio-auth", cred]
    if ALLOW_UI_EXTENSION_INSTALL:
        cmd += ["--enable-insecure-extension-access"]
    print("Login protection: ON", flush=True)
else:
    print("@@WARN backend is UNPROTECTED: add APP_USER and APP_PASS to the secrets dataset", flush=True)
shown = [("<hidden>" if (APP_PASS and APP_PASS in c) else c) for c in cmd]
print("Starting WebUI with command:", " ".join(shown), flush=True)
_df("before launch")
phase("Starting reForge", 95)
proc = subprocess.Popen(cmd)
t_start = time.time()
state = {"ready": False, "t_ready": 0.0}


def _ready_watch():
    while proc.poll() is None and not state["ready"]:
        try:
            r = requests.get("http://127.0.0.1:7860/internal/ping", timeout=3)
            if r.status_code < 500:          # any answer (even 401) means the server is up
                state["ready"], state["t_ready"] = True, time.time()
                phase("Ready", 100)
                print("@@READY", flush=True)
                return
        except Exception:
            pass
        time.sleep(5)


threading.Thread(target=_ready_watch, daemon=True).start()


def _ts(s):
    s = re.sub(r"(\.\d{6})\d+", r"\1", s).replace("Z", "+00:00")
    return datetime.datetime.fromisoformat(s)


def _api_base():
    try:
        return ngrok.get_ngrok_process(pyngrok_cfg).api_url.rstrip("/")
    except Exception:
        return "http://127.0.0.1:4040"


def _seconds_since_last_request():
    """Age of the newest non-GET request seen by the local ngrok inspector (None = unavailable).
    Gradio UI actions and every generation are POSTs. The app's keep-alive is a POST to /internal/ping."""
    try:
        data = requests.get(_api_base() + "/api/requests/http?limit=60", timeout=5).json()
        newest = None
        for r in data.get("requests", []):
            if r.get("request", {}).get("method", "GET") != "GET":
                ts = _ts(r["start"])
                newest = ts if newest is None or ts > newest else newest
        if newest is None:
            return float("inf")
        return (datetime.datetime.now(datetime.timezone.utc) - newest).total_seconds()
    except Exception:
        return None


def _gpu_busy():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return max(int(x) for x in out.split()) > 15
    except Exception:
        return False


def _shutdown(reason):
    print(f"@@SHUTDOWN {reason}", flush=True)
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        ngrok.kill()                          # frees the free domain for the next start
    except Exception:
        pass
    os._exit(0)                               # ends the notebook run -> stops the quota clock


limit = IDLE_MINUTES * 60
last_active = time.time()
warned_inspector = False
tick = 0
while proc.poll() is None:
    time.sleep(30)
    tick += 1
    now = time.time()
    up = now - t_start
    if not state["ready"]:
        if up > READY_TIMEOUT_MIN * 60:
            _shutdown("backend never became ready")
        continue
    last_active = max(last_active, state["t_ready"])
    age = _seconds_since_last_request()
    if age is None and not warned_inspector:
        print("@@WARN ngrok inspector unavailable: idle detection uses GPU activity only", flush=True)
        warned_inspector = True
    if age is not None:
        last_active = max(last_active, now - age)
    if _gpu_busy():
        last_active = now
    idle = now - last_active
    if tick % 2 == 0:
        print(f"@@STATUS idle={int(idle)} limit={limit} up={int(up)}", flush=True)
    if idle >= limit:
        _shutdown(f"idle for {int(idle // 60)} min")
    if up >= MAX_HOURS * 3600:
        _shutdown(f"max session length {MAX_HOURS} h reached")

print(f"@@EXITED reForge stopped on its own (code {proc.returncode})", flush=True)
try:
    ngrok.kill()
except Exception:
    pass
