# Package IDS: 
# 1 -> gtFine_trainvaltest.zip (241MB)
# 2 -> gtCoarse.zip (1.3GB)
# 3 -> leftImg8bit_trainvaltest.zip (11GB)
# 4 -> leftImg8bit_trainextra.zip (44GB)
# 8 -> camera_trainvaltest.zip (2MB)
# 9 -> camera_trainextra.zip (8MB)
# 10 -> vehicle_trainvaltest.zip (2MB)
# 11 -> vehicle_trainextra.zip (7MB)
# 12 -> leftImg8bit_demoVideo.zip (6.6GB)
# 28 -> gtBbox_cityPersons_trainval.zip (2.2MB)


# Run: python download_cityscapes.py 1 3 8
import os, sys, re, subprocess, tempfile, shutil
from pathlib import Path
import yaml
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PACKAGE_FILES = {
    1:  "gtFine_trainvaltest.zip",
    2:  "gtCoarse.zip",
    3:  "leftImg8bit_trainvaltest.zip",
    4:  "leftImg8bit_trainextra.zip",
    8:  "camera_trainvaltest.zip",
    9:  "camera_trainextra.zip",
    10: "vehicle_trainvaltest.zip",
    11: "vehicle_trainextra.zip",
    12: "leftImg8bit_demoVideo.zip",
    28: "gtBbox_cityPersons_trainval.zip",
}

ROOT = Path(__file__).resolve().parents[1]          
CFG_PATH = ROOT / "config.yaml"

with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f) or {}

raw_dir = cfg.get("cityscapes", {}).get("data_dir", "data/cityscapes")
data_dir = Path(os.path.expanduser(os.path.expandvars(str(raw_dir))))
if not data_dir.is_absolute():
    data_dir = (ROOT / data_dir).resolve()
data_dir.mkdir(parents=True, exist_ok=True)

USERNAME = os.getenv("CITYSCAPES_USERNAME")
PASSWORD = os.getenv("CITYSCAPES_PASSWORD")
if not USERNAME or not PASSWORD:
    raise SystemExit("Set CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD in your .env")

LOGIN_URL = "https://www.cityscapes-dataset.com/login/"
BASE_URL  = "https://www.cityscapes-dataset.com/file-handling/?packageID="

PACKAGE_IDS = [1, 3]
if len(sys.argv) > 1:
    PACKAGE_IDS = [int(x) for x in sys.argv[1:]]

def curl_login(cookie_path: str) -> None:
    subprocess.run(
        [
            "curl", "--fail", "--location", "--show-error",
            "--cookie-jar", cookie_path,
            "--data-urlencode", f"username={USERNAME}",
            "--data-urlencode", f"password={PASSWORD}",
            "--data", "submit=Login",
            LOGIN_URL,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

def curl_headers(cookie_path: str, url: str) -> str:
    r = subprocess.run(
        ["curl", "-sIL", "-b", cookie_path, "-L", url],
        check=True, capture_output=True, text=True
    )
    return r.stdout

def parse_filename(headers: str, pkg_id: int) -> str:
    m = re.search(r'(?i)^content-disposition:.*?filename\*=[^=]*=\s*"?([^";\r\n]+)"?', headers, re.M)
    if not m:
        m = re.search(r'(?i)^content-disposition:.*?filename="?([^";\r\n]+)"?', headers, re.M)
    name = m.group(1) if m else f"{PACKAGE_FILES[pkg_id]}"
    return name.replace("/", "_")

def parse_content_length(headers: str) -> int | None:
    m_all = re.findall(r'(?i)^content-length:\s*([0-9]+)\s*$', headers, re.M)
    return int(m_all[-1]) if m_all else None

def filesize(p: Path) -> int:
    try:
        return p.stat().st_size
    except FileNotFoundError:
        return 0

def curl_download(cookie_path: str, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"→ Downloading {url}\n   → {dest}")
    subprocess.run(
        [
            "curl", "--fail", "--location", "--show-error", "--progress-bar",
            "-b", cookie_path,
            "-C", "-",              
            "-o", str(dest),        
            url
        ],
        check=True,
    )
    print()  

def extract_all_archives(dest_dir: Path) -> None:
    for p in sorted(dest_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith((".zip", ".tar", ".tgz", ".tar.gz")):
            try:
                print(f"→ Extracting {p.name} ...")
                shutil.unpack_archive(str(p), extract_dir=str(dest_dir))
            except shutil.ReadError:
                print(f"  ! Skipping {p.name}: not a supported archive")
    print("Extraction complete.")

cookie = tempfile.NamedTemporaryFile(delete=False)
cookie_path = cookie.name
cookie.close()

try:
    curl_login(cookie_path)

    for pkg_id in PACKAGE_IDS:
        url = f"{BASE_URL}{pkg_id}"
        headers = curl_headers(cookie_path, url)
        filename = parse_filename(headers, pkg_id)
        expected = parse_content_length(headers)
        out_path = data_dir / filename

        if expected is not None and out_path.exists() and filesize(out_path) == expected:
            print(f"Already present: {out_path} ({expected} bytes). Skipping.")
            continue

        curl_download(cookie_path, url, out_path)

    print(f"Downloads finished in: {data_dir}")

finally:
    try:
        os.remove(cookie_path)
    except FileNotFoundError:
        pass

# Extract after downloads
extract_all_archives(data_dir)

# delete the zip files
for p in data_dir.iterdir():
    if p.is_file() and p.name.lower().endswith((".zip", ".tar", ".tgz", ".tar.gz")):
        print(f"→ Deleting {p.name} ...")
        try:
            p.unlink()
        except OSError as e:
            print(f"  ! Error deleting {p.name}: {e}")
