import os 
import yaml
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "config.yaml"


with open(CFG_PATH, 'r') as file:
    config = yaml.safe_load(file)

if not isinstance(config, dict):
    raise TypeError(f"{CFG_PATH} must be a YAML mapping, got {type(config)}: {config!r}")
if "pytorch_hub" not in config or "weights_dir" not in config["pytorch_hub"]:
    raise KeyError("Missing 'pytorch_hub.weights_dir' in config.yaml")

raw = config["pytorch_hub"]["weights_dir"]  
p = Path(os.path.expandvars(raw)).expanduser()
weights_dir = (p if p.is_absolute() else (CFG_PATH.parent / p)).resolve()

os.environ["TORCH_HOME"] = str(weights_dir)