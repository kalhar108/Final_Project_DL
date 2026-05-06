from __future__ import annotations
import json, random, os
from pathlib import Path
import numpy as np

def set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)

def read_jsonl(path:str|Path):
    p=Path(path)
    if not p.exists(): return []
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path:str|Path, rows)->None:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        for row in rows: f.write(json.dumps(row)+"
")

def load_yaml(path:str|Path):
    import yaml
    with open(path) as f: return yaml.safe_load(f)

def save_json(path:str|Path, obj)->None:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))
