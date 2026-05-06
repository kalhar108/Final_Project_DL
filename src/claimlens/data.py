from __future__ import annotations
import argparse, hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split
from .utils import load_yaml, write_jsonl, read_jsonl, save_json, set_seed

SEED_ROWS = [
 {"question":"What must a final project repository include?","context":"A final project repository must include code artifacts, a report, screenshots, slide deck, a demo, and a README that organizes links to all deliverables.","answer":"code artifacts, report, screenshots, slide deck, demo, and README","label":0},
 {"question":"Is a notebook-only submission acceptable?","context":"A notebook-only submission is not acceptable because the project must include a runnable training and inference pipeline with a demo application.","answer":"No","label":0},
 {"question":"What is needed for the experiment section?","context":"The experiment section should include core metrics, ablation studies, hyperparameter sweeps, visualizations, and analysis of failure modes.","answer":"metrics, ablations, sweeps, visualizations, and failure modes","label":0},
 {"question":"Does the system need a user interface?","context":"A Gradio interface is the minimum user experience expected for demonstrating model inference.","answer":"Yes, at least a Gradio interface","label":0},
 {"question":"Are team contributions required?","context":"The report and README should clearly list team members and each member's contribution.","answer":"Yes","label":0},
 {"question":"Can the project use reinforcement learning?","context":"The project instructions say not to use reinforcement learning.","answer":"No","label":0},
 {"question":"What data sources are allowed?","context":"The dataset may be manually curated, synthetically generated, annotated using a labeling tool, or based on benchmark datasets with original training and evaluation work.","answer":"curated, generated, annotated, or benchmark datasets","label":1},
 {"question":"What is the exact final grade?","context":"The document lists rubric categories and extra-credit targets, but the final grade is assigned by the instructor after review.","answer":"Not directly specified","label":2}
]

def fingerprint(row):
    return hashlib.sha256((row['question']+row['context']).encode()).hexdigest()[:12]

def build_dataset(config_path:str):
    cfg=load_yaml(config_path); set_seed(cfg['seed'])
    raw=Path(cfg['paths']['raw_data'])
    if not raw.exists():
        rows=[]
        for r in SEED_ROWS:
            r=dict(r); r['id']=fingerprint(r); rows.append(r)
        write_jsonl(raw, rows)
    rows=read_jsonl(raw)
    train, temp = train_test_split(rows, test_size=cfg['training']['validation_split'], random_state=cfg['seed'])
    valid, test = train_test_split(temp, test_size=0.5, random_state=cfg['seed']) if len(temp)>1 else (temp, temp)
    out=Path(cfg['paths']['processed_dir']); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out/'train.jsonl', train); write_jsonl(out/'valid.jsonl', valid); write_jsonl(out/'test.jsonl', test)
    save_json(Path(cfg['paths']['artifacts_dir'])/'reports'/'data_profile.json', {"train":len(train),"valid":len(valid),"test":len(test),"labels":{"0":"Supported","1":"Partially Supported","2":"Not Supported"}})
    return rows

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config', default='configs/config.yaml')
    args=ap.parse_args(); print(f"Prepared {len(build_dataset(args.config))} records")
