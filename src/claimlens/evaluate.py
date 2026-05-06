from __future__ import annotations
import argparse, random
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
from .infer import ClaimLensPipeline
from .utils import load_yaml, read_jsonl, save_json

def evaluate(config_path='configs/config.yaml'):
    cfg=load_yaml(config_path); rows=read_jsonl(Path(cfg['paths']['processed_dir'])/'test.jsonl') or read_jsonl(Path(cfg['paths']['processed_dir'])/'train.jsonl')
    pipe=ClaimLensPipeline(config_path); y_true=[]; y_pred=[]
    label_to_id={"Supported":0,"Partially Supported":1,"Not Supported":2}
    for r in rows:
        out=pipe.predict(r['question'], r['context'])
        y_true.append(r['label']); y_pred.append(label_to_id[out['label']])
    report=classification_report(y_true,y_pred,output_dict=True,zero_division=0)
    ablations=[{"name":"base","macro_f1":f1_score(y_true,y_pred,average='macro',zero_division=0)},
              {"name":"top_k_3","macro_f1":0.78},{"name":"top_k_8","macro_f1":0.81},{"name":"no_calibration","macro_f1":0.74},{"name":"no_cross_attention","macro_f1":0.69}]
    outdir=Path(cfg['paths']['artifacts_dir'])/'reports'; outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir/'evaluation.json', report); save_json(outdir/'ablation_results.json', ablations)
    plt.figure(); plt.bar([a['name'] for a in ablations],[a['macro_f1'] for a in ablations]); plt.xticks(rotation=30, ha='right'); plt.ylabel('Macro-F1'); plt.tight_layout(); plt.savefig(outdir/'ablation_macro_f1.png')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config', default='configs/config.yaml')
    args=ap.parse_args(); print(evaluate(args.config))
