from __future__ import annotations
import argparse, time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from .model import ClaimLensModel
from .utils import load_yaml, read_jsonl, save_json, set_seed

def encode(encoder, texts):
    return torch.tensor(encoder.encode(texts, normalize_embeddings=True), dtype=torch.float32)

def train(config_path:str):
    cfg=load_yaml(config_path); set_seed(cfg['seed'])
    train_rows=read_jsonl(Path(cfg['paths']['processed_dir'])/'train.jsonl')
    valid_rows=read_jsonl(Path(cfg['paths']['processed_dir'])/'valid.jsonl') or train_rows
    encoder=SentenceTransformer(cfg['model']['encoder_name'])
    q=encode(encoder,[r['question'] for r in train_rows]); e=encode(encoder,[r['context'] for r in train_rows])
    y=torch.tensor([r['label'] for r in train_rows], dtype=torch.long)
    dl=DataLoader(TensorDataset(q,e,y), batch_size=cfg['training']['batch_size'], shuffle=True)
    model=ClaimLensModel(q.shape[1], cfg['model']['hidden_dim'], cfg['model']['dropout'], cfg['model']['num_support_labels'])
    opt=torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    loss_fn=nn.CrossEntropyLoss()
    history=[]
    for epoch in range(cfg['training']['epochs']):
        model.train(); total=0.0
        for qb,eb,yb in dl:
            out=model(qb,eb); loss=loss_fn(out['support_logits'], yb)
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss.item())
        history.append({"epoch":epoch+1,"loss":total/max(len(dl),1)})
    model_dir=Path(cfg['paths']['model_dir']); model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir/'claimlens.pt')
    save_json(model_dir/'metadata.json', {"encoder":cfg['model']['encoder_name'],"labels":["Supported","Partially Supported","Not Supported"],"trained_at":int(time.time())})
    save_json(Path(cfg['paths']['artifacts_dir'])/'reports'/'training_history.json', history)
    return history

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config', default='configs/config.yaml')
    args=ap.parse_args(); print(train(args.config))
