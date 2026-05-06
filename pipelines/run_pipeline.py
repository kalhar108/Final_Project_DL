from __future__ import annotations
import argparse
from claimlens.data import build_dataset
from claimlens.train import train
from claimlens.evaluate import evaluate

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config', default='configs/config.yaml')
    args=ap.parse_args()
    build_dataset(args.config); train(args.config); print(evaluate(args.config))
