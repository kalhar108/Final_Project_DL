from __future__ import annotations
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from .model import ClaimLensModel
from .retrieval import TfidfRetriever
from .utils import load_yaml, read_jsonl

LABELS=["Supported","Partially Supported","Not Supported"]

class ClaimLensPipeline:
    def __init__(self, config_path='configs/config.yaml'):
        self.cfg=load_yaml(config_path)
        self.encoder=SentenceTransformer(self.cfg['model']['encoder_name'])
        self.retriever=TfidfRetriever()
        rows=[]
        for split in ['train','valid','test']:
            rows += read_jsonl(Path(self.cfg['paths']['processed_dir'])/f'{split}.jsonl')
        self.rows=rows
        self.retriever.fit([r['context'] for r in rows] or ['No context loaded'])
        self.model=ClaimLensModel(384,self.cfg['model']['hidden_dim'],self.cfg['model']['dropout'],self.cfg['model']['num_support_labels'])
        ckpt=Path(self.cfg['paths']['model_dir'])/'claimlens.pt'
        if ckpt.exists(): self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()
    def _enc(self, text):
        return torch.tensor(self.encoder.encode([text], normalize_embeddings=True), dtype=torch.float32)
    def predict(self, question:str, passages:str=''):
        corpus=[p.strip() for p in passages.split('
') if p.strip()] if passages.strip() else [r['context'] for r in self.rows]
        self.retriever.fit(corpus or ['No passage supplied'])
        evidence=self.retriever.search(question, k=self.cfg['retrieval']['top_k'])
        top=evidence[0]['passage'] if evidence else ''
        with torch.no_grad():
            out=self.model(self._enc(question), self._enc(top))
            probs=torch.softmax(out['support_logits'], dim=-1).squeeze(0)
            idx=int(torch.argmax(probs))
        answer=top if idx != 2 else 'The supplied passages do not provide enough support.'
        return {"answer":answer,"label":LABELS[idx],"confidence":float(probs[idx]),"evidence":evidence}
