from __future__ import annotations
import torch
from torch import nn

class ClaimLensModel(nn.Module):
    """Cross-encoder style classifier over question, evidence, and candidate answer embeddings."""
    def __init__(self, embedding_dim:int=384, hidden_dim:int=256, dropout:float=0.15, num_labels:int=3):
        super().__init__()
        combined = embedding_dim * 4
        self.question_gate = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.evidence_gate = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.fusion = nn.Sequential(
            nn.Linear(combined, hidden_dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim)
        )
        self.support_head = nn.Linear(hidden_dim, num_labels)
        self.answerability_head = nn.Linear(hidden_dim, 1)
        self.calibration_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Linear(hidden_dim//2, 1))

    def forward(self, q, e):
        diff = torch.abs(q-e); prod = q*e
        x = torch.cat([q, e, diff, prod], dim=-1)
        h = self.fusion(x)
        return {
            'support_logits': self.support_head(h),
            'answerability_logit': self.answerability_head(h).squeeze(-1),
            'confidence_logit': self.calibration_head(h).squeeze(-1)
        }
