from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    def __init__(self):
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2), min_df=1)
        self.passages=[]; self.matrix=None
    def fit(self, passages):
        self.passages=list(passages); self.matrix=self.vectorizer.fit_transform(self.passages); return self
    def search(self, query, k=5):
        if self.matrix is None or len(self.passages)==0: return []
        q=self.vectorizer.transform([query]); sims=cosine_similarity(q,self.matrix).ravel()
        idx=np.argsort(-sims)[:k]
        return [{"passage":self.passages[i],"score":float(sims[i])} for i in idx]
