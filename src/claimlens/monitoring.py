from __future__ import annotations
import math
from collections import Counter

def token_distribution(texts):
    c=Counter()
    for t in texts: c.update(t.lower().split())
    total=sum(c.values()) or 1
    return {k:v/total for k,v in c.items()}

def jensen_shannon(p,q):
    keys=set(p)|set(q)
    m={k:(p.get(k,0)+q.get(k,0))/2 for k in keys}
    def kl(a,b): return sum(a.get(k,0)*math.log((a.get(k,1e-12)+1e-12)/(b.get(k,1e-12)+1e-12)) for k in keys)
    return 0.5*kl(p,m)+0.5*kl(q,m)

def drift_report(reference_texts, live_texts, threshold=0.18):
    score=jensen_shannon(token_distribution(reference_texts), token_distribution(live_texts))
    return {"drift_score":score,"threshold":threshold,"action":"retrain_review" if score>threshold else "continue_monitoring"}
