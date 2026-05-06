from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from claimlens.infer import ClaimLensPipeline

app=FastAPI(title='ClaimLens API')
pipe=ClaimLensPipeline()
class Request(BaseModel):
    question:str
    passages:str=''
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/predict')
def predict(req:Request): return pipe.predict(req.question, req.passages)
