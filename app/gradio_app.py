from __future__ import annotations
import gradio as gr
from claimlens.infer import ClaimLensPipeline

pipe = ClaimLensPipeline()

def run(question, passages):
    out=pipe.predict(question, passages or '')
    evidence='

'.join([f"{i+1}. score={e['score']:.3f}: {e['passage']}" for i,e in enumerate(out['evidence'])])
    return out['answer'], out['label'], round(out['confidence'],4), evidence

demo=gr.Interface(fn=run, inputs=[gr.Textbox(label='Question'), gr.Textbox(label='Passages', lines=8)], outputs=[gr.Textbox(label='Answer'), gr.Textbox(label='Support Label'), gr.Number(label='Confidence'), gr.Textbox(label='Evidence')], title='ClaimLens', description='Evidence-grounded answering and claim verification demo.')

if __name__ == '__main__': demo.launch(server_name='0.0.0.0', server_port=7860)
