from __future__ import annotations
import gradio as gr
from claimlens.infer import ClaimLensPipeline

pipe = ClaimLensPipeline()

def run(question, passages):
    out = pipe.predict(question, passages or '')
    
    # Color code the label
    label_color = "#10b981" if out['label'] == "Supported" else "#f59e0b" if out['label'] == "Partially Supported" else "#ef4444"
    label_html = f"<div style='padding: 12px; border-radius: 8px; background-color: {label_color}; color: white; font-weight: bold; text-align: center; font-size: 1.2em; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>{out['label']}</div>"
    
    # Format evidence nicely
    evidence_md = "### Supporting Evidence\n\n"
    if not out['evidence']:
        evidence_md += "*No supporting evidence found.*\n"
    else:
        for i, e in enumerate(out['evidence']):
            evidence_md += f"**{i+1}.** *(Score: {e['score']:.3f})* {e['passage']}\n\n---\n\n"
        
    return out['answer'], label_html, f"{out['confidence']:.2%}", evidence_md

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

body {
    font-family: 'Inter', sans-serif !important;
    background-color: #f8fafc;
}
.header-container {
    text-align: center;
    padding: 3rem 1rem;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}
.header-container h1 {
    margin: 0;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: white;
}
.header-container p {
    font-size: 1.25rem;
    opacity: 0.9;
    margin-top: 0.75rem;
    font-weight: 400;
}
"""

with gr.Blocks() as demo:
    gr.HTML("""
    <div class="header-container">
        <h1>🔍 ClaimLens</h1>
        <p>Evidence-Grounded Answering and Claim Verification</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Parameters")
            question_input = gr.Textbox(
                label="Question", 
                placeholder="e.g., What must a final project repository include?", 
                lines=2
            )
            passages_input = gr.Textbox(
                label="Document Passages (Optional)", 
                placeholder="Paste the source document passages here, separated by newlines...\nLeave blank to search over the default dataset.", 
                lines=8
            )
            submit_btn = gr.Button("Analyze Claim", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### Analysis Results")
            answer_output = gr.Textbox(label="Generated Answer", lines=3, interactive=False)
            
            with gr.Row():
                with gr.Column(scale=2):
                    label_output = gr.HTML(label="Support Label")
                with gr.Column(scale=1):
                    confidence_output = gr.Textbox(label="Confidence Score", interactive=False)
                    
            evidence_output = gr.Markdown(label="Evidence")

    submit_btn.click(
        fn=run,
        inputs=[question_input, passages_input],
        outputs=[answer_output, label_output, confidence_output, evidence_output]
    )

if __name__ == '__main__': 
    demo.launch(server_name='0.0.0.0', server_port=7860, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=custom_css)
