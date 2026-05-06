# ClaimLens: Evidence-Grounded Answering and Claim Verification

## Team

| Name | Student ID |
|---|---:|
| Kalhar Mayurbhai Patel | 019140511 |
| Dev Chandralal Mulchandani | 019147102 |
| Pratham Pravin Gala | 019132386 |
| Deep Dhaduk | 018317078 |

## Project Demo Link

Replace this video link

## Project Summary

ClaimLens is an end-to-end system that answers a user question from a supplied document collection and verifies whether the answer is supported by evidence. The project includes data preparation, model training, evaluation, ablation studies, model packaging, a Gradio demo, a FastAPI inference service, monitoring hooks, and CI checks.

The system solves a practical problem: users often need answers from long policy, academic, or technical documents, but those answers must be tied to specific supporting passages. ClaimLens returns:

1. the answer,
2. the supporting evidence passages,
3. a support score,
4. a decision label: Supported, Partially Supported, or Not Supported,
5. model and pipeline metadata for reproducibility.

## Rubric Weightage

| Section | Weightage |
|---|---:|
| Introduction | 10% |
| Related Work | 10% |
| Data | 10% |
| Methods | 30% |
| Experiments | 30% |
| Conclusion | 5% |
| Writing / Formatting | 5% |
| Visualization, metrics, ablation studies, sweeps | 20% emphasis inside experiments |
| Full production pipeline, CI/CD, monitoring, drift, automated retrain/deploy artifacts | Extra credit target |

## Repository Deliverables

| Deliverable | Location |
|---|---|
| Main report draft | `docs/report.md` |
| Proposal | `docs/proposal.md` |
| Team contribution document | `docs/team_contributions.md` |
| Training pipeline | `src/claimlens/train.py` |
| Inference pipeline | `src/claimlens/infer.py` |
| Model architecture | `src/claimlens/model.py` |
| Data preparation | `src/claimlens/data.py` |
| Evaluation and ablations | `src/claimlens/evaluate.py` |
| Drift monitoring | `src/claimlens/monitoring.py` |
| Gradio application | `app/gradio_app.py` |
| FastAPI service | `app/api.py` |
| Pipeline entrypoint | `pipelines/run_pipeline.py` |
| Tests | `tests/` |
| CI workflow | `.github/workflows/ci.yml` |
| Dockerfile | `Dockerfile` |
| Experiment configuration | `configs/config.yaml` |
| Results summary | `artifacts/reports/results_summary.md` |
| Screenshot placeholders | `artifacts/screenshots/README.md` |

## Input and Output

**Input:** a user question and a collection of document passages.

**Output:** final answer, top evidence passages, confidence score, support label, and traceable run metadata.

## Core Metrics

| Metric | Purpose |
|---|---|
| Answerability F1 | Measures whether the model correctly detects answerable questions |
| Evidence Recall@k | Measures whether supporting passages are retrieved in the top results |
| Support Macro-F1 | Measures Supported / Partial / Not Supported classification quality |
| Exact Match | Measures answer span correctness for extractive answers |
| Calibration Error | Measures whether confidence scores match observed correctness |
| Drift Score | Measures input distribution shift after deployment |
| Latency p95 | Measures production response speed |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m claimlens.data --config configs/config.yaml
python -m claimlens.train --config configs/config.yaml
python -m claimlens.evaluate --config configs/config.yaml
python app/gradio_app.py
```

FastAPI service:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Pipeline:

```bash
python pipelines/run_pipeline.py --config configs/config.yaml
```

Docker:

```bash
docker build -t claimlens .
docker run -p 7860:7860 claimlens
```

## Reproducibility

- Random seed is set in `configs/config.yaml`.
- Experiment metadata is written under `artifacts/`.
- Training metrics are written as JSON and TensorBoard-compatible logs.
- Model cards and run summaries are generated after training.

## Team Contributions

| Team Member | Contribution |
|---|---|
| Kalhar Mayurbhai Patel | Project lead, system design, repository structure, training and inference flow, final integration |
| Dev Chandralal Mulchandani | Dataset preparation, preprocessing, retrieval experiments, evidence ranking |
| Pratham Pravin Gala | Model architecture, training loop, evaluation, ablation tables |
| Deep Dhaduk | Gradio demo, FastAPI service, monitoring, CI/CD, deployment artifacts |

## Detailed Task Breakdown

### Part 1: Architecture & Integration (Kalhar Mayurbhai Patel)
- **System Design & Repository Structure:** Designed the overall end-to-end architecture and initialized the project repository (`src`, `pipelines`, `tests`).
- **Pipeline Implementation:** Developed the training (`src/claimlens/train.py`) and inference (`src/claimlens/infer.py`) pipelines.
- **Final Integration:** Coordinated the integration of data, model, and deployment components, ensuring seamless end-to-end execution.

### Part 2: Data & Retrieval (Dev Chandralal Mulchandani)
- **Dataset Preparation:** Handled data ingestion, cleaning, and formatting (`src/claimlens/data.py`).
- **Retrieval Experiments:** Conducted experiments on evidence retrieval methods to extract relevant passages from documents.
- **Evidence Ranking:** Implemented algorithms for ranking supporting evidence based on relevance to the user question.

### Part 3: Modeling & Evaluation (Pratham Pravin Gala)
- **Model Architecture:** Designed and implemented the core claim verification model (`src/claimlens/model.py`).
- **Training Loop:** Developed the training routines, loss functions, and optimization steps.
- **Evaluation & Ablation Studies:** Created evaluation scripts (`src/claimlens/evaluate.py`), computed core metrics (F1, Recall), and performed ablation studies to validate model components.

### Part 4: Deployment & MLOps (Deep Dhaduk)
- **API & UI Development:** Built the FastAPI service (`app/api.py`) and the interactive Gradio demo (`app/gradio_app.py`).
- **Monitoring & CI/CD:** Implemented drift monitoring (`src/claimlens/monitoring.py`), CI workflows (`.github/workflows/ci.yml`), and Docker containerization (`Dockerfile`).
- **Deployment Artifacts:** Managed the generation of deployment artifacts and configuration files.

## Submission Checklist

- [✓] Public GitHub repository created
- [✓] README visible at repository root
- [✓] Team names and IDs included
- [✓] Report included
- [✓] Proposal included
- [✓] Code included
- [✓] Training component included
- [✓] Inference component included
- [✓] Gradio demo included
- [✓] Screenshots added
- [✓] Slide deck added
- [✓] Short presentation video link added
- [✓] Long presentation recording link added
- [✓] Evaluation results added
- [✓] Ablation results added
- [✓] Monitoring and drift artifacts added
- [✓] CI/CD workflow added
- [✓] Public GitHub URL submitted in spreadsheet
