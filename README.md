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

ClaimLens is a robust, end-to-end Machine Learning system designed to answer complex user questions from a supplied collection of documents while simultaneously verifying whether the extracted answer is grounded in factual evidence. By synthesizing Information Retrieval (IR) and Natural Language Processing (NLP) techniques, ClaimLens ensures high-fidelity claim verification. The project encompasses the entire ML lifecycle: data preparation, deep neural network training, rigorous evaluation, ablation studies, model packaging, a rich Gradio user interface, a highly-available FastAPI inference service, production monitoring hooks, and Continuous Integration (CI) workflows.

The system addresses a critical challenge in modern AI: hallucination and unverified generation. Users often require answers from lengthy, dense materials—such as policy documents, academic journals, or technical manuals—but need the assurance that these answers are intrinsically tied to specific, verifiable supporting passages. 

To solve this, ClaimLens reliably outputs:
1. **The Final Answer:** A concise response to the user's inquiry.
2. **Supporting Evidence Passages:** The exact sentences or paragraphs extracted from the source documents that back up the claim.
3. **Support Confidence Score:** A calibrated probability score reflecting the model's certainty.
4. **Decision Label:** A categorical classification indicating whether the claim is *Supported*, *Partially Supported*, or *Not Supported*.
5. **Traceability Metadata:** Extensive model and pipeline metadata ensuring full experiment reproducibility and auditability.

### System Architecture & Pipeline Flow
- **Data Ingestion & Preprocessing (`src/claimlens/data.py`):** Handles raw JSONL datasets, deduplicates passages via cryptographic hashing, and splits data cleanly into train/valid/test sets.
- **Evidence Retrieval (`src/claimlens/retrieval.py`):** Uses optimized TF-IDF heuristics combined with semantic search to isolate the `top_k` most relevant passages from the corpus.
- **Deep Verification Model (`src/claimlens/model.py`):** A custom neural architecture that employs `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) to encode questions and evidence, followed by dense hidden layers with dropout to classify the support label.
- **Evaluation & Ablation (`src/claimlens/evaluate.py`):** Computes robust metrics such as Macro-F1, Precision, and Recall. Ablation scripts strip away components (like cross-attention or calibration) to quantify their impact on the baseline metrics.
- **Deployment & MLOps (`src/claimlens/monitoring.py`, `app/`):** The model is wrapped in a scalable FastAPI backend and monitored by Prometheus for data drift. A responsive Gradio frontend serves as the primary user touchpoint. Docker guarantees environment consistency across deployments.

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

## Detailed Repository Structure & File Descriptions

### 📁 `src/claimlens/` - Core Machine Learning Source Code
- **`data.py`**: Manages the data ingestion pipeline. It reads raw JSONL files, performs deduplication using SHA-256 cryptographic hashing to avoid data leakage, and systematically partitions the corpus into `train`, `valid`, and `test` sets for rigorous model evaluation.
- **`model.py`**: Contains the PyTorch neural network architecture (`ClaimLensModel`). It leverages `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) to generate dense contextual embeddings of both the question and the retrieved evidence, passing them through a multi-layer perceptron (MLP) with dropout regularization to output a 3-way support classification.
- **`retrieval.py`**: Implements the retrieval heuristic (`TfidfRetriever`). It indexes the document passages and uses TF-IDF vectorization combined with nearest-neighbors search to efficiently extract the top-K most relevant passages for any given query.
- **`train.py`**: The training loop orchestrator. It manages the PyTorch dataloaders, computes the cross-entropy loss, applies the AdamW optimizer with weight decay, tracks training/validation metrics per epoch, and handles optimal model checkpointing.
- **`evaluate.py`**: Evaluates the trained model on hold-out test sets. It computes extensive evaluation metrics including Answerability F1, Evidence Recall, Support Macro-F1, and Calibration Error. It also sequentially executes ablation configurations (e.g., removing cross-attention) defined in the config file to validate architecture decisions.
- **`infer.py`**: Wraps the trained model and retriever into a cohesive, production-ready `ClaimLensPipeline` class. This provides a clean interface (`pipe.predict()`) used by the API and frontend to generate end-to-end answers in real-time.
- **`monitoring.py`**: Includes essential hooks for production MLOps. It tracks data distributions, inference latency percentiles (p95), and input drift scores, exposing these critical metrics for Prometheus scraping.
- **`utils.py`**: A comprehensive collection of robust helper functions for setting global random seeds (to ensure absolute reproducibility), and safely parsing/writing JSONL and YAML files.

### 📁 `app/` - User Interfaces & APIs
- **`gradio_app.py`**: The primary user touchpoint. A rich, interactive web application built with Gradio Blocks. It allows users to input custom questions and documents, returning a sleek dashboard containing the generated answer, a color-coded support label, a confidence score, and cleanly formatted markdown evidence.
- **`api.py`**: A highly-scalable FastAPI service that wraps the inference pipeline. It exposes REST endpoints (`/predict`) for programmatic access, enabling seamless integration into larger microservice architectures and handling concurrent requests efficiently.

### 📁 `pipelines/` & `configs/` - Orchestration
- **`run_pipeline.py`**: A centralized workflow entrypoint script that sequentially triggers data preparation, training, and evaluation in one single automated command.
- **`config.yaml`**: The single source of truth for all experiment configurations. It defines model hyperparameters (batch size, learning rate), directory paths, model architecture dimensions, and explicit definitions for the ablation studies.

### 📁 Infrastructure, Tests, & Documentation
- **`Dockerfile`**: Containerization instructions to securely package the FastAPI and Gradio applications along with all underlying dependencies into a perfectly reproducible Docker image.
- **`.github/workflows/ci.yml`**: Continuous Integration (CI) pipeline instructions for GitHub Actions. It runs automated syntax checks and smoke tests on every push to guarantee absolute code stability.
- **`tests/test_smoke.py`**: Automated unit and integration tests to verify the pipeline's core functionality before deployment.
- **`docs/report.md` & `docs/proposal.md`**: The comprehensive academic report and initial project proposal extensively outlining the methodologies, related work, experimental setup, and analytical findings.

## Dataset & Methodology

### Data Sourcing and Preparation
The performance of any claim verification system is heavily bounded by the quality of its underlying dataset. For ClaimLens, we prioritized high-fidelity text that closely mimics real-world policy, academic, and technical documents. 
- **Cryptographic Deduplication:** We compute SHA-256 hashes for every combination of `question` and `context`. Any colliding hashes are strictly dropped from the corpus, guaranteeing that no identical passages leak between the train and test sets.
- **Stratified Splitting:** The final dataset is carefully split into `train` (80%), `valid` (10%), and `test` (10%) partitions. We apply class-aware stratification to ensure that the distribution of our three target labels (`Supported`, `Partially Supported`, and `Not Supported`) remains perfectly consistent across all three splits, preventing catastrophic forgetting of minority classes during gradient descent.

### Model Architecture and Optimization Strategy
Rather than training a monolithic Large Language Model (LLM) from scratch, ClaimLens employs a highly optimized, parameter-efficient pipeline designed for low-latency production environments:
- **Dual-Encoder Strategy:** We utilize lightweight `sentence-transformers` models (specifically `all-MiniLM-L6-v2`) which map sentences to a 384-dimensional dense vector space. This allows our system to capture deep semantic similarity far better than traditional bag-of-words or standard BM25 approaches.
- **Multi-Layer Perceptron (MLP) Classifier:** The dense embeddings for the user's question and the highest-ranked retrieved evidence are concatenated and passed through a custom PyTorch Multi-Layer Perceptron. This classifier uses a hidden dimension of `256` with a Dropout probability of `p=0.15` to heavily penalize over-reliance on specific features and prevent overfitting.
- **AdamW Optimization:** We train the MLP using the AdamW optimizer with a conservative learning rate of `3e-5` and a weight decay of `0.01`. This decoupled weight decay setup aggressively penalizes large weights, ensuring the model generalizes well to entirely unseen out-of-distribution documents.

## Ablation Studies Setup

To rigorously prove the necessity of each component in our architecture, we defined systematic ablation studies executed automatically via our `evaluate.py` script:
1. **`no_cross_attention`:** Tests the system's performance if we completely remove the interaction layer between the question embedding and the evidence embedding. **Hypothesis:** Support Macro-F1 will drop significantly because the model cannot properly contextualize the evidence against the specific constraints of the question.
2. **`no_calibration`:** Removes the temperature-scaling calibration step applied to the final Softmax logits. **Hypothesis:** Core accuracy remains unchanged, but Calibration Error spikes, meaning the model's reported confidence scores become mathematically untrustworthy for downstream decision-making.
3. **`top_k_3` vs `top_k_8`:** Adjusts the number of passages retrieved by the TF-IDF module. **Hypothesis:** A lower K increases system throughput/speed but drops Evidence Recall, while a higher K improves Recall but introduces noisy text that may confuse the MLP classifier and degrade the Support Macro-F1 score.

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
pip install -e .
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
