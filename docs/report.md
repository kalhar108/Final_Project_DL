# ClaimLens: Evidence-Grounded Answering and Claim Verification

## Title and Authors
ClaimLens: Evidence-Grounded Answering and Claim Verification

Kalhar Mayurbhai Patel (019140511), Dev Chandralal Mulchandani (019147102), Pratham Pravin Gala (019132386), Deep Dhaduk (018317078)

## Abstract
ClaimLens is a document-question answering and claim verification system. Given a question and a set of passages, it retrieves evidence, predicts whether the answer is supported, partially supported, or not supported, and returns a confidence score. The system includes data preparation, model training, evaluation, ablation studies, inference service, application interface, and monitoring artifacts. Results are reported with support classification metrics, retrieval metrics, calibration metrics, and production readiness metrics.

## Introduction (10%)
Many document-answering systems return fluent answers without making evidence easy to inspect. This creates risk in academic, policy, compliance, and technical settings where users must verify the source of each answer. ClaimLens addresses this by pairing answer generation with evidence retrieval and support classification.

## Related Work (10%)
The project builds on extractive question answering, neural retrieval, natural language inference, and confidence calibration. Traditional retrieval methods provide traceability but limited reasoning. Sequence classifiers can judge support but need strong evidence selection. ClaimLens combines retrieval, support prediction, and calibrated confidence in a single runnable pipeline.

## Data (10%)
The dataset consists of question, passage, answer, and support-label records. The initial seed data is included in the repository so the pipeline is runnable. The structure supports expansion with manually curated course, policy, technical, or benchmark passages. Preprocessing includes record validation, deterministic splits, label mapping, and data profiling.

## Methods (30%)
The system has four main components: passage retrieval, embedding generation, support classification, and calibrated scoring. The classifier receives question and evidence embeddings and uses feature fusion over raw embeddings, absolute difference, and elementwise product. The model outputs support label logits, answerability logits, and confidence logits.

## Experiments (30%)
Experiments include classification metrics, retrieval recall, ablations, hyperparameter sweeps, latency measurement, and drift analysis. The ablation plan compares top-k retrieval settings, calibration removal, and fusion removal. Visualizations are generated under `artifacts/reports/`.

## Conclusion (5%)
ClaimLens provides a complete evidence-grounded answering pipeline with training, inference, evaluation, interface, and production artifacts. Future extensions include larger curated datasets, deployment on managed cloud infrastructure, and scheduled retraining after monitored drift.

## Writing / Formatting (5%)
The final submission should be exported as a 6-8 page PDF with figures, tables, and appendix links.
