# LegalAnalyzer.ua - Core ML Engine

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Llama.cpp](https://img.shields.io/badge/llama.cpp-GGUF-orange.svg)
![Status](https://img.shields.io/badge/status-active_development-success.svg)

## Overview
**LegalAnalyzer.ua** is a local, offline-first machine learning solution designed to automatically summarize complex Ukrainian legal documents (laws, contracts, court decrees) into concise, single-sentence abstracts. 

This repository contains the core Natural Language Processing (NLP) engine, including data preparation pipelines, inference scripts for quantized Large Language Models (LLMs), and automated evaluation modules.

The primary engineering goal of this project is to provide high-quality abstractive summarization while guaranteeing **100% data privacy** (offline execution) and strictly mitigating AI hallucinations (factual inaccuracies) common in legal tech applications.

## Key Features
* **Absolute Confidentiality:** Runs entirely offline to protect attorney-client privilege and comply with data protection regulations. No API calls to third-party services (e.g., OpenAI, Anthropic).
* **Hardware Optimization:** Utilizes 4-bit quantization (GGUF format) via `llama.cpp`, compressing a 7B parameter model to ~4.3 GB, enabling fast inference on standard consumer CPU/GPUs.
* **High Semantic Accuracy:** The underlying model (Mistral-7B architecture) is adapted for the Ukrainian legal domain using Parameter-Efficient Fine-Tuning (LoRA).
* **Hallucination Mitigation:** Implements a strict stateless inference architecture and directive prompt engineering to prevent parametric hallucinations and context pollution.

## Tech Stack
* **Language:** Python
* **LLM Frameworks:** Hugging Face (Transformers, PEFT, Datasets)
* **Inference:** `llama-cpp-python`
* **Evaluation Metrics:** `evaluate`, `rouge-score`, `bert_score`

## Repository Structure (Initial)
* `requirements.txt` — Project dependencies.
* `data_preparation.py` — Pipeline for cleaning and formatting raw legal texts into instruction-based JSONL datasets for Fine-Tuning.
* `llama_cpp_inference.py` — The main Singleton-based class for loading the GGUF model and generating summaries.
* `evaluate_metrics.py` *(WIP)* — Scripts for calculating ROUGE and BERTScore metrics to validate semantic consistency.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/ekatsiashvili/Capstone_Project_DS-DA.git](https://github.com/ekatsiashvili/Capstone_Project_DS-DA.git)
   cd Capstone_Project_DS-DA.git # LegalAnalyzer.ua - Core ML Engine
   ```
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Download the quantized model weights (.gguf file) and place them in the models/ directory. (Note: Model weights are not tracked in Git due to size limits).
   
4. Run the inference test:

   ```bash
   python llama_cpp_inference.py
   ```

## Evaluation and Testing
The system's performance is heavily tested against a human-written "Gold Standard" dataset. We utilize BERTScore to ensure the core legal meaning remains intact despite deep syntactic paraphrasing, achieving an F1 score of ~0.77.

License
This project is developed as an engineering diploma project.