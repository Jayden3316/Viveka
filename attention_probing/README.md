# Attention Probing

This directory contains the code for the attention probing experiment.

## Overview

Typical probing methods are useful to detect behaviour in the model, but are not useful to understand the mechanisms that give rise to them. In this experiment, we aim to use a method to allow for detection along with insight into underlying mechanisms.

We focus on detecting/ calibrating uncertainty in text generation in LLMs. 

### Assumptions:
- LLMs are capable of detecting uncertainty in their predictions through their intermediate activations.
- These intermediate activations contain some representation of the uncertainty in the predictions.
- These may arise at the attention/ MLP, but multiple layers may contribute to these representations arising.

### Dataset:
- We use the first 20,000 queries from `trivia-qa` for training and 2000 questions from `trivia-qa`, `halueval` and `pop-qa` for validation.

### Methodology:
- We add an additional `[PROBE]` token to the vocabulary of the LLM.
- At each attention layer, attention outputs from all token positions are added to the `[PROBE]` token position except for the `<bos>` token. The `<bos>` token cannot carry any information about uncertainty in a model trained with causal attention.
- Potentially, a gating mechanism can be added select whether the MLP outputs should be added to the `[PROBE]` token position.

### Objective function:
- The probe is trained as:
    - A classification task to predict whether the sentence is uncertain or not
    - A regression task to predict the uncertainty score of the sentence

For both tasks, we base our study `semantic entropy`.

All logging is done using `wandb`. Checkpoints can be found at `attention_probing/checkpoints`.

### Directory Structure:
```
attention_probing/
├── README.md
├── requirements.txt
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   ├── generation_config.yaml
│   └── data_config.yaml
├── checkpoints/
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── data.py
│   ├── uncertainty.py
│   ├── train.py
│   ├── generate.py
│   ├── inference.py
│   ├── utils.py
│   └── visualization.py
├── main.py
└── notebooks/
    └── sandbox.ipynb
```

### Example usage
#### Train
```bash
python main.py train --config configs/training_config.yaml
```
#### Inference on single question
```bash
python main.py inference \
    --checkpoint checkpoints/best_model.pt \
    --question "What is the capital of France?"
```
#### Evaluate on dataset
```bash
python main.py evaluate \
    --checkpoint checkpoints/best_model.pt \
    --eval-data data/eval_questions.json \
    --output results/evaluation.json
```
#### Analyze layer importance
```bash
python main.py analyze \
    --checkpoint checkpoints/best_model.pt \
    --questions-file data/test_questions.json \
    --output results/layer_analysis.json
```
#### Generate samples from base models
Generate completions using standard models (without probe tokens) on datasets like TriviaQA, PopQA, etc. results are saved to JSON.

**Run for a specific model:**
```bash
python src/generate.py --config configs/generation_config.yaml --model llama-3.1-8b-instruct
```

**Run for all models in config:**
```bash
python src/generate.py --config configs/generation_config.yaml --all
```
