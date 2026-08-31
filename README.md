# AI-Driven NLP Using Qwen3

**Implementation, Capability Evaluation, and Performance Analysis of the Qwen3-4B-Instruct Language Model**

Internship: ShadowFox
Domain: Natural Language Processing & Machine Learning

---

## Overview

This project implements and evaluates **Qwen/Qwen3-4B-Instruct-2507**, an open-weight instruction-tuned language model, using the Hugging Face `transformers` library. Beyond just running the model, the project systematically tests it across multiple NLP tasks, measures its computational performance, studies how prompt design and sampling temperature affect output quality, and analyzes its strengths, limitations, and ethical implications.

## Why Qwen3-4B-Instruct?

- **Instruction-tuned** — built for chat-style, task-following interactions, which suits multi-task evaluation.
- **Manageable size (4B params)** — runs on a single GPU (float16) while still showing strong reasoning and generation quality.
- **Recent architecture** — a current-generation open-weight model, more representative of modern LM design than older baselines like GPT-2 or BERT.
- **Open weights on Hugging Face** — gives full local control over generation parameters (temperature, sampling, max tokens) needed for the experiments below.

## What's Inside the Notebook

| Section | What it does |
|---|---|
| Setup | GPU check, library installation, tokenizer/model loading, parameter count |
| Basic Text Generation | Baseline response to a simple prompt, with timing and token throughput |
| Question Answering | 5 ML-concept questions, scored on correctness/relevance/clarity/completeness |
| Text Summarization | Condenses multi-sentence passages into 3-sentence summaries |
| Sentiment Analysis | Classifies 10 labeled statements as Positive/Negative/Neutral; scored with `accuracy_score` and a classification report |
| Reasoning | 4 logic/arithmetic word problems, checked against expected answers |
| Code Generation | Generates a working `is_prime(n)` Python function from a natural-language spec |
| Prompt Engineering | Compares zero-shot vs. few-shot vs. structured prompting on the same classification task |
| Temperature Analysis | Same prompt run at temperature = 0.1, 0.5, 0.8, 1.0; compares output length and diversity |
| Performance Evaluation | Inference time and tokens/sec per task, visualized with bar and line charts |
| Hallucination Check | Probes the model with a question about a future event to see if it fabricates a confident but false answer |
| Ethical Considerations | Bias, hallucination, privacy, human oversight, and misuse risks |
| Research Questions (RQ1–RQ5) | Direct findings tied back to the experiments above |

## Key Findings

- Qwen3-4B-Instruct performed strongly and coherently across QA, summarization, sentiment classification, and code generation.
- **Prompt design matters**: few-shot and structured prompts produced more consistent, correctly-formatted outputs than zero-shot prompts.
- **Temperature trade-off**: lower temperatures (≈0.1–0.3) gave focused, deterministic answers; higher temperatures (≈0.8–1.0) increased diversity at some cost to factual precision.
- The model achieved practical inference speed on GPU hardware, making it usable for interactive applications.
- Limitations observed: occasional factual inaccuracies, sensitivity to prompt phrasing, degraded performance on more complex multi-step reasoning, and hallucination when asked about events outside its knowledge (e.g., a future sports result).

## Repository Contents

```
AI-Driven-NLP-Qwen3/
├── AI_Driven_NLP_Qwen3.ipynb   # Full notebook: implementation + analysis
├── README.md                   # This file
```

## How to Run

This model requires a GPU (float16 inference). The easiest way to run it is via **Google Colab**:

1. Open the notebook in Colab (upload it or open directly from GitHub via `File > Open Notebook > GitHub`).
2. Set the runtime to a GPU (`Runtime > Change runtime type > GPU`).
3. Run all cells top to bottom.

To run locally with a CUDA-capable GPU:

```bash
git clone https://github.com/<your-username>/AI-Driven-NLP-Qwen3.git
cd AI-Driven-NLP-Qwen3
pip install -r requirements.txt
jupyter notebook AI_Driven_NLP_Qwen3.ipynb
```

## Ethical Considerations

Large language models like Qwen3 offer significant capability but carry real risks: training-data bias can surface in outputs, the model can generate plausible-sounding but incorrect information (hallucination), sensitive data should never be submitted without safeguards, and outputs used in high-stakes domains (healthcare, law, finance) require human review. These considerations are discussed in more detail within the notebook.

## Potential Applications

Educational assistants, customer support chatbots, document summarization, content generation, sentiment analysis pipelines, coding assistance, and research support tools.

## Conclusion

This project demonstrates that Qwen3-4B-Instruct is a capable, versatile language model suitable for a range of real-world NLP tasks, while also surfacing the practical limitations (hallucination, prompt sensitivity, reasoning gaps) that make human oversight necessary when deploying LLMs in production settings.
