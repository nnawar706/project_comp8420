# Intelligent Customer Service System

### COMP8420 – Use Case 1

## Project Overview

This project presents an end-to-end Intelligent Customer Service System that combines traditional Natural Language Processing (NLP) techniques with Large Language Models (LLMs) to automate customer support workflows.

Given a customer enquiry written in natural language, the system performs the following tasks:

1. **Intent Classification** – identifies the type of customer request and assigns a confidence score.
2. **Sentiment Analysis** – determines the emotional tone of the message.
3. **Named Entity Recognition (NER)** – extracts relevant entities such as order numbers, customer information, and issue descriptions.
4. **Knowledge Retrieval (RAG)** – retrieves relevant company policy information from a knowledge base.
5. **Response Generation** – produces a grounded customer service response using a local LLM.
6. **Escalation Decision** – determines whether the enquiry can be resolved automatically or should be escalated to a human support agent.

The system integrates multiple NLP and LLM techniques into a single customer support workflow while maintaining privacy through local execution without external API dependencies.

---

## Dataset

The project uses the **Bitext Customer Support Dataset**, containing approximately 27,000 labelled customer support message–response pairs across multiple customer service categories.

The project specification permits the use of relevant alternative datasets. The Bitext dataset was selected because it provides realistic customer service interactions and clearly labelled support intents suitable for classification, retrieval, and response generation tasks.

---

## Notebook Structure

The implementation is organised into a series of Jupyter notebooks. Reusable functionality is implemented within the `src/` directory, while the notebooks focus on experimentation, analysis, visualisation, and evaluation.

| Notebook                                | Description                                                                                                                             |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **01_data_exploration.ipynb**           | Dataset loading, exploratory data analysis, class distribution analysis, and train/test preparation.                                    |
| **02_preprocessing.ipynb**              | Text cleaning, tokenisation, normalisation, lemmatisation, PII redaction, and privacy considerations.                                   |
| **03_basic_nlp.ipynb**                  | Traditional NLP techniques including TF-IDF classification, sentiment analysis, named entity recognition, and POS tagging.              |
| **04_llm_prompting.ipynb**              | Prompt engineering experiments using local LLMs, including few-shot prompting, Chain-of-Thought reasoning, and structured JSON outputs. |
| **05_rag.ipynb**                        | Retrieval-Augmented Generation (RAG) pipeline using sentence embeddings, ChromaDB, and retrieval comparisons.                           |
| **06_agentic_pipeline.ipynb**           | Agentic customer service workflow with tool use, dialogue state tracking, escalation logic, and user interface integration.             |
| **07_evaluation.ipynb**                 | System evaluation including classification metrics, NER evaluation, RAG ablations, prompt ablations, and human evaluation preparation.  |
| **08_calibration_and_robustness.ipynb** | Confidence calibration, escalation robustness, out-of-domain detection, and reliability analysis.                                       |
| **09_neural_robustness_intent.ipynb**   | Neural versus classical classifiers, noise robustness experiments, and fine-grained intent classification.                              |

---

## Implemented Techniques

### Basic NLP Techniques

* Text preprocessing and normalisation
* TF-IDF feature extraction
* Intent classification
* Sentiment analysis
* Named entity recognition (NER)
* Part-of-speech (POS) tagging
* Rule-based information extraction
* TF-IDF knowledge retrieval

### Advanced NLP and LLM Techniques

* Foundation language models
* Prompt engineering
* Few-shot prompting
* Chain-of-Thought reasoning
* Retrieval-Augmented Generation (RAG)
* Agentic workflow design
* Automated escalation logic
* LLM-based evaluation
* Confidence calibration
* Out-of-domain detection

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

For notebooks that use local LLMs:

```bash
# Install Ollama
# https://ollama.com

ollama pull llama3.2
```

---

## Project Setup

```bash
python scripts/download_data.py
python scripts/build_knowledge_base.py
python scripts/build_index.py
python scripts/train_classifier.py
```

---

## Running the System

### Launch the Streamlit Interface

```bash
streamlit run app/streamlit_app.py
```

### Run the Pipeline from the Command Line

```bash
python scripts/run_pipeline.py "My order #48213 hasn't arrived and I'm furious"
```

---

## Evaluation and Reproducibility

The project includes comprehensive evaluation covering:

* Classification performance
* Per-class error analysis
* Named entity recognition evaluation
* Confidence calibration
* Escalation robustness
* Retrieval-Augmented Generation ablations
* Prompt engineering ablations
* Noise robustness testing
* Fine-grained intent classification

To support reproducibility:

* Fixed random seeds are used where applicable.
* Model artefacts are saved locally.
* Dependencies are specified in `requirements.txt`.
* Reusable logic is separated into the `src/` package.
* All experiments are documented within the accompanying notebooks.

---

## Project Structure

```text
Codes/
│
├── 01-09_*.ipynb
├── config.py
│
├── src/
│   ├── preprocessing.py
│   ├── classification.py
│   ├── sentiment.py
│   ├── ner.py
│   ├── pos_tagging.py
│   ├── rag.py
│   ├── generation.py
│   ├── agent.py
│   ├── pipeline.py
│   ├── evaluation.py
│   └── data_loader.py
│
├── scripts/
├── app/
├── data/
├── models/
├── vector_db/
└── requirements.txt
```

---

**Author:** Group C
**Unit:** COMP8420 – Natural Language Processing
**Project:** Intelligent Customer Service System
