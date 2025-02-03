<h1 align="center">
    SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model
</h1>
<p align="center">


# Introduction
This repository contains the official code of SafeRAG, a novel benchmark for evaluting the RAG (Retrieval-Augmented Generation) security. It includes the datasets we created for evaluating RAG security, and a tutorial on how to run the experiments on our benchmark.

# Project Structure
```bash
├── configs # This folder comprises scripts used to initialize the loading parameters of the large language models (LLMs) in RAG systems.
│   └── config.py # Before running the project, users need to fill in their own key or local model path information to the corresponding location. 
├── embeddings # The embedding model used to build vector databases.
│   └──base.py
├── knowledge_base # Path to knowledge_base.
│   ├──SN # The knowledge base used for silver noise.
│   |   ├──add_SN # Add attacks to the clean knowledge base.
│   |   └──db.txt # a clean knowledge base.
│   ├──ICC # The knowledge base used for inter-context conflict.
│   |   ├──add_ICC # Add attacks to the clean knowledge base.
│   |   └──db.txt # a clean knowledge base.
│   ├──SA  # The knowledge base used for soft ad.
│   |   ├──add_SA # Add attacks to the clean knowledge base.
│   |   └──db.txt # a clean knowledge base.
│   └──WDoS # The knowledge base used for White DoS.
│       ├──add_WDoS # Add attacks to the clean knowledge base.
│       └──db.txt # a clean knowledge base.
├── llms # This folder contains scripts used to load the LLMs.
│   ├── api_model.py # Call GPT-series models.
│   ├── local_model.py # Call a locally deployed model.
│   └── remote_model.py # Call the model deployed remotely and encapsulated into an API.
├── metric # The evaluation metric we used in the experiments.
│   ├── common.py  # bleu, rouge, bertScore.
│   └── quest_eval.py # Multiple-choice QuestEval. Note that using such metric requires calling a LLM such as GPT to answer questions, or modifying the code and deploying the question answering model yourself.
├── datasets # This folder contains scripts used to load the dataset.
├── output # The evaluation results will be retained here.
├── prompts # The prompts we used in the experiments.
├── retrievers # The retriever used in RAG system.
└── tasks # The evaluation attack tasks.
```

# Quick Start
- Install dependency packages
```bash
pip install -r requirements.txt
```

- Start the milvus-lite service(vector database)
```bash
milvus-server
```

- Download the bge-base-zh-v1.5 model to the /path/to/your/bge-base-zh-v1.5/ directory

- Modify config.py according to your need.

- Run quick_start_nctd.py

```bash
python quick_start_nctd.py \
  --retriever_name 'bm25' \
  --retrieve_top_k 6 \
  --filter_module 'off' \
  --model_name 'gpt-3.5-turbo' \
  --quest_eval_model 'deepseek-chat' \
  --attack_task 'SN' \
  --attack_module 'indexing' \
  --attack_intensity 0.5 \
  --shuffle True \
  --bert_score_eval \
  --quest_eval \
  --num_threads 5 \
  --show_progress_bar True
```


