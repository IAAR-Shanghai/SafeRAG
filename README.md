<div align="center"><h2>
<img src="https://github.com/IAAR-Shanghai/SafeRAG/blob/main/assets/sfr_logo.png" alt="sfr_logo" width=23px>SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model</h2></div>



**🎯 Who Should Pay Attention to Our Work?**

- **Exploring attacks on RAG systems?** SafeRAG introduces a **Threat Framework** that executes **Noise, Conflict, Toxicity, and Denial-of-Service (DoS) attacks** at various stages of the **RAG Pipeline**, aiming to **bypass RAG security components as effectively as possible and exploit its vulnerabilities**.
- **Developing robust and trustworthy RAG systems?** Our benchmark provides a new **Security Evaluation Framework** to test defenses and reveals systemic weaknesses in the **RAG Pipeline**.
- **Shaping RAG security policies?** SafeRAG provides **empirical evidence** of how **Data Injection** attacks can impact AI reliability.

> \[!IMPORTANT\]
>
> 🌟 **Star Us!** By starring our project on GitHub, you'll receive all release notifications instantly. We appreciate your support!

## :loudspeaker: News
- **[2025/01]** We released SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model.

## Overview
<div align="center">
    <img src="https://github.com/IAAR-Shanghai/SafeRAG/blob/main/assets/ACL_8pgs_Page3.png" alt="SafeRAG" width="93%">
</div>

<details><summary>Abstract</summary>
Retrieval-Augmented Generation (RAG) seamlessly integrates advanced retrieval and generation techniques, making it particularly well-suited for high-stakes domains such as law, healthcare, and finance, where factual accuracy is paramount. This approach significantly enhances the professional applicability of large language models (LLMs).  

But is RAG truly secure? **Clearly, attackers can manipulate the data flow at any stage of the RAG pipeline**—including **indexing, retrieval, and filtering**—by injecting malicious, low-quality, misleading, or incorrect texts into **knowledge bases, retrieved contexts, and filtered contexts**. These adversarial modifications **indirectly influence the LLM’s outputs**, potentially compromising its reliability.  

**SafeRAG systematically evaluates the security vulnerabilities of RAG components from both retrieval and generation perspectives.** Experiments conducted on **14 mainstream RAG components** reveal that **most RAG systems fail to effectively defend against data injection attacks**. Attackers can **manipulate the data flow within the RAG pipeline**, deceiving the model into generating **low-quality, inaccurate, or misleading content**, and in some cases, even **inducing a denial-of-service (DoS) response**.
</details>

We summarize our primary contributions as follows:

- We reveal four attack tasks capable of bypassing the **retriever**, **filter**, and **generator**. For each attack task, we develop a lightweight RAG security evaluation dataset, primarily constructed by humans with LLM assistance.
- We propose an economical, efficient, and accurate RAG security evaluation framework that incorporates attack-specific metrics, which are highly consistent with human judgment.
- We introduce the first Chinese RAG security benchmark, \textbf{SafeRAG}, which analyzes the risks posed to the **retriever** and **generator** by the injection of **Noise**, **Conflict**, **Toxicity**, and **DoS** at various stages of the RAG pipeline.

## Quick Start
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


> \[!Tip\]
> - You can modify the RAG components to be evaluated, attack tasks, and other parameters based on your specific evaluation needs.


## Results
The default retrieval window for the silver noise task is set to top K = 6, with a default attack injection ratio of 3/6. For other tasks, the default retrieval window is top K = 2, and the attack injection ratio is fixed at 1/2. We evaluated the impact of using different retrievers **DPR**, **BM25**, **Hybrid**, **Hybrid-Rerank** and filters (OFF, **filter NLI**, **compressor SKR** across different RAG stages **indexing**, **retrieval**, **generation**) on the contexts retrieved for various generators (**DeepSeek**, **GPT-3.5-turbo**, **GPT-4**, **GPT-4o**, **Qwen 7B**, **Qwen 14B**, **Baichuan 13B**, **ChatGLM 6B**). The bold values represent the default settings.
Then, we adopt a unified sentence chunking strategy to segment the knowledge base during indexing. The embedding model used is bge-base-zh-v1.5, the reranker is bge-reranker-base. 

### Results on Noise
We inject different noise ratios into the text accessible in the RAG pipeline, including the **knowledge base**, **retrieved context**, and **filtered context**.

<div align="center">
    <img src="https://github.com/IAAR-Shanghai/SafeRAG/blob/main/assets/sfr_result_of_task_N.png" alt="SafeRAG" width="93%">
</div>

> - Regardless of the stage where noise is injected, the F1 (avg) score exhibits a downward trend as the noise ratio increases, indicating a decline in generation diversity.
> - The retriever demonstrates some noise resistance, as noise injected at the knowledge base has approximately 50\% chance of not being retrieved. The results support this point. Specifically, as the noise ratio increases, the Retrieval Accuracy (RA) of injecting silver noise into the retrieved context or filtered context significantly outperforms that of injecting it into the knowledge base.
> - The performance of injecting noise into the retrieved context and filtered context is similar, indicating that the filter cannot effectively resist silver noise since silver noise still supports answering the query.
> - Different retrievers exhibit varying levels of robustness to noise. Overall, the ranking is Hybrid-Rerank > Hybrid > BM25 > DPR, suggesting that compared to attack contexts, hybrid retriever and rerankers show a preference for retrieving golden contexts.  
> - Compression-based filters like SKR are not sufficiently secure, as they tend to lose detailed information, leading to a decrease in F1 (avg).
### Results on Conflict, Toxicity, and DoS
<div align="center">
    <img src="https://github.com/IAAR-Shanghai/SafeRAG/blob/main/assets/sfr_result_of_task_C.png" alt="SafeRAG" width="93%">
    <img src="https://github.com/IAAR-Shanghai/SafeRAG/blob/main/assets/sfr_result_of_task_T.png" alt="SafeRAG" width="93%">
    <img src="https://github.com/IAAR-Shanghai/SafeRAG/blob/main/assets/sfr_result_of_task_D.png" alt="SafeRAG" width="93%">
</div>

> - After injecting different types of attacks into the texts accessible by the RAG pipeline, it was observed that the retrieval accuracy (RA) and the attack failure rate (AFR) decreased across all three tasks. The ranking of attack effectiveness at different RAG stages was: filtered context > retrieved context > knowledge base. Furthermore, adding conflict attack increased the likelihood of misjudging incorrect options as correct, leading to a drop in F1 (avg). Introducing DoS attack reduced F1 (avg) and severely impacted generative diversity. 
> - Retrievers exhibited different vulnerabilities to various attacks. For instance, Hybrid-Rerank was more susceptible to conflict attack, while DPR was more prone to DoS attack. Both experienced a significant decrease in AFR. Additionally, all retrievers showed consistent AFR degradation under toxicity attack. After adding conflict attack, the F1 (avg) scores of all retrievers became similar, indicating stable attack effectiveness. However, DPR was more affected by DoS attack compared to other retrievers, as evidenced by its significantly larger decline in the diversity metric F1 (avg).  
> - The RA of different retrievers was largely consistent across different attack tasks.  
> - In conflict tasks, using the SKR filter was less secure because it could compress conflict details, resulting in a decline in F1 (avg). In toxicity and DoS tasks, the NLI filter was generally ineffective, with its AFR close to that of disabling the filter. However, the SKR filter proved to be safe in these tasks, as it was able to compress soft ads and warnings.  


## TODOs
<details><summary>Click me to show all TODOs</summary>

- [ ] feat: add SafeRAG PyPI package.
- [ ] feat: release SafeRAG dataset on Hugging Face.
- [ ] docs: extend dataset.
</details>

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

<p align="right"><a href="#top">🔝Back to top</a></p>
