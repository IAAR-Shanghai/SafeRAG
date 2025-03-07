<div align="center"><h2>
<img src="https://github.com/IAAR-Shanghai/SafeRAG/blob/main/assets/sfr_logo.png" alt="sfr_logo" width=23px> SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model</h2></div>


<p align="center">
<!-- arXiv badge with a more vibrant academic red -->
<a href="https://arxiv.org/abs/2501.18636">
<img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=flat-square&logo=arxiv&logoColor=white">
</a>
<!-- Github badge with clean dark color -->
<a href="https://github.com/IAAR-Shanghai/SafeRAG">
<img src="https://img.shields.io/badge/Github-Code-181717?style=flat-square&logo=github&logoColor=white">
</a>
<!-- Huggingface Collection badge with more dynamic orange -->
<a href="https://huggingface.co/collections/zaown/saferag-67b422f4072adee12c784bde">
<img src="https://img.shields.io/badge/Huggingface-Collection-FF6F00?style=flat-square&logo=huggingface&logoColor=white">
</a>
</p>

<div align="center">
<p>
Xun Liang<sup>1,*</sup>,
<a href="https://github.com/siminniu">Simin Niu</a><sup>1,*</sup>,
<a href="https://github.com/zhiyulee-RUC">Zhiyu Li</a><sup>2,†</sup>,
Sensen Zhang<sup>1</sup>, Hanyu Wang<sup>1</sup>, Feiyu Xiong<sup>2</sup>, Jason Zhaoxin Fan<sup>3</sup>, 
Bo Tang<sup>2</sup>, Shichao Song<sup>1</sup>, Mengwei Wang<sup>1</sup>, Jiawei Yang<sup>1</sup>
</p>
<p>
<sup>1</sup><a href="https://en.ruc.edu.cn/">Renmin University of China</a>, 
<sup>2</sup><a href="https://www.iaar.ac.cn/">Institute for Advanced Algorithms Research, Shanghai</a>,
<sup>3</sup><a href="https://www.buaa.edu.cn/">Beihang University</a>
</p>
</div>

<div align="center">
<h5>For business inquiries, please contact us at <a href="mailto:lizy@iaar.ac.cn">lizy@iaar.ac.cn</a>.</h5>
</div>



**🎯 Who Should Pay Attention to Our Work?**

- **Exploring attacks on RAG systems?** SafeRAG introduces a **Threat Framework** that executes **Noise, Conflict, Toxicity, and Denial-of-Service (DoS) attacks** at various stages of the **RAG Pipeline**, aiming to **bypass RAG security components as effectively as possible and exploit its vulnerabilities**.
- **Developing robust and trustworthy RAG systems?** Our benchmark provides a new **Security Evaluation Framework** to test defenses and reveals systemic weaknesses in the **RAG Pipeline**.
- **Shaping RAG security policies?** SafeRAG provides **empirical evidence** of how **Data Injection** attacks can impact AI reliability.


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
- We introduce the first Chinese RAG security benchmark, SafeRAG, which analyzes the risks posed to the **retriever** and **generator** by the injection of **Noise**, **Conflict**, **Toxicity**, and **DoS** at various stages of the RAG pipeline.

## Quick Start
- Install dependency packages
```bash
pip install -r requirements.txt
```

- Start the milvus-lite service (vector database)
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
The default retrieval window for the silver noise task is set to top K = 6, with a default attack injection ratio of 3/6. For other tasks, the default retrieval window is top K = 2, and the attack injection ratio is fixed at 1/2. 
We evaluate the security of 14 different types of RAG components against injected attack texts at different RAG stages (**indexing**, retrieval, and generation), including: (1) retrievers (**DPR**, BM25, Hybrid, Hybrid-Rerank); (2) filters (OFF, **filter NLI**, compressor SKR); and (3) generators (**DeepSeek**, GPT-3.5-turbo, GPT-4, GPT-4o, Qwen 7B, Qwen 14B, Baichuan 13B, ChatGLM 6B).
The bold values represent the default settings. Additionally, we adopt a unified sentence chunking strategy to segment the knowledge base during the indexing. The embedding model used is bge-base-zh-v1.5, the reranker is bge-reranker-base.  

### Results on Noise
We inject different noise ratios into the text accessible in the RAG pipeline, including the **knowledge base**, **retrieved context**, and **filtered context**.

<div align="center">
    <img src="https://github.com/IAAR-Shanghai/SafeRAG/blob/main/assets/sfr_result_of_task_N.png" alt="SafeRAG" width="93%">
</div>


> - Regardless of the stage at which noise injection is performed, the F1 (avg) decreases as the noise ratio increases, indicating a decline in the diversity of generated responses.
> - Different retrievers exhibit varying degrees of noise resistance. The overall ranking of retrievers' robustness against noise attacks is Hybrid-Rerank > Hybrid > BM25 > DPR. This suggests that hybrid retrievers and rerankers are more inclined to retrieve diverse golden contexts rather than homogeneous attack contexts.
> - When the noise ratio increases, the retrieval accuracy (RA) for noise injected into the retrieved or filtered context is significantly higher than that for noise injected into the knowledge base. This is because noise injected into the knowledge base has approximately a 50% chance of not being retrieved.
> - The compressor SKR lacks sufficient security. Although it attempts to merge redundant information in silver noise, it severely compresses the detailed information necessary to answer questions within the retrieved context, leading to a decrease in F1 (avg).

### Results on Conflict, Toxicity, and DoS
<div align="center">
    <img src="https://github.com/IAAR-Shanghai/SafeRAG/blob/main/assets/sfr_result_of_task_C.png" alt="SafeRAG" width="93%">
    <img src="https://github.com/IAAR-Shanghai/SafeRAG/blob/main/assets/sfr_result_of_task_T.png" alt="SafeRAG" width="93%">
    <img src="https://github.com/IAAR-Shanghai/SafeRAG/blob/main/assets/sfr_result_of_task_D.png" alt="SafeRAG" width="93%">
</div>

> - After injecting different types of attacks into the texts accessible at any stage of the RAG pipeline, both F1 (avg) and the attack failure rate (AFR) decline across all three tasks. Specifically, conflict attacks make it difficult for the RAG to determine which information is true, potentially leading to the use of fabricated facts from the attack context, resulting in a drop in metrics. Toxicity attacks cause the RAG to misinterpret disguised authoritative statements as factual, leading to the automatic propagation of soft ads in generated responses, which also contributes to the metric decline. DoS attacks, on the other hand, make the RAG more likely to refuse to answer, even when relevant evidence is retrieved, further reducing the performance metrics. Overall, the ranking of attack effectiveness across different stages is: filtered context > retrieved context > knowledge base.   
> - Different retrievers exhibit varying vulnerabilities to different types of attacks. For instance, Hybrid-Rerank is more susceptible to conflict attacks, while DPR is more prone to DoS attacks. The vulnerability levels of retrievers under toxicity attacks are generally consistent. 
> - Across different attack tasks, the changes in RA remain largely consistent regardless of the retriever used.   
> - In conflict tasks, using the compressor SKR is less secure as it compresses conflict details, leading to a decline in F1 (avg). In toxicity and DoS tasks, the filter NLI is generally ineffective, with its AFR close to that of disabling the filter. However, in toxicity and DoS tasks, the SKR compressor proves to be secure as it effectively compresses soft ads and warning content. 

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
