# 🧠 Clean RAG Architecture: LangChain (LCEL) & Local Causal LLM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
[![Framework: LangChain](https://img.shields.io/badge/LangChain-LCEL-green)](https://python.langchain.com/)
[![Model: TinyLlama](https://img.shields.io/badge/LLM-TinyLlama_1.1B-blue)](https://huggingface.co/TinyLlama)

## 📌 Project Overview
This project demonstrates a production-grade, 100% local **Retrieval-Augmented Generation (RAG)** pipeline. It moves beyond basic API wrappers (like OpenAI) to showcase deep competence in open-source AI deployment, vector mathematics, and advanced pipeline orchestration using **LangChain Expression Language (LCEL)**.

The engine extracts unstructured web data, maps it into mathematical vectors using local embedding models, stores it in a **FAISS Vector Database**, and utilizes a modern **Causal Decoder LLM** (`TinyLlama-1.1B-Chat`) to synthesize accurate, context-aware answers without relying on paid, closed-source APIs.

## 🛠️ Core Technologies & Competencies
* **Framework orchestration:** Utilizing LangChain & LCEL to build highly readable, modular, and maintainable AI chains.
* **Data Ingestion & Chunking:** `WebBaseLoader` for web scraping and `RecursiveCharacterTextSplitter` for semantic document windowing.
* **Vector Mathematics & Search:** Implementing `Sentence-Transformers` (`all-MiniLM-L6-v2`) integrated with Facebook AI Similarity Search (**FAISS**) for ultra-fast, memory-efficient nearest-neighbor retrieval.
* **Generative AI Deployment:** Circumventing outdated Seq2Seq limitations by deploying a modern, instruction-tuned decoder model (`TinyLlama`) locally via `HuggingFacePipeline`.

## 🏗️ Pipeline Architecture
The core of this project is the implementation of a "Clean Pipeline" using LCEL syntax and modern ChatML instruction tags. The data flow is explicitly defined in a single, declarative chain:

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | llm
    | StrOutputParser()
)
