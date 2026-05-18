# FLSG
From Linear to Structured Graphs: Enhancing Multi-Hop Question Answering with Executable Node-Level Retrieval and Reasoning
# From Linear to Structured Graphs (FLSG)

Official implementation for the paper:

**From Linear to Structured Graphs: Enhancing Multi-Hop Question Answering with Executable Node-Level Retrieval and Reasoning**

---

## Overview

Multi-hop Question Answering (QA) remains a challenging task for Large Language Models (LLMs), primarily due to hallucination and weak multi-step reasoning abilities.

This repository implements several reasoning baselines and our proposed framework:

- Few-Shot
- Chain-of-Thought (CoT)
- Self-Ask
- ITER-RETGEN
- ThinkingMethod (Our proposed FLSG framework)

Our method transforms traditional linear reasoning into a structured executable graph for multi-hop QA. By decomposing a complex question into interconnected sub-questions and executing them in a topology-aware layer-wise manner, the framework significantly improves reasoning accuracy and retrieval quality.

---

## Abstract

For large language models (LLMs), multi-hop question answering (QA) continues to pose a significant challenge, largely because these models are prone to generating hallucinated content and lack the ability to carry out systematic, multi-step reasoning.

Although Retrieval-Augmented Generation (RAG) helps reduce hallucinations through the integration of external knowledge, existing approaches largely rely on linear reasoning pipelines and iterative retrieval-generation frameworks. As a consequence, the complex relationships between reasoning steps are not well modeled, causing these methods to underperform on multi-hop QA tasks.

To address these issues, we propose a novel framework: **From Linear to Structured Graphs (FLSG)** that transforms linear reasoning into a structured, executable graph for multi-hop QA.

Specifically, FLSG adopts a Structured Graph decomposition approach, breaking a complex question into a set of interconnected sub-questions. The graph is executed in a layer-wise manner using a topology-aware scheduling strategy, enabling parallel reasoning over independent nodes.

Furthermore, we integrate node-level retrieval using BM25, allowing each sub-question to retrieve tailored evidence, and employ answer propagation to ensure consistency across reasoning steps.

Experiments on HotpotQA, 2WikiMultihopQA, and MuSiQue demonstrate that our approach outperforms competitive baselines such as CoT, Self-Ask, and ITER-RETGEN under both retrieval and non-retrieval settings.

---

## Project Structure

```bash
project/
│
├── Few-Shot/
├── CoT/
├── Self-Ask/
├── Iter-Test/
├── ThinkingMethod/
│
├── datasets/
├── rag_paragraph_corpus.json
├── requirements.txt
└── README.md
