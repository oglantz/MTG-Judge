---
layout: default
title:  Final
---
## Video  

## Project Summary
<!-- Short paragraph: main idea, further updated/clarified from status. -->
MTG-Judge aims to build a system that answers Magic: The Gathering rules questions by retrieving relevant rules and card text and then generating a clear, cited ruling. The system takes a question (with card names marked, e.g. [[Card Name]]), gathers accurate rules and card data, and produces an answer that is both correct and explainable. It uses retrieval-augmented generation (RAG) and grounded card data: a vector index over the Comprehensive Rules supplies relevant rule text, an external API supplies exact card text and rulings, and an open-source LLM uses that context to produce the final ruling. The focus is on improving retrieval and prompting so the model relies on the retrieved context instead of hallucinating.  

The task is difficult for several reasons. The rules corpus is large and highly structured, and correct answers often depend on several rules and card wordings together; small phrasing differences can change the outcome. Commercial LLMs also tend to hallucinate card text and misapply rules when asked ruling questions, so using an LLM alone is not reliable. Retrieval and grounding address this: the system must first fetch the right rules and accurate card information, then reason over that context. Pure keyword search or hand-written rules cannot robustly handle the variety of phrasings and the need for semantic matching across the full rule set, so we treat this as an AI/ML problem that combines vector retrieval, external APIs, and an LLM so the model can both find and use the right evidence.  

Through this problem, MTG-Judge shows how retrieval and grounding can improve the reliability of LLMs in expert, citation-heavy domains. The same ideas (RAG over a large text corpus and grounding with authoritative sources) apply beyond gaming to legal, medical, or technical QA where correctness and traceability matter. The project illustrates how combining retrieval, structured data, and language models can address tasks that are neither trivial for rule-based systems nor safe for raw LLM use.  

## Approach  

## Evaluation  

## Resources Used
<!-- Code docs, libraries, source code, StackOverflow, etc. Include a comprehensive description of any use of AI tools. -->
[LlamaIndex](https://pypi.org/project/llama-index/)

[PyTorch](https://pytorch.org/)

[FAISS](https://pypi.org/project/faiss/)

HuggingFace Models (specifically [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), [bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli), and [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct))

[Scryfall API](https://scryfall.com/docs/api)

### AI Usage  

Commercial LLMs were used in the process of iterating on the specific implementations of system designs we tested, but the design of the pipelines themselves and research into relevant techniques were all done manually. 

