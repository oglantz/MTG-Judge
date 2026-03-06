---
layout: default
title:  Final
---
## Video  

## Project Summary
<!-- Short paragraph: main idea, further updated/clarified from status. -->
MTG-Judge aims to build a system that answers Magic: The Gathering rules questions by retrieving relevant rules and card text and then generating a clear, cited ruling. The system takes a question (with card names marked, e.g. `[[Card Name]]`), gathers accurate rules and card data, and produces an answer that is both correct and explainable. It uses retrieval-augmented generation (RAG) and grounded card data: a vector index over the Comprehensive Rules supplies relevant rule text, an external API supplies exact card text and rulings, and an open-source LLM uses that context to produce the final ruling. The focus is on improving retrieval and prompting so the model relies on the retrieved context instead of hallucinating.  

The task is difficult for several reasons. The rules corpus is large and highly structured, and correct answers often depend on several rules and card wordings together; small phrasing differences can change the outcome. Commercial LLMs also tend to hallucinate card text and misapply rules when asked ruling questions, so using an LLM alone is not reliable. Retrieval and grounding address this: the system must first fetch the right rules and accurate card information, then reason over that context. Pure keyword search or hand-written rules cannot robustly handle the variety of phrasings and the need for semantic matching across the full rule set, so we treat this as an AI/ML problem that combines vector retrieval, external APIs, and an LLM so the model can both find and use the right evidence.  

Through this problem, MTG-Judge shows how retrieval and grounding can improve the reliability of LLMs in expert, citation-heavy domains. The same ideas (RAG over a large text corpus and grounding with authoritative sources) apply beyond gaming to legal, medical, or technical QA where correctness and traceability matter. The project illustrates how combining retrieval, structured data, and language models can address tasks that are neither trivial for rule-based systems nor safe for raw LLM use.  

## Approach

**End-to-end pipeline.** The user submits a rules question with card names in double brackets. The query processor extracts those names, cleans the query (replacing `[[Card Name]]` with the name), and calls the Scryfall API for oracle text and official rulings for each card. That yields a card context string (formatted for the LLM). The cleaned query is sent to the RAG module, which returns a rules context string. The LLM receives the concatenation of card context, rules context, and the cleaned question and produces a single, cited ruling. We use Qwen 2.5 7B Instruct (HuggingFace) as the generator.

**Rule parsing and indexing.** The MTG Comprehensive Rules source is parsed with a stateful, regex-based parser that walks the document line by line and uses patterns for section, topic, rule, and subrule lines (e.g. 501, 501.1, 501.1a). Each rule or subrule is emitted as a document with metadata (section code, topic code, rule code, subrule code if present). Blank lines separate entries so that multi-line rules and "Example:" blocks stay with the correct rule. The parser also maintains a rule map: for each rule code, the full text of that rule (including all subrules) is stored so we can expand the top retrievals into full rules at query time.

**Zero-shot rule tagging.** Each parsed document is tagged with a zero-shot classifier so that retrieval can later use rule type. We use BART (facebook/bart-large-mnli) via the HuggingFace zero-shot-classification pipeline with a fixed set of system labels (e.g. combat, casting, mana, abilities, state-based actions, continuous effects, priority). The model scores each label for the rule text; we keep labels above a confidence threshold (e.g. 0.7) and store them in the document's metadata as system tags. Tagging is run in batch over all documents at index-build time so it is done once per index rebuild. Documents with at least one tag are retained for retrieval; the rest remain in the index but without system metadata. This gives every rule/subrule a consistent set of tags for filtering or ranking at query time.

**RAG-based rule retrieval.** Tagged documents are embedded with all-mpnet-base-v2 (sentence-transformers) and indexed in FAISS (LlamaIndex with faiss-cpu). The FAISS index, LlamaIndex storage context, and the rule map are persisted so later runs load from disk instead of re-parsing and re-tagging. At query time we embed the cleaned question, run top-k retrieval (e.g. top 10) from the vector index, then build the rules context string: for the top 3 results we append the full rule from the rule map (by rule code), and for the remaining results we append only the retrieved subrule text. The result is the rules context passed to the LLM. In the final implementation, query tagging (classifying the user question with the same or similar labels) can be added so that retrieval optionally filters or ranks by overlap between query tags and rule tags; the current pipeline already has the rule-side tags in place to support that.

**Card grounding.** Card names extracted from the query are looked up via the Scryfall API. For each card we obtain oracle text and any official rulings; mana symbols and other notation in the text are normalized to plain language (e.g. {T} to "tap") so the LLM sees a consistent format. This card context is included in the prompt so the model reasons from actual card text instead of guessing.

**LLM generation.** The prompt is structured as: (1) system message defining the model's role as an MTG judge and instructing it to reason from the provided context and cite rules; (2) user message containing the card context, rules context, and the cleaned question. We use a low temperature and standard decoding (e.g. top_p) so outputs are stable and focused. The model's reply is returned as the ruling. No post-processing or second-pass refinement is applied in the current pipeline.

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

