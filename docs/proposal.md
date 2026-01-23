---
layout: default
title: Proposal
---



## Summary of the Project

Magic: The Gathering has a lot of rules, and they're all weird. Interpretering them is a hard enough task that if you want to be a real judge, you must pass multiple exams. We propose an application driven project leveraging LLM's, Vector DBs, and RL based fine-tuning to make a system that takes in a properly formatted MTG rules question, and outputs a clear, concise, accurate ruling.


## Project Goals

* **Minimum goal**
    - Rulings returned by our system are accurate to at least a casual fan's level of knowledge.

* **Realistic goal**
    - Our system is capable of achieving a passing score on a L1 judge exam.

* **Moonshot goal**
    - Our system is capable of achieving a passing score on a L2/L3 judge exam, and deploying a working MVP to the public.


## AI/ML Algorithms
We plan on leveraging LLMs, vector embedding models, and off policy RL fine tuning.


## Evaluation Plan
Quantitatively, we have judge exams that will be used to verify how well our application performs. Even if we can't automate our system taking the test, they will give us an exact number to inform us if we have succeeded in our goals (passing <= 65%). Other metrics that we will consider, but not prioritize over accuracy will be latency and VectorDB file size.

Qualitatively, we want the shape of our output to match that of a good judge. This includes, relevant rule citation, non-judgmental tone, unambiguous wording, reasoning behind the ruling, and brevity where possible. Qualitative verification should be relatively simple in a text domain; our visualization will be the text output.


## AI Tool Usage
All code will be written by group members, however code may be debugged using AI. Commercial LLMs are quite poor at MTG ruling judgments, so they will be of little use outside of this.