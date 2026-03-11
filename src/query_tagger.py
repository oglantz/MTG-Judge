"""
Zero-shot query tagger for classifying incoming MTG rules queries.

Uses the same facebook/bart-large-mnli model as the rule parser so that
queries and rules are tagged by the same model with identical label descriptions.

Usage:
    from query_tagger import QueryTagger

    tagger = QueryTagger()          # GPU auto-detected
    tags = tagger.tag(cleaned_query, card_context)
    # e.g. ["combat", "abilities"]
"""

import torch
from transformers import pipeline

# Single source of truth for the tag vocabulary.
# Each entry is (short_name, description_passed_to_classifier).
# slop.py's system_labels instance variable must stay in sync with this list.
SYSTEM_LABELS: list[tuple[str, str]] = [
    ("combat",            "This Magic: The Gathering question is about combat"),
    ("casting",           "This Magic: The Gathering question is about casting spells"),
    ("mana",              "This Magic: The Gathering question is about mana"),
    ("abilities",         "This Magic: The Gathering question is about card abilities"),
    ("state-based actions", "This Magic: The Gathering question is about state-based actions"),
    ("continuous effects","This Magic: The Gathering question is about continuous effects"),
    ("priority",          "This Magic: The Gathering question is about priority"),
    ("stack",             "This Magic: The Gathering question is about the stack")
]


class QueryTagger:
    def __init__(self, device=None):
        """
        Args:
            device: HuggingFace device id. Pass 0 for GPU, -1 for CPU.
                    Defaults to GPU if available, otherwise CPU.
        """
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self._tagger = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device,
        )

        self._label_descriptions = [desc for _, desc in SYSTEM_LABELS]
        self._desc_to_name = {desc: name for name, desc in SYSTEM_LABELS}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag(self, query: str, card_context: str = "", threshold: float = 0.8) -> list[str]:
        """
        Classify a query and return a list of matching system tag names.

        Args:
            query:        the cleaned query text.
            card_context: optional oracle text / rulings block from QueryProcessor.
                          When provided it is appended to the query so the model
                          has richer context when scoring each label.
            threshold:    minimum confidence score to include a label (default 0.7).

        Returns an empty list if no label clears the threshold — never raises.
        """
        text = query
        if card_context:
            text = f"{query}\n\n{card_context}"

        result = self._tagger(text, candidate_labels=self._label_descriptions, multi_label=True)
        # for label, score in zip(result["labels"], result["scores"]):
        #     print(f"LABEL: {label}\nSCORE: {score}")
        return [
            self._desc_to_name[label]
            for label, score in zip(result["labels"], result["scores"])
            if score >= threshold
        ]
