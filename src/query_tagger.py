"""
Few-shot query tagger for classifying incoming MTG rules queries.

Usage:
    from query_tagger import QueryTagger, SYSTEM_LABELS
    from llm import LLMClient

    examples = [
        {"query": "Can I attack with a tapped creature?", "tags": ["combat"]},
        {"query": "Does this triggered ability use the stack?", "tags": ["abilities", "priority"]},
        ...
    ]

    tagger = QueryTagger(examples, llm_client)

    query_context = query_processor.extract_context(query)
    tags = tagger.tag(query_context["cleaned_query"], query_context["card_context"])
    # e.g. ["mana", "priority"]
"""

import json
import re

# Single source of truth for the tag vocabulary.
# slop.py's system_labels instance variable should match this list.
SYSTEM_LABELS: list[str] = [
    "combat",
    "casting",
    "mana",
    "abilities",
    "state_based_actions",
    "continuous_effects",
    "priority",
]


class QueryTagger:
    def __init__(self, examples: list[dict], llm_client):
        """
        Args:
            examples:   list of {"query": str, "tags": list[str]} dicts.
                        Every tag in each example must appear in SYSTEM_LABELS.
            llm_client: a loaded (or lazy-loading) LLMClient instance.
        """
        self._validate_examples(examples)
        self._examples = examples
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag(self, query: str, card_context: str = "") -> list[str]:
        """
        Classify a query and return a list of matching system tags.

        Args:
            query:        the cleaned query text (card [[brackets]] already stripped).
            card_context: optional oracle text / rulings block from QueryProcessor.
                          When provided, the model uses it to identify which game
                          systems the referenced cards interact with.

        Returns an empty list if the model output cannot be parsed or no
        tags match the known vocabulary — never raises.
        """
        prompt = self._build_prompt(query, card_context)
        raw = self._llm.classify(prompt)
        return self._parse_tags(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_examples(self, examples: list[dict]) -> None:
        """Raise ValueError on construction if any example tag is unknown."""
        for i, ex in enumerate(examples):
            bad = [t for t in ex.get("tags", []) if t not in SYSTEM_LABELS]
            if bad:
                raise ValueError(
                    f"Example {i} contains unknown tags {bad}. "
                    f"Valid tags are: {SYSTEM_LABELS}"
                )

    def _build_prompt(self, query: str, card_context: str = "") -> str:
        """
        Format the few-shot user message.

        Card context is placed immediately before the query being classified,
        NOT before the examples. Putting it at the top would prime the model
        on card mechanics before it has seen the task pattern, causing it to
        tag based on what the cards do rather than what the question asks.

        Structure:
            ### Examples
            Query: "..."
            Tags: [...]

            ...

            ### Classify
            Card Information:       <- only present when card_context is non-empty
            <oracle text / rulings>

            Query: "..."
            Tags:
        """
        lines: list[str] = []

        lines.append("### Examples")
        for ex in self._examples:
            lines.append(f'Query: "{ex["query"]}"')
            lines.append(f"Tags: {json.dumps(ex['tags'])}")
            lines.append("")

        lines.append("### Classify")
        if card_context:
            lines.append("Card Information:")
            lines.append(card_context)
            lines.append("")
        lines.append(f'Query: "{query}"')
        lines.append("Tags:")
        return "\n".join(lines)

    def _parse_tags(self, raw: str) -> list[str]:
        """
        Extract a JSON array from the raw LLM output and filter it to only
        known SYSTEM_LABELS.  Returns [] on any parse failure.
        """
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            tags = json.loads(match.group())
        except json.JSONDecodeError:
            return []

        return [t for t in tags if isinstance(t, str) and t in SYSTEM_LABELS]
