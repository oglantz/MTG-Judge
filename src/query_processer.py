import re
from scryfall_client import ScryfallClient


class QueryProcessor:
    _CARD_PATTERN = re.compile(r"\[\[(.+?)\]\]")
    _CONSECUTIVE_SYMBOLS = re.compile(r"(\{[^}]+\})+")
    # Matches the cost portion before the first ":" in an activated ability.
    # Costs contain mana/tap symbols or common cost keywords.
    _ACTIVATED_ABILITY = re.compile(
        r"^((?:\{[^}]+\}|,\s*)+|(?:sacrifice|discard|pay|remove|exile)\b[^:]*)",
        re.IGNORECASE
    )

    _NUMBER_WORDS = {
        str(i): w for i, w in enumerate([
            "zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen",
        ])
    }

    _COLOR_NAMES = {
        "W": "white", "U": "blue", "B": "black",
        "R": "red",   "G": "green", "C": "colorless", "S": "snow",
    }

    def __init__(self):
        self._scryfall = ScryfallClient()

    def _clean_query(self, query: str) -> str:
        """Replaces [[Card Name]] references with just the card name."""
        return self._CARD_PATTERN.sub(lambda m: m.group(1), query)

    def extract_card_names(self, query: str) -> set[str]:
        return set(self._CARD_PATTERN.findall(query))

    def _convert_symbol(self, code: str) -> str:
        if code == "T":
            return "tap"
        if code == "Q":
            return "untap"
        if code == "E":
            return "one energy"
        if code in self._COLOR_NAMES:
            return f"one {self._COLOR_NAMES[code]} mana"
        if code in self._NUMBER_WORDS:
            return f"{self._NUMBER_WORDS[code]} generic mana"
        if code.isdigit():
            return f"{code} generic mana"
        if code in ("X", "Y", "Z"):
            return f"{code} mana"
        if "/" in code:
            left, right = code.split("/", 1)
            if right == "P":
                color = self._COLOR_NAMES.get(left, left)
                return f"one {color} mana or 2 life"
            if left in self._NUMBER_WORDS or left.isdigit():
                num = self._NUMBER_WORDS.get(left, left)
                color = self._COLOR_NAMES.get(right, right)
                return f"{num} generic or one {color} mana"
            c1 = self._COLOR_NAMES.get(left, left)
            c2 = self._COLOR_NAMES.get(right, right)
            return f"one {c1} or {c2} mana"
        return code

    def _clean_text(self, text: str) -> str:
        def replace_group(match: re.Match) -> str:
            codes = re.findall(r"\{([^}]+)\}", match.group(0))
            return " and ".join(self._convert_symbol(c) for c in codes)

        text = self._CONSECUTIVE_SYMBOLS.sub(replace_group, text)
        text = text.encode("ascii", errors="ignore").decode("ascii")
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _format_oracle_text(self, oracle_text: str) -> str:
        """
        Formats oracle text paragraph by paragraph.
        Lines matching the "cost: effect" pattern of an activated ability are
        labeled explicitly so the LLM understands the structure.
        """
        paragraphs = oracle_text.split("\n")
        formatted = []

        for paragraph in paragraphs:
            keyword_label = ""
            text = paragraph
            em_dash_idx = paragraph.find(" \u2014 ")
            if em_dash_idx != -1:
                potential_keyword = paragraph[:em_dash_idx].strip()
                if not re.search(r"\{", potential_keyword):
                    keyword_label = self._clean_text(potential_keyword)
                    text = paragraph[em_dash_idx + 3:]

            colon_idx = text.find(":")
            if colon_idx != -1:
                raw_cost = text[:colon_idx].strip()
                if self._ACTIVATED_ABILITY.match(raw_cost):
                    cost = self._clean_text(raw_cost)
                    effect = self._clean_text(text[colon_idx + 1:].strip())
                    full_cost = f"{keyword_label}, {cost}" if keyword_label else cost
                    formatted.append(
                        f"Activated Ability:\n  Cost: {full_cost}\n  Effect: {effect}"
                    )
                    continue

            formatted.append(self._clean_text(paragraph))

        return "\n".join(formatted)

    def _fetch_all_card_infos(self, query: str) -> list[tuple[str, dict]]:
        """Fetch Scryfall data for every card mentioned in the query. Returns a list
        of (original_name, info_dict) pairs, preserving order."""
        return [
            (name, self._scryfall.get_card_info(name))
            for name in self.extract_card_names(query)
        ]

    def _format_card_context(self, card_infos: list[tuple[str, dict]], include_rulings: bool = True) -> str:
        """Format a list of (name, info) pairs into a context string.

        Args:
            card_infos:      output of _fetch_all_card_infos.
            include_rulings: when False, rulings are omitted (used for tagging).
        """
        sections = []

        for name, info in card_infos:
            if "error" in info:
                sections.append(f"[Card: {name}]\nNot found in Scryfall database.")
                continue

            lines = [f"[Card: {info['name']}]"]

            if info.get("oracle_text"):
                lines.append(f"Oracle Text:\n{self._format_oracle_text(info['oracle_text'])}")

            if include_rulings and info.get("has_rulings"):
                ruling_texts = [
                    f"- ({r['published_at']}) {self._clean_text(r['comment'])}"
                    for r in info["rulings"]
                ]
                lines.append("Rulings:\n" + "\n".join(ruling_texts))

            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def build_card_context(self, query: str) -> str:
        """
        Extracts card names from the query, fetches their oracle text and rulings,
        and returns a formatted string suitable for use as LLM context.
        """
        return self._format_card_context(self._fetch_all_card_infos(query), include_rulings=True)

    def extract_context(self, query: str) -> dict:
        """
        Extracts the cleaned query and card context from the query.

        Returns a dict with:
            cleaned_query:  query with [[Card Name]] brackets stripped.
            card_context:   oracle text + rulings for every mentioned card (for the LLM).
            oracle_context: oracle text only, no rulings (for the query tagger).
        """
        cleaned_query = self._clean_query(query)
        card_infos = self._fetch_all_card_infos(query)
        return {
            "cleaned_query": cleaned_query,
            "card_context":  self._format_card_context(card_infos, include_rulings=True),
            "oracle_context": self._format_card_context(card_infos, include_rulings=False),
        }




# q = QueryProcessor()
# print(q.extract_context("Alessandra controls [[Mox Opal]] and [[Trinisphere]]. Alessandra casts another [[Mox Opal]]. After it resolves, can they tap it for mana before it's put into their graveyard?")["card_context"])