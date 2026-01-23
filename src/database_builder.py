from dataclasses import dataclass
import re
import pathlib
import faiss

@dataclass
class Document:
    _text: str
    _metadata: dict

    def getText(self):
        return self._text

    def getMetadata(self):
        return self._metadata
class RuleParser:
    def __init__(self, filename):
        self._filename = filename

        # Update these in the main loop for proper metadata
        self._section = ""  # 5. Turn Structure
        self._sectionCode = "" # 5
        self._topic = ""  # 501. Beginning Phase
        self._topicCode = "" # 501
        self._rule = ""  # 502.2. Second, if it’s day...
        self._ruleCode = "" # 502.2
        self._subruleCode = "" # 502.2a
        self._finalText = "" # 502.2a Multiplayer games...
        self._state = ""

    def buildDocuments(self):
        documents = []
        source = pathlib.Path(self._filename)
        with open(source, "r", encoding="utf-8") as tablefile:
            for line in tablefile.readlines():
                if line == "\n":
                    document = self._buildDocument()
                    if document is not None:
                        documents.append(document)

                elif match := re.match(r"[0-9]{3}\.[0-9]+[a-z]", line): # Lettered subrules only go to z. Safe for now...
                    self._state = "subrule"
                    self._subruleCode = match.group()
                    self._finalText = line


                elif match := re.match(r"[0-9]{3}\.[0-9]+", line): # Rules
                    self._state = "rule"
                    self._rule = line
                    self._ruleCode = match.group()
                    self._finalText = line

                elif match := re.match(r"[0-9]{3}", line): # Topics
                    self._state = "topic"
                    self._topic = line
                    self._topicCode = match.group()

                elif match := re.match(r"[0-9]", line): # Sections
                    self._state = "section"
                    self._section = line
                    self._sectionCode = match.group()

                elif match := re.match(r"Example: ", line): # Example additions
                    self._rule += line

        self.reset()
        return documents

    def _buildDocument(self):
        if self._state == "section" or self._state == "topic":
            return None
        if self._state == "rule":
            return self._buildRuleDocument()
        if self._state == "subrule":
            return self._buildSubruleDocument()

        raise Exception("Invalid state.")

    def _buildRuleDocument(self):
        metadata = {"type": "rule", "section_code": self._sectionCode, "section": self._section,
                    "topic_code": self._topicCode, "topic": self._topic,
                    "rule_code": self._ruleCode}
        return Document(self._finalText, metadata)

    def _buildSubruleDocument(self):
        metadata = {"type": "subrule", "section_code": self._sectionCode, "section": self._section,
                    "topic_code": self._topicCode, "topic": self._topic,
                    "rule_code": self._ruleCode, "subrule_code": self._subruleCode}
        return Document(self._finalText, metadata)

    def reset(self):
        self._section = ""
        self._sectionCode = ""
        self._topic = ""
        self._topicCode = ""
        self._rule = ""
        self._ruleCode = ""
        self._subruleCode = ""
        self._finalText = ""
        self._state = ""

def main():
    parser = RuleParser("rsrc/rulestext.txt")
    documents = parser.buildDocuments()
    [print(doc) for doc in documents]

if __name__ == "__main__":
    main()