from dataclasses import dataclass
import re
import pathlib
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import pipeline
import torch
import faiss

class RuleParser:
    def __init__(self, filename, device=None):
        self._filename = filename

        self._section = ""
        self._sectionCode = ""
        self._topic = ""
        self._topicCode = ""
        self._rule = ""
        self._ruleCode = ""
        self._subruleCode = ""
        self._finalText = ""
        self._state = ""

        # Auto-detect GPU if available
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        # HuggingFace zero-shot pipeline for auto-tagging
        self.tagger = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )

        # Candidate labels for tagging
        self.decision_role_labels = ["restriction", "procedure", "triggers", "consequence", "definition"]
        self.system_labels = ["combat", "casting", "mana", "triggered_abilities", "activated_abilities", "state_based_actions", "continuous_effects", "priority"]

    def buildDocuments(self):
        print("STARTING")
        documents = []
        source = pathlib.Path(self._filename)

        with open(source, "r", encoding="utf-8") as tablefile:
            for line in tablefile.readlines():
                if line == "\n":
                    document = self._buildDocument()
                    if document is not None:
                        documents.append(document)  # append first, tagging later
                elif match := re.match(r"[0-9]{3}\.[0-9]+[a-z]", line):
                    self._state = "subrule"
                    self._subruleCode = match.group()
                    self._finalText = line
                elif match := re.match(r"[0-9]{3}\.[0-9]+", line):
                    self._state = "rule"
                    self._rule = line
                    self._ruleCode = match.group()
                    self._finalText = line
                elif match := re.match(r"[0-9]{3}", line):
                    self._state = "topic"
                    self._topic = line
                    self._topicCode = match.group()
                elif match := re.match(r"[0-9]", line):
                    self._state = "section"
                    self._section = line
                    self._sectionCode = match.group()
                elif re.match(r"Example: ", line):
                    self._finalText += line

        # Add last document
        document = self._buildDocument()
        if document is not None:
            documents.append(document)

        self.reset()
        print("YOOHOO!!")

        # --- Batch auto-tagging for efficiency ---
        if documents:
            all_texts = [doc.text for doc in documents]

            decision_results = self.tagger(all_texts, candidate_labels=self.decision_role_labels, multi_label=True)
            system_results = self.tagger(all_texts, candidate_labels=self.system_labels, multi_label=True)

            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "decision_role": [label for label, score in zip(decision_results[i]['labels'], decision_results[i]['scores']) if score > 0.7],
                    "system": [label for label, score in zip(system_results[i]['labels'], system_results[i]['scores']) if score > 0.7],
                })

        return documents

    def auto_tag(self, text):
        """Auto-tag a document using zero-shot classification (legacy, not used in batch)."""
        print("Starting auto-tag")
        decision = self.tagger(text, candidate_labels=self.decision_role_labels, multi_label=True)
        print("Decisions tagged")
        system = self.tagger(text, candidate_labels=self.system_labels, multi_label=True)
        print("System tagged")

        return {
            "decision_role": [label for label, score in zip(decision['labels'], decision['scores']) if score > 0.5],
            "system": [label for label, score in zip(system['labels'], system['scores']) if score > 0.5],
        }

    def _buildDocument(self):
        if self._state in ["section", "topic"]:
            return None
        if self._state == "rule":
            return self._buildRuleDocument()
        if self._state == "subrule":
            return self._buildSubruleDocument()
        raise Exception("Invalid state.")

    def _buildRuleDocument(self):
        metadata = {
            "type": "rule",
            "section_code": self._sectionCode,
            "section": self._section,
            "topic_code": self._topicCode,
            "topic": self._topic,
            "rule_code": self._ruleCode
        }
        return Document(text=self._finalText, metadata=metadata)

    def _buildSubruleDocument(self):
        metadata = {
            "type": "subrule",
            "section_code": self._sectionCode,
            "section": self._section,
            "topic_code": self._topicCode,
            "topic": self._topic,
            "rule_code": self._ruleCode,
            "subrule_code": self._subruleCode
        }
        return Document(text=self._finalText, metadata=metadata)

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
    print("Loading parser...")
    parser = RuleParser("rule_src/rulestext.txt")  # GPU auto-detected
    documents = parser.buildDocuments()
    print(f"{len(documents)} documents parsed and auto-tagged.")

    # Embedding model
    model = HuggingFaceEmbedding(model_name="all-mpnet-base-v2")
    print("Embedding model loaded.")

    # FAISS setup
    f_index = faiss.IndexFlatL2(768)
    Settings.llm = None
    vstore = FaissVectorStore(faiss_index=f_index)

    # Build index
    index = VectorStoreIndex.from_documents(documents, embed_model=model, vector_store=vstore)
    retriever = index.as_retriever()
    retriever.similarity_top_k = 20

    # Manual test query
    test_text = "Can I attack with a tapped creature?"
    test_metadata = {
        "decision_role": ["restriction", "procedure"],
        "system": ["combat"],
    }
    test_doc = Document(text=test_text, metadata=test_metadata)
    results = retriever.retrieve(test_text)

    print("\nTest Query Results (manual tags applied):")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print("Text:", result.text)
        print("Metadata:", result.metadata)
    print("___________\n\n\n\n\n\n\n\n\n_____________________")
    test_text = "The active player chooses which creatures that they control, if any, will attack. The chosen creatures must be untapped, they can’t also be battles, and each one must either have haste or have been controlled by the active player continuously since the turn began."
    test_metadata = {}
    test_doc = Document(text=test_text, metadata=test_metadata)
    results = retriever.retrieve(test_text)
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print("Text:", result.text)
        print("Metadata:", result.metadata)

if __name__ == "__main__":
    main()
