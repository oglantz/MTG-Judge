from dataclasses import dataclass
import re
import pathlib
import json
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
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
        self._fullRuleText = ""
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
        self.system_labels = [("combat", "This Magic: The Gathering rule is related to combat"), 
                              ("casting", "This Magic: The Gathering rule is related to casting"), 
                              ("mana", "This Magic: The Gathering rule is related to mana"), 
                              ("abilities", "This Magic: The Gathering rule is related to abilities"), 
                              ("state-based actions", "This Magic: The Gathering rule is related to state-based actions"), 
                              ("continuous effects", "This Magic: The Gathering rule is related to continuous effects"), 
                              ("priority", "This Magic: The Gathering rule is related to priority"),
                              ("stack", "This Magic: The Gathering rule is related to the stack")]

    def buildDocuments(self):
        print("STARTING")
        documents = []
        ruleMap = {}
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
                    self._fullRuleText += line
                elif match := re.match(r"[0-9]{3}\.[0-9]+", line):
                    if self._fullRuleText:
                        ruleMap[self._ruleCode] = self._fullRuleText
                        self._fullRuleText = ""
                    self._state = "rule"
                    self._rule = line
                    self._fullRuleText += line
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
                    self._fullRuleText += line

        # Add last document
        document = self._buildDocument()
        if document is not None:
            documents.append(document)

        self.reset()
        print("YOOHOO!!")

        # --- Batch auto-tagging for efficiency ---
        if documents:
            all_texts = [doc.text for doc in documents]

            # unpack tuples to get the descriptions
            label_descriptions = [desc for _, desc in self.system_labels]
            # map the descriptions to the tag names
            desc_to_name = {desc: name for name, desc in self.system_labels}

            CHUNK_SIZE = 64
            system_results = []
            for i in range(0, len(all_texts), CHUNK_SIZE):
                batch = all_texts[i:i + CHUNK_SIZE]
                batch_results = self.tagger(batch, candidate_labels=label_descriptions, multi_label=True, batch_size=CHUNK_SIZE)
                system_results.extend(batch_results)
                print(f"Tagged {min(i + CHUNK_SIZE, len(all_texts))} / {len(all_texts)} documents")
        
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    # Map the returned description back to its tag name
                    "system": [desc_to_name[label] for label, score in zip(system_results[i]['labels'], system_results[i]['scores']) if score > 0.6],
                })
        
        tagged = [d for d in documents if d.metadata.get("system")]
        print(f"{len(tagged)} / {len(documents)} documents have at least one system tag")

        # Save a record of each rule/subrule code mapped to its assigned tags
        eval_dir = pathlib.Path("eval")
        eval_dir.mkdir(parents=True, exist_ok=True)
        tagger_eval = {
            (doc.metadata.get("subrule_code") or doc.metadata.get("rule_code")): doc.metadata.get("system", [])
            for doc in documents
            if doc.metadata.get("subrule_code") or doc.metadata.get("rule_code")
        }
        eval_path = eval_dir / "tagger_eval.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(tagger_eval, f, indent=2)
        print(f"Tagger eval saved to {eval_path}")

        print(f"RULE MAP:\n{ruleMap}")
        return documents, ruleMap

    # LEGACY: 
    # def auto_tag(self, text):
    #     """Auto-tag a document using zero-shot classification (legacy, not used in batch)."""
    #     print("Starting auto-tag")
    #     system = self.tagger(text, candidate_labels=self.system_labels, multi_label=True)
    #     print("System tagged")

    #     return {
    #         "system": [label for label, score in zip(system['labels'], system['scores']) if score > 0.5],
    #     }

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
        self._fullRuleText = ""

class RAG:
    def __init__(self, vecStore: VectorStoreIndex, ruleMap: dict):
        self.vecStore = vecStore
        self.ruleMap = ruleMap
        self.retriever = vecStore.as_retriever()
        self.retriever.similarity_top_k = 10 # We can fuck with this later

    def ragSearch(self, query: str):
        output = ""
        results = list(self.retriever.retrieve(query))

        # for i, result in enumerate(results):
        #     print(f"\nResult {i + 1}:")
        #     print("Text:", result.text)
        #     print("Metadata:", result.metadata)

        for result in results[3:]:
            output += result.text + "\n"

        for result in results[0:3]: # get the top three
            output += self.ruleMap[result.metadata["rule_code"]]

        # print(f"\n\nOUTPUT:\n\n")
        # print(output)
        return output



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

PERSIST_DIR = pathlib.Path("../storage/slop_index")
FAISS_INDEX_PATH = PERSIST_DIR / "faiss.index"
RULE_MAP_PATH = PERSIST_DIR / "rule_map.json"


def build_or_load_index(rules_file: str, persist_dir: pathlib.Path = PERSIST_DIR):
    """Load a persisted FAISS index + ruleMap if available, otherwise build and save."""
    model = HuggingFaceEmbedding(model_name="all-mpnet-base-v2")
    Settings.llm = None

    faiss_path = persist_dir / "faiss.index"
    rule_map_path = persist_dir / "rule_map.json"

    if faiss_path.exists() and rule_map_path.exists():
        print("Loading persisted index from disk...")
        f_index = faiss.read_index(str(faiss_path))
        vstore = FaissVectorStore(faiss_index=f_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vstore,
            persist_dir=str(persist_dir)
        )
        index = load_index_from_storage(storage_context, embed_model=model)
        with open(rule_map_path, "r", encoding="utf-8") as f:
            ruleMap = json.load(f)
        print("Index loaded from disk.")
    else:
        print("No persisted index found — building from scratch...")
        parser = RuleParser(rules_file)
        documents, ruleMap = parser.buildDocuments()

        f_index = faiss.IndexFlatL2(768)
        vstore = FaissVectorStore(faiss_index=f_index)
        storage_context = StorageContext.from_defaults(vector_store=vstore)
        index = VectorStoreIndex.from_documents(
            documents, embed_model=model, storage_context=storage_context
        )

        persist_dir.mkdir(parents=True, exist_ok=True)
        storage_context.persist(persist_dir=str(persist_dir))
        faiss.write_index(f_index, str(faiss_path))
        with open(rule_map_path, "w", encoding="utf-8") as f:
            json.dump(ruleMap, f)
        print(f"Index persisted to {persist_dir}")

    return index, ruleMap

DB_SOURCE = "../rsrc/rulestext.txt"
def get_rules(query: str) -> str:
    print("starting rag")
    index, ruleMap = build_or_load_index("../rsrc/rulestext.txt")
    rag = RAG(index, ruleMap)
    print("returning rules...")
    return rag.ragSearch(query)


