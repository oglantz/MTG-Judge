"""
LLM interface for Qwen2.5-7B-Instruct.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# Set to a local directory path to load from disk instead of HuggingFace cache.
# After the first run, save the model locally with:
#   tokenizer.save_pretrained("models/qwen2.5-7b-instruct")
#   model.save_pretrained("models/qwen2.5-7b-instruct")
# Then point MODEL_ID at that path, e.g.: MODEL_ID = "models/qwen2.5-7b-instruct"

SYSTEM_PROMPT = """\
You are an expert Magic: The Gathering judge. Answer rules questions accurately \
and concisely, citing specific rules or card text when relevant.\
"""


class LLMClient:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self._model_id = model_id
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        print(f"Loading {self._model_id} on {self._device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            dtype=torch.float16 if self._device == "cuda" else torch.float32,
            device_map=self._device,
        )
        self._model.eval()
        print("Model loaded.")

    def generate(
        self,
        context: dict,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a response given the context dict produced by
        QueryProcessor.extract_context().

        Expected keys in 'context':
          - "cleaned_query"  : the user's question with card references resolved
          - "card_context"   : oracle text / rulings for referenced cards (may be empty)
          - "rules_context"  : relevant MTG rules passages (may be absent / empty)
        """
        self._load()

        cleaned_query: str = context.get("cleaned_query", "")
        card_context: str = context.get("card_context", "")
        rules_context: str = context.get("rules_context", "")

        user_content_parts = []
        if card_context:
            user_content_parts.append(f"### Card Information\n{card_context}")
        if rules_context:
            user_content_parts.append(f"### Relevant Rules\n{rules_context}")
        user_content_parts.append(f"### Question\n{cleaned_query}")
        user_content = "\n\n".join(user_content_parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"].to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        generated = output_ids[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
