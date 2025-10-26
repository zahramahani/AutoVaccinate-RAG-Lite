import json
import time
import asyncio
def safe_invoke(llm, prompt, retries=5, delay=5):
    """
    Retry wrapper for Mistral API calls to handle temporary 429 / capacity errors.
    """
    for attempt in range(1, retries + 1):
        try:
            # if asyncio.iscoroutinefunction(llm._generate) or asyncio.iscoroutinefunction(llm.agenerate_text):
            #     response_text =llm.agenerate_text(prompt)
            # else:
            response_text = llm.invoke(prompt)
            return response_text
        except Exception as e:
            msg = str(e)
            if "capacity" in msg.lower() or "429" in msg:
                print(f"⚠️ [Attempt {attempt}/{retries}] Mistral capacity issue, retrying in {delay * attempt}s...")
                time.sleep(delay * attempt)
                # await asyncio.sleep(delay * attempt) 
            else:
                print(f"❌ Mistral error (non-retryable): {e}")
                raise
    print("🚫 Mistral API unavailable after multiple retries — skipping call.")
    return None  # fallback to None if all retries fail

class PromptPatcher:
    def __init__(self, config_path="configs/prompts.json"):
        """Load available prompt templates from a JSON config."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)
        self.active_id = "default"

    def use(self, prompt_id):
        """Switch active prompt template."""
        self.active_id = prompt_id if prompt_id in self.prompts else "default"

    def format(self, query, retrieved_docs, kg_text=""):
        """
        Build the full LLM prompt using the current template.
        - query: the user question
        - retrieved_docs: list of {"text": "..."} or list of plain strings
        - kg_text: string representation of KG facts
        """
        # Normalize retrieved_docs
        retrieved_texts = []
        for d in retrieved_docs:
            if isinstance(d, dict) and "text" in d:
                retrieved_texts.append(d["text"])
            else:
                retrieved_texts.append(str(d))

        # Select the active template
        template = self.prompts.get(
            self.active_id,
            "Context:\n{kg}\n\nRetrieved passages:\n{retrieved}\n\nQuestion: {query}\nAnswer:",
        )

        # Format according to your structured style
        prompt = template.format(
            kg=kg_text.strip(),
            retrieved="\n\n".join(retrieved_texts),
            query=query.strip(),
        )
        return prompt

    def run_with_model(self, llm, query, retrieved_docs, kg_text=""):
        """
        Combine prompt formatting and LLM invocation in one step.
        """
        prompt = self.format(query, retrieved_docs, kg_text)
        response = safe_invoke(llm, prompt)
        return response
