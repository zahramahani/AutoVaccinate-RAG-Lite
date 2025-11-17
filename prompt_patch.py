#prompt_patch.py
"""
prompt_patch.py
Enhanced prompt patcher with API token tracking.
"""

import json
import time
import logging

logger = logging.getLogger("PromptPatcher")


def safe_invoke(llm, prompt, retries=5, delay=5):
    """
    Retry wrapper for Mistral API with token tracking.
    Returns: (response, token_count)
    """
    for attempt in range(1, retries + 1):
        try:
            response = llm.invoke(prompt)
            
            # Extract token usage if available
            token_count = 0
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('usage', {})
                token_count = usage.get('total_tokens', 0)
            
            logger.debug(f"✅ API call successful: {token_count} tokens")
            return response, token_count
            
        except Exception as e:
            msg = str(e)
            if "capacity" in msg.lower() or "429" in msg:
                logger.warning(
                    f"⚠️ [Attempt {attempt}/{retries}] Mistral capacity issue, "
                    f"retrying in {delay * attempt}s..."
                )
                time.sleep(delay * attempt)
            else:
                logger.error(f"❌ Mistral error (non-retryable): {e}")
                raise
    
    logger.error("🚫 Mistral API unavailable after multiple retries")
    return None, 0


class PromptPatcher:
    def __init__(self, config_path="configs/prompts.json"):
        """Load available prompt templates from a JSON config."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)
        self.active_id = "default"
        logger.info(f"📋 Loaded {len(self.prompts)} prompt templates")

    def use(self, prompt_id):
        """Switch active prompt template."""
        if prompt_id in self.prompts:
            self.active_id = prompt_id
            logger.debug(f"🔄 Switched to prompt: {prompt_id}")
        else:
            logger.warning(f"⚠️ Unknown prompt_id '{prompt_id}', using 'default'")
            self.active_id = "default"

    def format(self, query, retrieved_docs, kg_text=""):
        """
        Build the full LLM prompt using the current template.
        """
        # Normalize retrieved_docs
        retrieved_texts = []
        for d in retrieved_docs:
            if isinstance(d, dict) and "text" in d:
                retrieved_texts.append(d["text"])
            elif hasattr(d, 'page_content'):
                retrieved_texts.append(d.page_content)
            else:
                retrieved_texts.append(str(d))

        # Select template
        template = self.prompts.get(
            self.active_id,
            "Context: {kg} Retrieved passages: {retrieved} Question: {query} Answer:",
        )

        # Format
        prompt = template.format(
            kg=kg_text.strip(),
            retrieved="".join(retrieved_texts),
            query=query.strip(),
        )
        return prompt

    def run_with_model(self, llm, query, retrieved_docs, kg_text=""):
        """
        Combine prompt formatting and LLM invocation.
        Returns: (response, token_count)
        """
        prompt = self.format(query, retrieved_docs, kg_text)
        response, token_count = safe_invoke(llm, prompt)
        return response, token_count