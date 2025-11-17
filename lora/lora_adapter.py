# lora_adapter.py
"""
LoRA adapter manager for local generation.
- Supports CPU-only and GPU (QLoRA) inference.
- Provides LocalLLM with .invoke(prompt) → response and token accounting.
"""
import os
import math
import torch
from typing import Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

DEFAULT_BASE = os.environ.get("LORA_BASE_MODEL", "microsoft/phi-3-mini-4k-instruct")

class LocalLLM:
    def __init__(self, base_model_name: str = DEFAULT_BASE, lora_path: Optional[str] = None, device: Optional[str] = None):
        """
        :param base_model_name: HF model name for base
        :param lora_path: optional LoRA adapter directory (PEFT)
        :param device: "cuda" or "cpu" (auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

        # Inference-friendly load
        load_kwargs = {}
        if self.device == "cuda":
            # GPU path; can be 4-bit if base supports it
            try:
                import bitsandbytes as bnb  # noqa
                load_kwargs.update(dict(
                    device_map="auto",
                    torch_dtype=torch.float16
                ))
            except Exception:
                load_kwargs.update(dict(
                    device_map="auto",
                    torch_dtype=torch.float16
                ))
        else:
            # CPU path
            load_kwargs.update(dict(
                torch_dtype=torch.float32,
                device_map={"": "cpu"}
            ))

        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)

        # Attach LoRA if provided
        if lora_path and os.path.isdir(lora_path):
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.eval()
        self.streamer = None  # optional: TextStreamer(self.tokenizer)

    def invoke(self, prompt: str, max_new_tokens=256, temperature=0.2, top_p=0.9) -> Dict:
        """
        Minimal interface to mimic LangChain .invoke() contract used in PromptPatcher.safe_invoke.
        Returns an object with .content and .response_metadata. Token count approximated via tokenizer.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        total_in = input_ids.shape[-1]

        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature,
                top_p=top_p,
                streamer=self.streamer
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Extract only the generated continuation after prompt
        gen_text = text[len(self.tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
        total_out = len(self.tokenizer(gen_text).input_ids)

        class Response:
            def __init__(self, content, total_tokens):
                self.content = content
                self.response_metadata = {
                    "usage": {
                        "total_tokens": total_in + total_out,
                        "prompt_tokens": total_in,
                        "completion_tokens": total_out
                    },
                    "generator": "local_lora"
                }
        return Response(gen_text.strip(), total_in + total_out)

def load_local_llm(base_model=DEFAULT_BASE, lora_dir=None) -> LocalLLM:
    return LocalLLM(base_model_name=base_model, lora_path=lora_dir)
