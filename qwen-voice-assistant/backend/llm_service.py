"""
LLM Service for Qwen3 model inference with LoRA adapter.
Supports CPU-only mode with quantization for laptops without GPU.
"""

import os
import re
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class LLMService:
    def __init__(
        self,
        model_path: str = None,
        adapter_path: str = None,
        max_history: int = 6,
        use_cpu: bool = None,
        use_4bit: bool = None,
    ):
        self.model_path = model_path or os.environ.get(
            "MODEL_PATH", "Qwen/Qwen3-4B-Instruct-2507"
        )
        self.adapter_path = adapter_path or os.environ.get(
            "LORA_ADAPTER_PATH", "../qwen_lora"
        )
        self.max_history = max_history
        self.conversations: Dict[str, List[dict]] = {}

        # Auto-detect: use CPU if no CUDA available
        if use_cpu is None:
            self.use_cpu = not torch.cuda.is_available()
        else:
            self.use_cpu = use_cpu

        if use_4bit is None:
            env_4bit = os.environ.get("USE_4BIT", "false").strip().lower()
            use_4bit = env_4bit in {"1", "true", "yes", "on"}
        self.use_4bit = use_4bit and not self.use_cpu  # 4-bit only works with GPU
        self.max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "120"))
        self.temperature = float(os.environ.get("TEMPERATURE", "0.6"))
        self.top_p = float(os.environ.get("TOP_P", "0.9"))

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model: {self.model_path}")
        print(f"Device: {'CPU' if self.use_cpu else 'CUDA'}")

        try:
            if self.use_cpu:
                # CPU mode: Use smaller model or accept slower inference
                print("Running on CPU - inference will be slower")
                print("Consider using a smaller model like Qwen/Qwen3-1.7B-Instruct")

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="cpu",
                    torch_dtype=torch.float32,  # CPU works better with float32
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            elif self.use_4bit:
                # GPU with 4-bit quantization (saves VRAM)
                print("Loading with 4-bit quantization")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                )
            else:
                # GPU without quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )

            if os.path.exists(self.adapter_path):
                print(f"Loading LoRA adapter: {self.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            else:
                print("No LoRA adapter found, using base model")

            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Running in mock mode for UI testing")
            self.model = None
            self.tokenizer = None

    @staticmethod
    def _clean_reply(text: str) -> str:
        text = text.strip()
        text = re.sub(r"<\|.*?\|>", "", text)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def chat(
        self,
        user_id: str,
        message: str,
        system_prompt: str,
    ) -> Tuple[str, List[dict]]:
        if user_id not in self.conversations:
            self.conversations[user_id] = []

        conversation = self.conversations[user_id]

        if self.model is None:
            reply = f"[Mock Response] I understand you said: '{message}'. I'm here to help!"
            conversation.append({"role": "user", "content": message})
            conversation.append({"role": "assistant", "content": reply})
            return reply, conversation

        working_conversation = [
            {"role": "system", "content": system_prompt}
        ] + conversation + [
            {"role": "user", "content": message}
        ]

        formatted = self.tokenizer.apply_chat_template(
            working_conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0.0,
                use_cache=True,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        reply = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        reply = self._clean_reply(reply)

        conversation.append({"role": "user", "content": message})
        conversation.append({"role": "assistant", "content": reply})

        if len(conversation) > self.max_history:
            self.conversations[user_id] = conversation[-self.max_history:]

        return reply, self.conversations[user_id]

    def clear_conversation(self, user_id: str):
        if user_id in self.conversations:
            self.conversations[user_id] = []

    def is_loaded(self) -> bool:
        return self.model is not None
