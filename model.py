import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = Path(__file__).resolve().parent / "EduLLM_Final"
SYSTEM_PROMPT = (
    "You are EduLLM, a computer science tutor. Explain concepts in simple "
    "language, use examples where helpful, and acknowledge uncertainty."
)


class EduLLM:
    """Loads the local LoRA adapter and generates educational responses."""

    def __init__(self) -> None:
        config_path = ADAPTER_PATH / "adapter_config.json"
        weights_path = ADAPTER_PATH / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"The LoRA adapter is incomplete. Expected both "
                f"{config_path} and {weights_path}."
            )

        try:
            with config_path.open(encoding="utf-8") as config_file:
                adapter_config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Could not read the LoRA adapter configuration at {config_path}: "
                f"{error}"
            ) from error

        adapter_base = adapter_config.get("base_model_name_or_path")
        if adapter_base != BASE_MODEL:
            raise ValueError(
                f"The adapter expects base model {adapter_base!r}, but this app "
                f"is configured for {BASE_MODEL!r}."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            if not self.tokenizer.chat_template:
                raise RuntimeError(
                    "The Qwen tokenizer did not provide a chat template."
                )

            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            self.model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
            self.model.to(self.device)
            self.model.eval()
        except Exception as error:
            raise RuntimeError(
                f"Could not load {BASE_MODEL} with the EduLLM LoRA adapter. "
                "Check your internet connection, available memory, and adapter "
                f"files. Details: {error}"
            ) from error

    def generate(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        max_new_tokens: int = 200,
    ) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a question before generating a response.")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        if history:
            messages.extend(
                message
                for message in history[-8:]
                if message.get("role") in {"user", "assistant"}
                and isinstance(message.get("content"), str)
            )
        messages.append({"role": "user", "content": question.strip()})

        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {
                name: value.to(self.device)
                for name, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
            input_length = inputs["input_ids"].shape[-1]
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id

            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )

            answer = self.tokenizer.decode(
                output[0][input_length:],
                skip_special_tokens=True,
            ).strip()
            if not answer:
                raise RuntimeError("The model returned an empty response.")
            return answer
        except Exception as error:
            raise RuntimeError(f"EduLLM could not generate a response: {error}") from error


def load_model() -> EduLLM:
    """Create the model instance; callers should cache it for their process."""
    return EduLLM()
