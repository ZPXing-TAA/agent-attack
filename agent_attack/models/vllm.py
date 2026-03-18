"""
vllm.py

OpenAI-compatible wrapper for locally deployed multimodal models served by vLLM.
"""

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from accelerate import PartialState
from openai import OpenAI
from PIL import Image

from agent_attack.util.interfaces import VLM


def encode_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class VLLM(VLM):
    def __init__(
        self,
        hub_path: Path,
        max_length: int = 512,
        temperature: float = 0.0,
        **_: str,
    ) -> None:
        raw_hub_path = str(hub_path)
        if raw_hub_path.startswith("vllm:"):
            self.hub_path = raw_hub_path.split("vllm:", maxsplit=1)[1]
        elif raw_hub_path.startswith("vllm/"):
            self.hub_path = raw_hub_path.split("vllm/", maxsplit=1)[1]
        else:
            self.hub_path = raw_hub_path

        self.distributed_state = PartialState()

        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {
            "max_tokens": self.max_length,
            "temperature": self.temperature,
            "logprobs": True,
            "top_logprobs": int(os.getenv("VLLM_TOP_LOGPROBS", "20")),
        }

        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.last_logprobs = []

    def set_generate_kwargs(self, generate_kwargs):
        self.generate_kwargs = generate_kwargs

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> Any:
        raise NotImplementedError("Prompt builder is not implemented for VLLM wrapper.")

    def get_captioning_prompt_fn(self) -> Callable[[str], str]:
        def captioning_prompt_fn() -> str:
            return "Provide a short image description in one sentence (e.g., the color, the object, the activity, and the location)."

        return captioning_prompt_fn

    @torch.inference_mode()
    def generate_answer(
        self,
        images: List[Image.Image],
        question_prompts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], Tuple[List[str], List[List[float]]]]:
        # Keep the signature aligned with the VLM protocol.
        # For vLLM, we always request token-level logprobs from the server and cache them in
        # `self.last_logprobs` for downstream consumers.
        _ = return_string_probabilities

        generate_kwargs = self.generate_kwargs.copy()
        if temperature is not None:
            generate_kwargs["temperature"] = temperature

        responses = []
        self.last_logprobs = []
        for image, question_prompt in zip(images, question_prompts, strict=True):
            base64_image = encode_image(image)
            content = [
                {"type": "text", "text": question_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                    },
                },
            ]

            completion = self.client.chat.completions.create(
                model=self.hub_path,
                messages=[{"role": "user", "content": content}],
                **generate_kwargs,
            )
            responses.append(completion.choices[0].message.content)
            choice_logprobs = None
            if completion.choices and hasattr(completion.choices[0], "logprobs"):
                choice_logprobs = completion.choices[0].logprobs
            self.last_logprobs.append(choice_logprobs)

        return responses
