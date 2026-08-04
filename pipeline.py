"""
flux2tiny — Inference pipeline.

Wraps diffusers' Flux2KleinPipeline, replacing the Qwen3-4B text encoder
with a student encoder + trained projection adapter.

Supports multi-GPU inference and auto fp16/bf16 detection.
"""

from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2

from config import get_student_config, StudentConfig, get_default_dtype


class Flux2TinyPipeline:
    """
    Student text encoder → adapter → Flux2KleinPipeline → image.

    Architecture:
      prompt → student tokenizer → student LM (frozen, extract 3 layers)
        → PerLayerProjection adapter → [B, seq, 7680] prompt_embeds
        → Flux2KleinPipeline (FLUX.2 Transformer + VAE) → PIL Image
    """

    def __init__(
        self,
        config: Union[str, StudentConfig] = "configs/minicpm5-1b.json",
        flux_model_id: Optional[str] = None,
        vae_model_id: Optional[str] = None,
        student_model_id: Optional[str] = None,
        adapter_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        adapter_type: str = "per_layer",
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        cpu_offload: bool = True,
        multi_gpu: bool = False,
    ):
        self.student_config = get_student_config(config) if isinstance(config, str) else config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or get_default_dtype()
        self.cpu_offload = cpu_offload
        self.multi_gpu = multi_gpu or (torch.cuda.is_available() and torch.cuda.device_count() > 1)
        self.student_extract_layers = self.student_config.extract_layers

        cfg = self.student_config
        flux_model_id = flux_model_id or cfg.teacher_model_id
        vae_model_id = vae_model_id or cfg.vae_model_id
        student_model_id = student_model_id or cfg.student_model_id

        if adapter_path is None:
            adapter_path = cfg.get_adapter_path("adapter_best.safetensors")
            if not Path(adapter_path).exists():
                adapter_path = cfg.get_adapter_path("adapter_final.safetensors")

        if lora_path is None:
            lora_path = cfg.get_lora_path("final/transformer_lora")

        print(f"=== Loading flux2tiny ({cfg.name}) ===")
        print(f"  Dtype: {self.dtype} | Multi-GPU: {self.multi_gpu}")

        self._load_text_encoder(student_model_id)
        self._load_adapter(adapter_path, adapter_type)
        self._load_flux(flux_model_id, vae_model_id, lora_path)

        print("=== Pipeline ready ===")

    def _load_text_encoder(self, model_id: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Text encoder: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=self.dtype, trust_remote_code=True,
        )
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        if self.cpu_offload:
            self.text_encoder.to("cpu")
            self._encoder_device = torch.device("cpu")
        else:
            self.text_encoder.to(self.device)
            self._encoder_device = self.device

    def _load_adapter(self, path: str, adapter_type: str):
        from adapter import load_adapter

        print(f"  Adapter: {path}")
        self.adapter = load_adapter(
            path, adapter_type=adapter_type,
            source_dim=self.student_config.hidden_size,
            target_dim=self.student_config.teacher_hidden_size,
            num_layers=self.student_config.num_layers,
            device=str(self.device), dtype=self.dtype,
        )

    def _load_flux(self, flux_id: str, vae_id: str, lora_path: Optional[str]):
        print(f"  FLUX.2: {flux_id}")
        vae = AutoencoderKLFlux2.from_pretrained(vae_id, torch_dtype=self.dtype)

        kwargs = {"vae": vae, "text_encoder": None, "tokenizer": None, "torch_dtype": self.dtype}
        if self.multi_gpu:
            kwargs["device_map"] = "balanced"

        self.pipe = Flux2KleinPipeline.from_pretrained(flux_id, **kwargs)

        if lora_path and Path(lora_path).exists():
            from peft import PeftModel
            print(f"  LoRA: {lora_path}")
            self.pipe.transformer = PeftModel.from_pretrained(self.pipe.transformer, lora_path)

        if self.cpu_offload and not self.multi_gpu:
            self.pipe.enable_model_cpu_offload()

    def encode_prompt(self, prompt: Union[str, list[str]], max_length: int = 128) -> torch.Tensor:
        """Encode prompt(s) → [B, seq, 7680] prompt_embeds."""
        if isinstance(prompt, str):
            prompt = [prompt]

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_length,
        ).to(self._encoder_device)

        with torch.no_grad():
            outputs = self.text_encoder(**inputs, output_hidden_states=True, return_dict=True)

        hidden_list = [
            outputs.hidden_states[i + 1].to(device=self.device, dtype=self.dtype)
            for i in self.student_extract_layers
        ]

        with torch.no_grad():
            return self.adapter(hidden_list)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        max_seq_len: int = 128,
    ) -> Image.Image:
        """Generate an image from a text prompt."""
        prompt_embeds = self.encode_prompt(prompt, max_length=max_seq_len)

        output = self.pipe(
            prompt=None, prompt_embeds=prompt_embeds,
            height=height, width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return output.images[0]
