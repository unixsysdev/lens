"""Model loading with nnsight wrapping."""

import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from typing import Optional
from rich.console import Console

from .utils import get_device, print_model_info, get_memory_info

console = Console()


class LensModel:
    """Wrapper around nnsight LanguageModel with convenience methods."""
    
    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = None,  # "4bit", "8bit", or None
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.device = device or get_device()
        
        console.print(f"[bold blue]Loading model:[/bold blue] {model_name}")
        
        # Setup quantization config
        quant_config = None
        torch_dtype = torch.float16
        config = None
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            config_dtype = getattr(config, "torch_dtype", None)
            resolved = self._resolve_torch_dtype(config_dtype)
            if resolved is not None:
                torch_dtype = resolved
        except Exception:
            config = None
        self.requested_dtype = torch_dtype
        
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            console.print("[dim]Using 4-bit quantization[/dim]")
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            console.print("[dim]Using 8-bit quantization[/dim]")
        
        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with nnsight wrapper
        # nnsight wraps the model and gives us tracing capabilities
        resolved_device_map = device_map
        explicit_none = False
        if resolved_device_map == "none":
            resolved_device_map = None
            explicit_none = True

        if resolved_device_map is None and not explicit_none:
            if quantization in {"4bit", "8bit"}:
                resolved_device_map = "auto"
            else:
                resolved_device_map = {"": self.device}
        elif isinstance(resolved_device_map, str):
            if resolved_device_map not in {"auto", "balanced", "sequential"}:
                resolved_device_map = {"": resolved_device_map}
        self.device_map = resolved_device_map

        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
        }
        if resolved_device_map is not None:
            load_kwargs["device_map"] = resolved_device_map
        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        self.model = LanguageModel(model_name, **load_kwargs)
        self.hf_model = None
        candidate = self._unwrap_envoy(self.model.model)
        if hasattr(candidate, "generate"):
            self.hf_model = candidate
        if self._model_is_meta() and quant_config is None:
            console.print("[yellow]Detected meta weights; reloading via transformers.[/yellow]")
            hf_kwargs = {
                "trust_remote_code": trust_remote_code,
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": False,
            }
            if resolved_device_map is not None:
                if self._has_accelerate():
                    hf_kwargs["device_map"] = resolved_device_map
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, **hf_kwargs)
            if "device_map" not in hf_kwargs and self.device != "cpu":
                hf_model.to(self.device)
            self.model = LanguageModel(hf_model, tokenizer=self.tokenizer)
            self.hf_model = hf_model
            if getattr(self.model, "config", None) is None:
                self.model.config = hf_model.config
        
        # Store config reference for convenience
        self.config = self._resolve_config()
        if getattr(self.model, "config", None) is None:
            self.model.config = self.config
        self._maybe_align_model_dtype(torch_dtype, resolved_device_map)
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        
        print_model_info(self.model, self.tokenizer)

    @staticmethod
    def _resolve_torch_dtype(value: Optional[object]) -> Optional[torch.dtype]:
        if value is None:
            return None
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"float16", "fp16", "half"}:
                return torch.float16
            if lowered in {"bfloat16", "bf16"}:
                return torch.bfloat16
            if lowered in {"float32", "fp32"}:
                return torch.float32
        return None

    def _collect_float_dtypes(self, module) -> dict:
        dtype_counts: dict[torch.dtype, int] = {}
        for param in module.parameters():
            if param is None or not param.is_floating_point():
                continue
            dtype_counts[param.dtype] = dtype_counts.get(param.dtype, 0) + param.numel()
        return dtype_counts

    def _device_supports_bf16(self) -> bool:
        try:
            device_type = torch.device(self.device).type
        except Exception:
            device_type = "cpu"
        if device_type == "cuda":
            return torch.cuda.is_bf16_supported()
        if device_type == "cpu":
            return True
        return False

    def _maybe_align_model_dtype(
        self,
        requested_dtype: Optional[torch.dtype],
        resolved_device_map: Optional[object],
    ) -> None:
        if self.quantization in {"4bit", "8bit"}:
            return
        module = self._unwrap_envoy(self.model.model)
        dtype_counts = self._collect_float_dtypes(module)
        if len(dtype_counts) <= 1:
            return
        target = None
        if requested_dtype is not None and requested_dtype in dtype_counts:
            target = requested_dtype
        elif torch.bfloat16 in dtype_counts and self._device_supports_bf16():
            target = torch.bfloat16
        elif torch.float16 in dtype_counts:
            target = torch.float16
        elif torch.float32 in dtype_counts:
            target = torch.float32
        else:
            target = next(iter(dtype_counts), None)
        if target is None:
            return
        if isinstance(resolved_device_map, str) and resolved_device_map in {"auto", "balanced", "sequential"}:
            console.print(
                "[yellow]Warning:[/yellow] Mixed precision weights detected but model is sharded; "
                "skipping dtype alignment. Consider --device-map none or cuda:0."
            )
            return
        dtype_list = ", ".join(str(dtype) for dtype in sorted(dtype_counts, key=lambda d: str(d)))
        console.print(
            f"[yellow]Mixed precision weights detected ({dtype_list}); casting to {target}.[/yellow]"
        )
        try:
            self.model.to(dtype=target)
        except Exception:
            module.to(dtype=target)
        if self.hf_model is not None:
            try:
                self.hf_model.to(dtype=target)
            except Exception:
                pass
        if self.config is not None:
            self.config.torch_dtype = target

    def _model_is_meta(self) -> bool:
        """Check if the underlying model is still on meta device."""
        try:
            module = self.get_hf_base_model()
            param = next(module.parameters(), None)
        except Exception:
            return False
        return bool(param is not None and getattr(param, "is_meta", False))

    def _resolve_config(self):
        config = getattr(self.model, "config", None)
        if config is None:
            inner = getattr(self.model, "model", None)
            config = getattr(inner, "config", None)
        if config is None:
            raise AttributeError("Model config is not available after load.")
        return config

    @staticmethod
    def _has_accelerate() -> bool:
        try:
            import accelerate  # noqa: F401
        except Exception:
            return False
        return True

    def _unwrap_envoy(self, module):
        for _ in range(4):
            inner = None
            for attr in ("_module", "_model", "module", "model"):
                if hasattr(module, attr):
                    inner = getattr(module, attr)
                    break
            if inner is None or inner is module:
                break
            module = inner
        return module

    def _find_with_generate(self, module):
        seen = set()
        stack = [module]
        while stack:
            current = stack.pop()
            if current is None:
                continue
            obj_id = id(current)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            if hasattr(current, "generate"):
                return current
            for attr in ("_module", "_model", "model", "module"):
                if hasattr(current, attr):
                    stack.append(getattr(current, attr))
        return None

    def get_hf_causal_lm(self):
        """Return the HF causal LM (with generate)."""
        if self.hf_model is not None and hasattr(self.hf_model, "generate"):
            return self.hf_model
        self.hf_model = None
        candidate = self._find_with_generate(self.model.model)
        if candidate is not None:
            self.hf_model = candidate
            return candidate
        raise AttributeError("Causal LM with generate is not available.")

    def get_hf_base_model(self):
        """Return the HF base model for hidden state access."""
        candidate = self._unwrap_envoy(self.model.model)
        if hasattr(candidate, "layers"):
            return candidate
        if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
            return candidate.model
        return candidate
    
    def apply_chat_template(self, prompt: str) -> str:
        """Apply chat template if available."""
        if self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        return prompt
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text and return tensor."""
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
    
    def decode(self, token_ids) -> str:
        """Decode token ids to string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
    
    def decode_token(self, token_id: int) -> str:
        """Decode a single token id."""
        return self.tokenizer.decode([token_id])
    
    def get_layer_module(self, layer_idx: int):
        """Get the module for a specific layer (architecture-agnostic)."""
        model_envoy = self._unwrap_envoy(self.model.model)
        model_inner = self.get_hf_base_model()

        def _get_layers(root):
            if hasattr(root, "layers"):
                return root.layers
            if hasattr(root, "model") and hasattr(root.model, "layers"):
                return root.model.layers
            if hasattr(root, "transformer") and hasattr(root.transformer, "h"):
                return root.transformer.h
            if hasattr(root, "gpt_neox") and hasattr(root.gpt_neox, "layers"):
                return root.gpt_neox.layers
            return None

        layers = _get_layers(model_envoy)
        if layers is None:
            layers = _get_layers(model_inner)
        if layers is None:
            raise ValueError(f"Unknown model architecture: {type(model_envoy)}")
        return layers[layer_idx]
    
    def get_lm_head(self):
        """Get the language model head (for unembedding)."""
        model_inner = self.get_hf_causal_lm()
        if hasattr(model_inner, "lm_head"):
            return model_inner.lm_head
        if hasattr(model_inner, "embed_out"):
            return model_inner.embed_out
        base = self.get_hf_base_model()
        if hasattr(base, "lm_head"):
            return base.lm_head
        if hasattr(base, "embed_out"):
            return base.embed_out
        raise ValueError("Could not find lm_head")
    
    def get_final_norm(self):
        """Get the final layer norm."""
        model_inner = self.get_hf_base_model()
        if hasattr(model_inner, "model") and hasattr(model_inner.model, "norm"):
            return model_inner.model.norm
        if hasattr(model_inner, "norm"):
            return model_inner.norm
        if hasattr(model_inner, "transformer") and hasattr(model_inner.transformer, "ln_f"):
            return model_inner.transformer.ln_f
        if hasattr(model_inner, "gpt_neox") and hasattr(model_inner.gpt_neox, "final_layer_norm"):
            return model_inner.gpt_neox.final_layer_norm
        raise ValueError("Could not find final norm layer")


def load_model(
    model_name: str,
    quantization: Optional[str] = None,
    device: Optional[str] = None,
    device_map: Optional[str] = None,
) -> LensModel:
    """Convenience function to load a model."""
    return LensModel(
        model_name,
        quantization=quantization,
        device=device,
        device_map=device_map,
    )
