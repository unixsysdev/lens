"""Neuron discovery: find neurons that distinguish between concepts."""

import torch
from typing import List, Tuple, Optional, Union, Sequence
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

from .loader import LensModel
from .utils import clear_memory

console = Console()


@dataclass
class NeuronDiff:
    """A neuron that shows significant difference between pos/neg prompts."""
    neuron_idx: int
    layer_idx: int
    pos_activation: float
    neg_activation: float
    delta: float


@dataclass
class DiscoveryResult:
    """Result of neuron discovery."""
    pos_prompt: str
    neg_prompt: str
    layer_idx: int
    top_neurons: List[NeuronDiff]
    # The direction vector (can be used for steering)
    direction_vector: Optional[torch.Tensor] = None


class NeuronDiscovery:
    """Find neurons that distinguish between concepts."""
    
    def __init__(self, model: LensModel):
        self.model = model
        self._nnsight_warned = False

    @staticmethod
    def _normalize_prompts(prompts: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(prompts, str):
            cleaned = prompts.strip()
            return [cleaned] if cleaned else []
        return [prompt.strip() for prompt in prompts if prompt and prompt.strip()]

    @staticmethod
    def _summarize_prompts(prompts: List[str], max_items: int = 3) -> str:
        if len(prompts) <= max_items:
            return " | ".join(f"'{prompt}'" for prompt in prompts)
        preview = " | ".join(f"'{prompt}'" for prompt in prompts[:max_items])
        return f"{preview} | +{len(prompts) - max_items} more"

    def _get_all_hidden_states(self, input_ids: torch.Tensor):
        """Fetch all hidden states via HF forward as a fallback."""
        hf_model = self.model.get_hf_base_model()
        attention_mask = torch.ones_like(input_ids)
        outputs = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("HF forward did not return hidden_states.")
        return hidden_states

    def _select_hidden_layer(self, hidden_states, layer_idx: int) -> torch.Tensor:
        if len(hidden_states) == self.model.num_layers + 1:
            return hidden_states[layer_idx + 1]
        if len(hidden_states) == self.model.num_layers:
            return hidden_states[layer_idx]
        raise ValueError("Unexpected hidden_states length for layer selection.")
    
    def _get_mlp_activations(self, layer_idx: int, input_ids: torch.Tensor) -> torch.Tensor:
        """Get MLP intermediate activations (post-nonlinearity) at a layer."""
        model_inner = self.model.model
        
        with model_inner.trace(input_ids):
            layer = self.model.get_layer_module(layer_idx)
            
            # Try to get MLP activations - architecture dependent
            # Most models have mlp.act or mlp.activation_fn output
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                # Try common patterns
                if hasattr(mlp, "act_fn"):
                    # Llama style - capture after activation
                    act = mlp.act_fn.output.save()
                elif hasattr(mlp, "activation_fn"):
                    act = mlp.activation_fn.output.save()
                else:
                    # Fallback: capture full MLP output
                    act = mlp.output.save()
            else:
                # GPT-2 style
                act = layer.mlp.output.save()
        
        return act.value
    
    def _get_residual_stream(self, layer_idx: int, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the residual stream (hidden state) at a layer."""
        model_inner = self.model.model
        layer = self.model.get_layer_module(layer_idx)
        use_nnsight = hasattr(layer, "output")
        if use_nnsight:
            try:
                with model_inner.trace(input_ids):
                    layer = self.model.get_layer_module(layer_idx)
                    hidden = layer.output[0].save()
                return hidden.value
            except Exception:
                use_nnsight = False

        if not use_nnsight:
            if not self._nnsight_warned:
                console.print("[yellow]nnsight tracing unavailable; using HF hidden_states.[/yellow]")
                self._nnsight_warned = True
            hidden_states = self._get_all_hidden_states(input_ids)
            return self._select_hidden_layer(hidden_states, layer_idx)
    
    def find_concept_neurons(
        self,
        pos_prompt: Union[str, Sequence[str]],
        neg_prompt: Union[str, Sequence[str]],
        layer_idx: int,
        top_k: int = 10,
        use_chat_template: bool = True,
        position: int = -1,  # Which position to analyze
    ) -> DiscoveryResult:
        """Find neurons that fire differently for pos vs neg prompts.
        
        This is how you find "the math neuron" or "the difficulty neuron".
        """
        pos_prompts = self._normalize_prompts(pos_prompt)
        neg_prompts = self._normalize_prompts(neg_prompt)
        if not pos_prompts or not neg_prompts:
            raise ValueError("Both positive and negative prompt lists must be non-empty.")

        console.print(f"[bold]Finding concept neurons at layer {layer_idx}[/bold]")
        if len(pos_prompts) == 1:
            console.print(f"[green]Positive:[/green] '{pos_prompts[0]}'")
        else:
            console.print(f"[green]Positive ({len(pos_prompts)} prompts):[/green] {self._summarize_prompts(pos_prompts)}")
        if len(neg_prompts) == 1:
            console.print(f"[red]Negative:[/red] '{neg_prompts[0]}'")
        else:
            console.print(f"[red]Negative ({len(neg_prompts)} prompts):[/red] {self._summarize_prompts(neg_prompts)}")
        console.print()
        
        def _mean_activation(prompts: List[str]) -> torch.Tensor:
            total = None
            with torch.no_grad():
                for prompt in prompts:
                    if use_chat_template:
                        formatted = self.model.apply_chat_template(prompt)
                    else:
                        formatted = prompt
                    input_ids = self.model.tokenize(formatted)
                    hidden = self._get_residual_stream(layer_idx, input_ids)
                    act = hidden[0, position, :]
                    total = act.clone() if total is None else total + act
            return total / len(prompts)

        pos_act = _mean_activation(pos_prompts)
        neg_act = _mean_activation(neg_prompts)
        
        # Compute delta
        delta = pos_act - neg_act
        
        # Find top neurons by absolute delta
        abs_delta = delta.abs()
        top_values, top_indices = torch.topk(abs_delta, top_k)
        
        # Build results
        top_neurons = []
        for val, idx in zip(top_values, top_indices):
            idx = idx.item()
            top_neurons.append(NeuronDiff(
                neuron_idx=idx,
                layer_idx=layer_idx,
                pos_activation=pos_act[idx].item(),
                neg_activation=neg_act[idx].item(),
                delta=delta[idx].item(),
            ))
        
        clear_memory()
        
        return DiscoveryResult(
            pos_prompt=self._summarize_prompts(pos_prompts, max_items=5),
            neg_prompt=self._summarize_prompts(neg_prompts, max_items=5),
            layer_idx=layer_idx,
            top_neurons=top_neurons,
            direction_vector=delta.detach().cpu(),
        )
    
    def find_across_layers(
        self,
        pos_prompt: Union[str, Sequence[str]],
        neg_prompt: Union[str, Sequence[str]],
        top_k: int = 5,
        use_chat_template: bool = True,
    ) -> List[DiscoveryResult]:
        """Find concept neurons across all layers."""
        results = []
        for layer_idx in range(self.model.num_layers):
            result = self.find_concept_neurons(
                pos_prompt, neg_prompt, layer_idx, 
                top_k=top_k, use_chat_template=use_chat_template
            )
            results.append(result)
        return results


def print_discovery_result(result: DiscoveryResult):
    """Pretty print discovery results."""
    console.print()
    console.print(f"[bold]=== Top Concept Neurons at Layer {result.layer_idx} ===[/bold]")
    
    table = Table(show_header=True)
    table.add_column("Neuron", style="cyan")
    table.add_column("Pos Act", justify="right")
    table.add_column("Neg Act", justify="right")
    table.add_column("Delta", justify="right", style="bold")
    
    for n in result.top_neurons:
        delta_style = "green" if n.delta > 0 else "red"
        table.add_row(
            str(n.neuron_idx),
            f"{n.pos_activation:.4f}",
            f"{n.neg_activation:.4f}",
            f"[{delta_style}]{n.delta:+.4f}[/{delta_style}]",
        )
    
    console.print(table)
