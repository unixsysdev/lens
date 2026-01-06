"""Steering: inject vectors or clamp neurons to change model behavior."""

import torch
from typing import Optional, Callable
from dataclasses import dataclass
from rich.console import Console

from .loader import LensModel
from .utils import clear_memory

console = Console()


@dataclass
class SteeringResult:
    """Result of a steered generation."""
    prompt: str
    original_output: str
    steered_output: str
    layer_idx: int
    intervention_type: str  # "neuron" or "vector"
    intervention_details: str


class Steering:
    """Steer model behavior by intervening on activations."""
    
    def __init__(self, model: LensModel):
        self.model = model
        self._nnsight_warned = False

    def _supports_nnsight(self, layer_idx: int) -> bool:
        try:
            layer = self.model.get_layer_module(layer_idx)
        except Exception:
            return False
        return hasattr(layer, "output")

    @staticmethod
    def _apply_to_output(output, fn):
        if torch.is_tensor(output):
            return fn(output)
        if isinstance(output, tuple):
            if not output:
                return output
            hidden = output[0]
            if torch.is_tensor(hidden):
                return (fn(hidden),) + output[1:]
            return output
        if isinstance(output, list):
            if not output:
                return output
            hidden = output[0]
            if torch.is_tensor(hidden):
                new_output = list(output)
                new_output[0] = fn(hidden)
                return new_output
            return output
        return output

    def _apply_vector_to_hidden(
        self,
        hidden: torch.Tensor,
        direction: torch.Tensor,
        coefficient: float,
    ) -> torch.Tensor:
        if hidden.dim() < 3:
            return hidden
        if direction.dim() > 1:
            direction = direction[-1]
        direction = direction.to(device=hidden.device, dtype=hidden.dtype)
        output = hidden.clone()
        output[:, -1, :] = output[:, -1, :] + (coefficient * direction)
        return output

    def _apply_neuron_to_hidden(
        self,
        hidden: torch.Tensor,
        neuron_idx: int,
        value: float,
        mode: str,
    ) -> torch.Tensor:
        if hidden.dim() < 3:
            return hidden
        output = hidden.clone()
        if mode == "clamp":
            output[:, -1, neuron_idx] = value
        elif mode == "boost":
            output[:, -1, neuron_idx] = output[:, -1, neuron_idx] * value
        return output

    def _run_hf_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        layer: Optional[torch.nn.Module] = None,
        hook: Optional[Callable] = None,
    ) -> torch.Tensor:
        hf_model = self.model.get_hf_causal_lm()
        attention_mask = torch.ones_like(input_ids)
        handle = None
        if layer is not None and hook is not None:
            handle = layer.register_forward_hook(hook)
        try:
            output_ids = hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
        finally:
            if handle is not None:
                handle.remove()
        if output_ids.dim() == 1:
            output_ids = output_ids.unsqueeze(0)
        return output_ids
    
    def generate_with_neuron_clamp(
        self,
        prompt: str,
        layer_idx: int,
        neuron_idx: int,
        value: float = 10.0,
        max_new_tokens: int = 20,
        use_chat_template: bool = True,
    ) -> SteeringResult:
        """Generate with a specific neuron clamped to a value.
        
        This is how you "flip the switch" on a concept neuron.
        """
        console.print(f"[bold]Steering: Clamping neuron {neuron_idx} at layer {layer_idx} to {value}[/bold]")
        
        # Format prompt
        if use_chat_template:
            formatted = self.model.apply_chat_template(prompt)
        else:
            formatted = prompt
        
        input_ids = self.model.tokenize(formatted)
        use_nnsight = self._supports_nnsight(layer_idx)

        if use_nnsight:
            # First, generate without intervention (baseline)
            console.print("[dim]Generating baseline...[/dim]")
            with self.model.model.generate(input_ids, max_new_tokens=max_new_tokens) as gen:
                baseline_output = gen.output.save()
            original_text = self.model.decode(baseline_output.value[0])

            # Now generate with neuron clamped
            console.print("[dim]Generating with steering...[/dim]")
            with self.model.model.generate(input_ids, max_new_tokens=max_new_tokens) as gen:
                layer = self.model.get_layer_module(layer_idx)
                # Clamp the neuron at the current (last) position
                layer.output[0][:, -1, neuron_idx] = value
                steered_output = gen.output.save()
            steered_text = self.model.decode(steered_output.value[0])
        else:
            if not self._nnsight_warned:
                console.print("[yellow]nnsight tracing unavailable; using HF hooks for steering.[/yellow]")
                self._nnsight_warned = True
            console.print("[dim]Generating baseline...[/dim]")
            baseline_ids = self._run_hf_generate(input_ids, max_new_tokens)
            original_text = self.model.decode(baseline_ids[0])

            console.print("[dim]Generating with steering...[/dim]")
            layer = self.model.get_layer_module(layer_idx)

            def hook_fn(module, inputs, output):
                return self._apply_to_output(
                    output,
                    lambda hidden: self._apply_neuron_to_hidden(hidden, neuron_idx, value, "clamp"),
                )

            steered_ids = self._run_hf_generate(input_ids, max_new_tokens, layer=layer, hook=hook_fn)
            steered_text = self.model.decode(steered_ids[0])
        
        clear_memory()
        
        return SteeringResult(
            prompt=prompt,
            original_output=original_text,
            steered_output=steered_text,
            layer_idx=layer_idx,
            intervention_type="neuron",
            intervention_details=f"neuron {neuron_idx} = {value}",
        )
    
    def generate_with_vector(
        self,
        prompt: str,
        layer_idx: int,
        direction_vector: torch.Tensor,
        coefficient: float = 1.0,
        max_new_tokens: int = 20,
        use_chat_template: bool = True,
    ) -> SteeringResult:
        """Generate with a direction vector added to the residual stream.
        
        This is activation addition / representation engineering.
        The direction_vector typically comes from NeuronDiscovery.find_concept_neurons().
        """
        console.print(f"[bold]Steering: Adding direction vector at layer {layer_idx} (coeff={coefficient})[/bold]")
        
        # Format prompt
        if use_chat_template:
            formatted = self.model.apply_chat_template(prompt)
        else:
            formatted = prompt
        
        input_ids = self.model.tokenize(formatted)
        use_nnsight = self._supports_nnsight(layer_idx)

        # Move vector to device
        direction = direction_vector.to(self.model.device)

        if use_nnsight:
            # Generate baseline
            console.print("[dim]Generating baseline...[/dim]")
            with self.model.model.generate(input_ids, max_new_tokens=max_new_tokens) as gen:
                baseline_output = gen.output.save()
            original_text = self.model.decode(baseline_output.value[0])

            # Generate with vector steering
            console.print("[dim]Generating with steering...[/dim]")
            with self.model.model.generate(input_ids, max_new_tokens=max_new_tokens) as gen:
                layer = self.model.get_layer_module(layer_idx)
                # Add the direction vector to the residual stream at the last position
                if direction.dim() > 1:
                    direction = direction[-1]
                direction = direction.view(1, -1)
                layer.output[0][:, -1, :] = layer.output[0][:, -1, :] + (coefficient * direction)
                steered_output = gen.output.save()
            steered_text = self.model.decode(steered_output.value[0])
        else:
            if not self._nnsight_warned:
                console.print("[yellow]nnsight tracing unavailable; using HF hooks for steering.[/yellow]")
                self._nnsight_warned = True
            console.print("[dim]Generating baseline...[/dim]")
            baseline_ids = self._run_hf_generate(input_ids, max_new_tokens)
            original_text = self.model.decode(baseline_ids[0])

            console.print("[dim]Generating with steering...[/dim]")
            layer = self.model.get_layer_module(layer_idx)

            def hook_fn(module, inputs, output):
                return self._apply_to_output(
                    output,
                    lambda hidden: self._apply_vector_to_hidden(hidden, direction, coefficient),
                )

            steered_ids = self._run_hf_generate(input_ids, max_new_tokens, layer=layer, hook=hook_fn)
            steered_text = self.model.decode(steered_ids[0])
        
        clear_memory()
        
        return SteeringResult(
            prompt=prompt,
            original_output=original_text,
            steered_output=steered_text,
            layer_idx=layer_idx,
            intervention_type="vector",
            intervention_details=f"direction vector * {coefficient}",
        )
    
    def generate_with_neuron_boost(
        self,
        prompt: str,
        layer_idx: int,
        neuron_idx: int,
        boost: float = 5.0,
        max_new_tokens: int = 20,
        use_chat_template: bool = True,
    ) -> SteeringResult:
        """Generate with a neuron boosted (multiplied) rather than clamped.
        
        More subtle than clamping - amplifies existing signal.
        """
        console.print(f"[bold]Steering: Boosting neuron {neuron_idx} at layer {layer_idx} by {boost}x[/bold]")
        
        # Format prompt
        if use_chat_template:
            formatted = self.model.apply_chat_template(prompt)
        else:
            formatted = prompt
        
        input_ids = self.model.tokenize(formatted)
        use_nnsight = self._supports_nnsight(layer_idx)

        if use_nnsight:
            # Generate baseline
            with self.model.model.generate(input_ids, max_new_tokens=max_new_tokens) as gen:
                baseline_output = gen.output.save()
            original_text = self.model.decode(baseline_output.value[0])

            # Generate with boost
            with self.model.model.generate(input_ids, max_new_tokens=max_new_tokens) as gen:
                layer = self.model.get_layer_module(layer_idx)
                # Multiply (boost) the specific neuron at the last position
                layer.output[0][:, -1, neuron_idx] = layer.output[0][:, -1, neuron_idx] * boost
                steered_output = gen.output.save()
            steered_text = self.model.decode(steered_output.value[0])
        else:
            if not self._nnsight_warned:
                console.print("[yellow]nnsight tracing unavailable; using HF hooks for steering.[/yellow]")
                self._nnsight_warned = True
            baseline_ids = self._run_hf_generate(input_ids, max_new_tokens)
            original_text = self.model.decode(baseline_ids[0])

            layer = self.model.get_layer_module(layer_idx)

            def hook_fn(module, inputs, output):
                return self._apply_to_output(
                    output,
                    lambda hidden: self._apply_neuron_to_hidden(hidden, neuron_idx, boost, "boost"),
                )

            steered_ids = self._run_hf_generate(input_ids, max_new_tokens, layer=layer, hook=hook_fn)
            steered_text = self.model.decode(steered_ids[0])
        
        clear_memory()
        
        return SteeringResult(
            prompt=prompt,
            original_output=original_text,
            steered_output=steered_text,
            layer_idx=layer_idx,
            intervention_type="neuron_boost",
            intervention_details=f"neuron {neuron_idx} * {boost}",
        )


def print_steering_result(result: SteeringResult):
    """Pretty print steering results."""
    console.print()
    console.print(f"[bold]=== Steering Result ===[/bold]")
    console.print(f"[bold]Intervention:[/bold] {result.intervention_type} at layer {result.layer_idx}")
    console.print(f"[bold]Details:[/bold] {result.intervention_details}")
    console.print()
    console.print(f"[bold]Original:[/bold]")
    console.print(f"  {result.original_output}")
    console.print()
    console.print(f"[bold green]Steered:[/bold green]")
    console.print(f"  {result.steered_output}")
