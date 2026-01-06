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
        
        # Move vector to device
        direction = direction_vector.to(self.model.device)
        
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
