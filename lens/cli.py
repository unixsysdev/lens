"""CLI interface for lens tool."""

import typer
from typing import Optional, List
from rich.console import Console

app = typer.Typer(
    name="lens",
    help="Mechanistic interpretability CLI - logit lens, neuron discovery, steering",
    no_args_is_help=True,
)
console = Console()


def _parse_int_list(value: Optional[str], label: str) -> Optional[List[int]]:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    items = [item.strip() for item in cleaned.split(",") if item.strip()]
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid {label} list: {value}") from exc


def _parse_range(value: Optional[str], label: str) -> Optional[List[int]]:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if ":" not in cleaned:
        raise typer.BadParameter(f"Invalid {label} range: {value} (expected start:end)")
    start_str, end_str = cleaned.split(":", 1)
    try:
        start = int(start_str.strip())
        end = int(end_str.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid {label} range: {value}") from exc
    if end < start:
        raise typer.BadParameter(f"Invalid {label} range: {value} (end < start)")
    return list(range(start, end + 1))


def _parse_prompt_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    cleaned = value.strip()
    if not cleaned:
        return []
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def _load_prompt_file(path: Optional[str], label: str) -> List[str]:
    if path is None:
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = [
                line.strip()
                for line in handle
                if line.strip() and not line.strip().startswith("#")
            ]
    except OSError as exc:
        raise typer.BadParameter(f"Could not read {label} file: {path}") from exc
    return lines


@app.command()
def analyze(
    model: str = typer.Option(..., "-m", "--model", help="Model name or path (HuggingFace)"),
    prompt: str = typer.Option(..., "-p", "--prompt", help="Prompt to analyze"),
    all_layers: bool = typer.Option(False, "--all-layers", help="Show all layers (default: show all)"),
    top_k: int = typer.Option(1, "--top-k", help="Number of top predictions per layer"),
    position: int = typer.Option(-1, "--position", help="Token position to analyze (-1 = last)"),
    positions: Optional[str] = typer.Option(None, "--positions", help="Comma-separated token positions to analyze"),
    position_range: Optional[str] = typer.Option(None, "--range", help="Token position range start:end (inclusive)"),
    layers: Optional[str] = typer.Option(None, "--layers", help="Comma-separated layer indices to analyze"),
    quantization: Optional[str] = typer.Option(None, "-q", "--quantization", help="Quantization: 4bit, 8bit, or none"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="Device map: auto, cpu, cuda, cuda:0, or none"),
    no_chat: bool = typer.Option(False, "--no-chat", help="Don't apply chat template"),
    find_answer: bool = typer.Option(False, "--find-answer", help="Auto-detect answer position"),
    generate_tokens: int = typer.Option(30, "--generate", help="Tokens to generate for answer detection or tracing"),
    trace_generation: bool = typer.Option(False, "--trace-generation", help="Analyze each generated token position"),
    expected: Optional[str] = typer.Option(None, "--expected", help="Expected token string to mark layer it appears"),
    num_samples: int = typer.Option(1, "-n", "--num-samples", help="Number of samples to run"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Sampling temperature for generation"),
    top_p: Optional[float] = typer.Option(None, "--top-p", help="Top-p nucleus sampling"),
    group_by: str = typer.Option("position", "--group-by", help="Group output by position or layer"),
    timeline_probs: bool = typer.Option(False, "--timeline-probs", help="Show probs in layer-grouped timeline"),
    timeline_columns: bool = typer.Option(False, "--timeline-columns", help="Show layer-grouped timeline as columns"),
    steer_layer: Optional[int] = typer.Option(None, "--steer-layer", help="Layer to steer during analyze"),
    steer_neuron: Optional[int] = typer.Option(None, "--steer-neuron", help="Neuron index to steer"),
    steer_strength: float = typer.Option(5.0, "--steer-strength", help="Boost factor for neuron steering"),
    steer_clamp: Optional[float] = typer.Option(None, "--steer-clamp", help="Clamp neuron to this value"),
    steer_vector: Optional[str] = typer.Option(None, "--steer-vector", help="Path to direction vector for steering"),
    steer_coeff: float = typer.Option(1.0, "--steer-coeff", help="Coefficient for steering vector"),
):
    """Run logit lens analysis - see what the model thinks at each layer."""
    from lens.loader import load_model
    from lens.analyzer import (
        Analyzer,
        SteeringConfig,
        print_analysis_result,
        print_sequence_analysis,
    )
    
    # Load model
    lm = load_model(model, quantization=quantization, device_map=device_map)
    analyzer = Analyzer(lm)

    layers_list = _parse_int_list(layers, "layers")
    positions_list = _parse_int_list(positions, "positions")
    range_list = _parse_range(position_range, "positions")
    if positions_list is not None and range_list is not None:
        console.print("[red]Error: Use only one of --positions or --range[/red]")
        raise typer.Exit(1)
    if range_list is not None:
        positions_list = range_list

    if num_samples < 1:
        console.print("[red]Error: --num-samples must be >= 1[/red]")
        raise typer.Exit(1)

    generate_kwargs = {}
    if temperature is not None:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["do_sample"] = True
    if top_p is not None:
        generate_kwargs["top_p"] = top_p
        generate_kwargs["do_sample"] = True
    if num_samples > 1 and "do_sample" not in generate_kwargs:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = 1.0

    expected_token_id = None
    if expected:
        token_ids = lm.tokenizer.encode(expected, add_special_tokens=False)
        if not token_ids:
            console.print("[red]Error: --expected token did not tokenize to any ids[/red]")
            raise typer.Exit(1)
        if len(token_ids) > 1:
            console.print(f"[yellow]Warning: --expected tokenized to multiple ids; using first ({token_ids[0]}).[/yellow]")
        expected_token_id = token_ids[0]

    steering = None
    if steer_layer is not None or steer_neuron is not None or steer_vector is not None:
        if steer_layer is None:
            console.print("[red]Error: --steer-layer is required when using steering options[/red]")
            raise typer.Exit(1)
        if steer_vector and steer_neuron is not None:
            console.print("[red]Error: Use either --steer-vector or --steer-neuron, not both[/red]")
            raise typer.Exit(1)
        if steer_vector:
            import torch
            vector = torch.load(steer_vector)
            steering = SteeringConfig(
                layer_idx=steer_layer,
                mode="vector",
                vector=vector.to(lm.device),
                coefficient=steer_coeff,
            )
        else:
            if steer_neuron is None:
                console.print("[red]Error: --steer-neuron is required when not using --steer-vector[/red]")
                raise typer.Exit(1)
            mode = "neuron_boost"
            value = steer_strength
            if steer_clamp is not None:
                mode = "neuron_clamp"
                value = steer_clamp
            steering = SteeringConfig(
                layer_idx=steer_layer,
                mode=mode,
                neuron_idx=steer_neuron,
                value=value,
            )
        if steering is not None:
            if steering.mode == "vector":
                detail = f"vector coeff={steering.coefficient}"
            elif steering.mode == "neuron_boost":
                detail = f"neuron {steering.neuron_idx} boost={steering.value}"
            elif steering.mode == "neuron_clamp":
                detail = f"neuron {steering.neuron_idx} clamp={steering.value}"
            else:
                detail = steering.mode
            console.print(f"[dim]Steering enabled: layer {steering.layer_idx} ({detail})[/dim]")

    if group_by not in {"position", "layer"}:
        console.print("[red]Error: --group-by must be 'position' or 'layer'[/red]")
        raise typer.Exit(1)

    if trace_generation:
        if find_answer:
            console.print("[red]Error: --trace-generation cannot be combined with --find-answer[/red]")
            raise typer.Exit(1)
        if positions_list is not None and any(pos < 0 for pos in positions_list):
            console.print("[red]Error: --positions/--range cannot include negative values with --trace-generation[/red]")
            raise typer.Exit(1)
        for sample_idx in range(num_samples):
            if num_samples > 1:
                console.print(f"[bold]=== Sample {sample_idx + 1}/{num_samples} ===[/bold]")
            console.print(f"[dim]Generating {generate_tokens} tokens for timeline...[/dim]")
            full_ids, full_text, generated_text, prompt_len = analyzer.generate_sequence(
                prompt=prompt,
                generate_tokens=generate_tokens,
                use_chat_template=not no_chat,
                steering=steering,
                generate_kwargs=generate_kwargs,
            )
            if generated_text:
                console.print(f"[dim]Generated: {generated_text}[/dim]")
            if full_ids.shape[1] <= prompt_len:
                console.print("[red]Error: No generated tokens to analyze.[/red]")
                raise typer.Exit(1)
            if positions_list is None:
                positions_to_use = list(range(prompt_len, full_ids.shape[1]))
            else:
                positions_to_use = positions_list
            result = analyzer.analyze_positions_all_layers(
                prompt=prompt,
                positions=positions_to_use,
                top_k=top_k,
                use_chat_template=not no_chat,
                input_ids=full_ids,
                formatted_prompt=full_text,
                layers=layers_list,
                expected_token_id=expected_token_id,
                steering=steering,
            )
            print_sequence_analysis(
                result,
                show_all=all_layers,
                group_by=group_by,
                timeline_probs=timeline_probs,
                timeline_columns=timeline_columns,
            )
        return
    
    # If find_answer, generate first to find the answer position
    analysis_position = position
    analysis_input_ids = None
    analysis_formatted = None
    if find_answer:
        if positions_list is not None:
            console.print("[red]Error: --find-answer cannot be combined with --positions/--range[/red]")
            raise typer.Exit(1)
        console.print(f"[dim]Generating {generate_tokens} tokens to find answer position...[/dim]")
        detection = analyzer.detect_answer_position(
            prompt=prompt,
            generate_tokens=generate_tokens,
            use_chat_template=not no_chat,
            steering=steering,
            generate_kwargs=generate_kwargs,
        )
        analysis_position = detection.analysis_position
        analysis_input_ids = detection.input_ids
        analysis_formatted = detection.analysis_text
        if detection.generated_text:
            console.print(f"[dim]Generated: {detection.generated_text}[/dim]")
        if detection.answer_token:
            console.print(f"[dim]Answer token: '{detection.answer_token}' at position {analysis_position}[/dim]")
    
    for sample_idx in range(num_samples):
        if num_samples > 1:
            console.print(f"[bold]=== Sample {sample_idx + 1}/{num_samples} ===[/bold]")
        if positions_list is not None:
            result = analyzer.analyze_positions_all_layers(
                prompt=prompt,
                positions=positions_list,
                top_k=top_k,
                use_chat_template=not no_chat,
                input_ids=analysis_input_ids,
                formatted_prompt=analysis_formatted,
                layers=layers_list,
                expected_token_id=expected_token_id,
                steering=steering,
            )
            print_sequence_analysis(
                result,
                show_all=all_layers,
                group_by=group_by,
                timeline_probs=timeline_probs,
                timeline_columns=timeline_columns,
            )
        else:
            result = analyzer.analyze_all_layers(
                prompt=prompt,
                position=analysis_position,
                top_k=top_k,
                use_chat_template=not no_chat,
                input_ids=analysis_input_ids,
                formatted_prompt=analysis_formatted,
                layers=layers_list,
                expected_token_id=expected_token_id,
                steering=steering,
            )
            print_analysis_result(result, show_all=all_layers)


@app.command()
def diff(
    model: str = typer.Option(..., "-m", "--model", help="Model name or path"),
    pos: Optional[str] = typer.Option(None, "--pos", help="Positive concept prompt"),
    neg: Optional[str] = typer.Option(None, "--neg", help="Negative concept prompt"),
    pos_list: Optional[str] = typer.Option(None, "--pos-list", help="Comma-separated positive prompts"),
    neg_list: Optional[str] = typer.Option(None, "--neg-list", help="Comma-separated negative prompts"),
    pos_file: Optional[str] = typer.Option(None, "--pos-file", help="Path to file with positive prompts"),
    neg_file: Optional[str] = typer.Option(None, "--neg-file", help="Path to file with negative prompts"),
    layer: Optional[int] = typer.Option(None, "-l", "--layer", help="Layer to analyze"),
    layers: Optional[str] = typer.Option(None, "--layers", help="Comma-separated layer indices to analyze"),
    layer_range: Optional[str] = typer.Option(None, "--layer-range", help="Layer range start:end (inclusive)"),
    top_k: int = typer.Option(10, "--top-k", help="Number of top neurons to show"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="Device map: auto, cpu, cuda, cuda:0, or none"),
    no_chat: bool = typer.Option(False, "--no-chat", help="Don't apply chat template"),
    save_vector: Optional[str] = typer.Option(None, "--save", help="Save direction vector to file"),
):
    """Find neurons that distinguish between concepts (pos vs neg prompts)."""
    import torch
    from lens.loader import load_model
    from lens.discovery import NeuronDiscovery, print_discovery_result
    
    lm = load_model(model, device_map=device_map)
    discovery = NeuronDiscovery(lm)

    pos_prompts = []
    neg_prompts = []
    if pos:
        pos_prompts.append(pos)
    if neg:
        neg_prompts.append(neg)
    pos_prompts.extend(_parse_prompt_list(pos_list))
    neg_prompts.extend(_parse_prompt_list(neg_list))
    pos_prompts.extend(_load_prompt_file(pos_file, "positive"))
    neg_prompts.extend(_load_prompt_file(neg_file, "negative"))
    if not pos_prompts:
        console.print("[red]Error: At least one positive prompt is required.[/red]")
        raise typer.Exit(1)
    if not neg_prompts:
        console.print("[red]Error: At least one negative prompt is required.[/red]")
        raise typer.Exit(1)

    layers_list = _parse_int_list(layers, "layers")
    range_list = _parse_range(layer_range, "layers")
    if layers_list is not None and range_list is not None:
        console.print("[red]Error: Use only one of --layers or --layer-range[/red]")
        raise typer.Exit(1)
    if range_list is not None:
        layers_list = range_list
    if layer is not None and layers_list is not None:
        console.print("[red]Error: Use either --layer or --layers/--layer-range, not both[/red]")
        raise typer.Exit(1)
    if layer is None and layers_list is None:
        console.print("[red]Error: --layer is required unless --layers/--layer-range is provided[/red]")
        raise typer.Exit(1)
    if layer is not None:
        layers_list = [layer]

    def _layered_save_path(path: str, layer_idx: int) -> str:
        if path.endswith(".pt"):
            base = path[:-3]
            return f"{base}.layer{layer_idx}.pt"
        return f"{path}.layer{layer_idx}.pt"

    for idx, layer_idx in enumerate(layers_list or []):
        result = discovery.find_concept_neurons(
            pos_prompt=pos_prompts,
            neg_prompt=neg_prompts,
            layer_idx=layer_idx,
            top_k=top_k,
            use_chat_template=not no_chat,
        )

        if idx > 0:
            console.print()
        print_discovery_result(result)

        if save_vector and result.direction_vector is not None:
            save_path = save_vector
            if len(layers_list) > 1:
                save_path = _layered_save_path(save_vector, layer_idx)
            torch.save(result.direction_vector, save_path)
            console.print(f"[green]Direction vector saved to {save_path}[/green]")


@app.command()
def steer(
    model: str = typer.Option(..., "-m", "--model", help="Model name or path"),
    prompt: str = typer.Option(..., "-p", "--prompt", help="Prompt to generate from"),
    layer: int = typer.Option(..., "-l", "--layer", help="Layer to intervene on"),
    neuron: Optional[int] = typer.Option(None, "-n", "--neuron", help="Neuron index to clamp/boost"),
    boost: float = typer.Option(5.0, "-b", "--boost", help="Boost factor (multiplier) for neuron"),
    clamp: Optional[float] = typer.Option(None, "-c", "--clamp", help="Clamp neuron to this value"),
    vector_file: Optional[str] = typer.Option(None, "-v", "--vector", help="Load direction vector from file"),
    coeff: float = typer.Option(1.0, "--coeff", help="Coefficient for vector steering"),
    max_tokens: int = typer.Option(20, "--max-tokens", help="Max tokens to generate"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="Device map: auto, cpu, cuda, cuda:0, or none"),
    no_chat: bool = typer.Option(False, "--no-chat", help="Don't apply chat template"),
):
    """Steer model by clamping neurons or adding direction vectors."""
    import torch
    from lens.loader import load_model
    from lens.steering import Steering, print_steering_result
    
    lm = load_model(model, device_map=device_map)
    steering = Steering(lm)
    
    if vector_file:
        # Vector steering
        direction = torch.load(vector_file)
        result = steering.generate_with_vector(
            prompt=prompt,
            layer_idx=layer,
            direction_vector=direction,
            coefficient=coeff,
            max_new_tokens=max_tokens,
            use_chat_template=not no_chat,
        )
    elif neuron is not None:
        if clamp is not None:
            # Clamp neuron to specific value
            result = steering.generate_with_neuron_clamp(
                prompt=prompt,
                layer_idx=layer,
                neuron_idx=neuron,
                value=clamp,
                max_new_tokens=max_tokens,
                use_chat_template=not no_chat,
            )
        else:
            # Boost neuron
            result = steering.generate_with_neuron_boost(
                prompt=prompt,
                layer_idx=layer,
                neuron_idx=neuron,
                boost=boost,
                max_new_tokens=max_tokens,
                use_chat_template=not no_chat,
            )
    else:
        console.print("[red]Error: Must specify either --neuron or --vector[/red]")
        raise typer.Exit(1)
    
    print_steering_result(result)


@app.command()
def info(
    model: str = typer.Option(..., "-m", "--model", help="Model name or path"),
    device_map: Optional[str] = typer.Option(None, "--device-map", help="Device map: auto, cpu, cuda, cuda:0, or none"),
):
    """Show model info and memory requirements."""
    from lens.loader import load_model
    from lens.utils import get_memory_info
    
    mem = get_memory_info()
    console.print(f"[bold]Device:[/bold] {mem['device']}")
    if 'name' in mem:
        console.print(f"[bold]GPU:[/bold] {mem['name']}")
    if 'total_gb' in mem:
        console.print(f"[bold]VRAM:[/bold] {mem['total_gb']:.1f} GB")
    console.print()
    
    lm = load_model(model, device_map=device_map)
    # Model info is printed by load_model


if __name__ == "__main__":
    app()
