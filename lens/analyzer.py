"""Core analysis: logit lens, answer detection, layer-by-layer predictions."""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .loader import LensModel
from .utils import clear_memory

console = Console()


@dataclass
class LayerPrediction:
    """Prediction info for a single layer."""
    layer_idx: int
    top_tokens: List[int]
    top_probs: List[float]
    top_strings: List[str]


@dataclass 
class AnalysisResult:
    """Full analysis result."""
    prompt: str
    formatted_prompt: str
    tokens: List[str]
    token_ids: List[int]
    analysis_position: int  # Which token position we analyzed
    layer_predictions: List[LayerPrediction]
    final_prediction: str
    final_prob: float
    peak_layer: Optional[int] = None  # Layer where correct answer first appears with high confidence
    expected_token: Optional[str] = None
    expected_peak_layer: Optional[int] = None


@dataclass
class PositionAnalysis:
    """Analysis result for a single token position."""
    position: int
    token_id: int
    token: str
    layer_predictions: List[LayerPrediction]
    final_prediction: str
    final_prob: float
    peak_layer: Optional[int] = None
    expected_peak_layer: Optional[int] = None


@dataclass
class SequenceAnalysis:
    """Analysis result for multiple token positions."""
    prompt: str
    formatted_prompt: str
    tokens: List[str]
    token_ids: List[int]
    positions: List[int]
    position_results: List[PositionAnalysis]
    expected_token: Optional[str] = None
    expected_token_id: Optional[int] = None


@dataclass
class SteeringConfig:
    """Steering configuration for analysis/generation."""
    layer_idx: int
    mode: str  # "neuron_boost", "neuron_clamp", or "vector"
    neuron_idx: Optional[int] = None
    value: Optional[float] = None
    vector: Optional[torch.Tensor] = None
    coefficient: float = 1.0


@dataclass
class AnswerDetection:
    """Answer detection result for analysis."""
    input_ids: torch.Tensor
    analysis_position: int
    analysis_text: str
    generated_text: str
    answer_token: str


class Analyzer:
    """Performs logit lens analysis on a model."""
    
    def __init__(self, model: LensModel):
        self.model = model
    
    def _get_hidden_states_at_layer(self, layer_idx: int, input_ids: torch.Tensor) -> torch.Tensor:
        """Get hidden states at a specific layer using nnsight tracing."""
        with self.model.model.trace(input_ids) as tracer:
            # Access the layer output - this captures the residual stream after the layer
            layer = self.model.get_layer_module(layer_idx)
            # Most architectures output (hidden_states, ...) tuple
            hidden = layer.output[0].save()
        
        return hidden.value
    
    def _apply_logit_lens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply final norm and lm_head to get logits from hidden states."""
        # Get the final norm and lm_head
        with self.model.model.trace(torch.zeros(1, 1, dtype=torch.long, device=self.model.device)) as tracer:
            # We need to manually apply norm and lm_head
            # This is a bit hacky but works across architectures
            norm = self.model.get_final_norm()
            lm_head = self.model.get_lm_head()
        
        # Apply outside of trace
        normed = norm(hidden_states)
        logits = lm_head(normed)
        return logits

    def _apply_steering(self, hidden: torch.Tensor, config: SteeringConfig) -> torch.Tensor:
        if hidden.dim() < 3:
            return hidden
        output = hidden
        if config.mode == "vector" and config.vector is not None:
            direction = config.vector
            if direction.dim() > 1:
                direction = direction[-1]
            direction = direction.view(1, 1, -1)
            output = output.clone()
            output[:, -1:, :] = output[:, -1:, :] + (config.coefficient * direction)
            return output
        if config.neuron_idx is None:
            return output
        output = output.clone()
        if config.mode == "neuron_clamp" and config.value is not None:
            output[:, -1, config.neuron_idx] = config.value
        elif config.mode == "neuron_boost" and config.value is not None:
            output[:, -1, config.neuron_idx] = output[:, -1, config.neuron_idx] * config.value
        return output

    @contextmanager
    def _steering_context(self, config: Optional[SteeringConfig]):
        if config is None:
            yield
            return

        layer = self.model.get_layer_module(config.layer_idx)

        def hook_fn(module, inputs, output):
            if isinstance(output, (tuple, list)):
                hidden = output[0]
                steered = self._apply_steering(hidden, config)
                return (steered,) + tuple(output[1:])
            return self._apply_steering(output, config)

        handle = layer.register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()
    
    def analyze_all_layers(
        self,
        prompt: str,
        position: int = -1,  # Which token position to analyze (-1 = last)
        top_k: int = 1,
        use_chat_template: bool = True,
        input_ids: Optional[torch.Tensor] = None,
        formatted_prompt: Optional[str] = None,
        layers: Optional[List[int]] = None,
        expected_token_id: Optional[int] = None,
        steering: Optional[SteeringConfig] = None,
    ) -> AnalysisResult:
        """Run logit lens across all layers.
        
        This captures what the model "thinks" at each layer before reaching output.
        """
        # Format prompt and tokenize unless input_ids are provided (e.g., from generation)
        if input_ids is None:
            if use_chat_template:
                formatted = self.model.apply_chat_template(prompt)
            else:
                formatted = prompt
            input_ids = self.model.tokenize(formatted)
        else:
            formatted = formatted_prompt or self.model.decode(input_ids[0])
        
        console.print(f"[bold]Analyzing{'(with chat template)' if use_chat_template else ''}:[/bold] '{prompt}'")
        seq_len = input_ids.shape[1]
        
        # Get token strings for display
        token_ids_list = input_ids[0].tolist()
        token_strings = [self.model.decode_token(t) for t in token_ids_list]
        
        # Resolve position
        if position < 0:
            position = seq_len + position
        if position < 0 or position >= seq_len:
            raise ValueError(f"Analysis position {position} is out of bounds for seq_len={seq_len}")
        
        console.print(f"[dim]Tokens ({seq_len}): {token_strings[:5]}...{token_strings[-3:]}[/dim]")
        layer_indices = self._resolve_layers(layers)
        if len(layer_indices) == self.model.num_layers:
            layer_display = f"[0, 1, 2, ..., {self.model.num_layers - 1}]"
        else:
            layer_display = f"{layer_indices}"
        console.print(f"[dim]Captured layers: {layer_display}[/dim]")
        console.print(f"[dim]Analyzing prediction at position {position}[/dim]")
        console.print()
        
        layer_predictions = []
        hidden_states = None
        use_nnsight = True
        try:
            probe = self.model.get_layer_module(0)
            use_nnsight = hasattr(probe, "output")
        except Exception:
            use_nnsight = False
        if steering is not None:
            use_nnsight = False
        if not use_nnsight:
            console.print("[yellow]nnsight tracing unavailable; using HF hidden_states.[/yellow]")
            hidden_states = self._get_all_hidden_states(input_ids, steering=steering)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing layers...", total=len(layer_indices))
            
            for layer_idx in layer_indices:
                progress.update(task, description=f"Layer {layer_idx}/{self.model.num_layers-1}")
                
                # Get hidden states at this layer
                if hidden_states is None:
                    hidden = self._get_hidden_states_at_layer_efficient(layer_idx, input_ids)
                else:
                    hidden = self._select_hidden_layer(hidden_states, layer_idx)
                
                # Get the hidden state at our position of interest
                hidden_at_pos = hidden[0, position, :].unsqueeze(0).unsqueeze(0)
                
                # Apply logit lens (norm + unembed)
                logits = self._apply_logit_lens_efficient(hidden_at_pos)
                
                # Get probabilities
                probs = F.softmax(logits[0, 0], dim=-1)
                top_probs, top_indices = torch.topk(probs, top_k)
                
                top_tokens = top_indices.tolist()
                top_probs_list = top_probs.tolist()
                top_strings = [self.model.decode_token(t) for t in top_tokens]
                
                layer_predictions.append(LayerPrediction(
                    layer_idx=layer_idx,
                    top_tokens=top_tokens,
                    top_probs=top_probs_list,
                    top_strings=top_strings,
                ))
                
                progress.advance(task)
        
        # Get final prediction (last layer)
        final = layer_predictions[-1]
        
        # Find peak layer (where the final answer first appears with high confidence)
        peak_layer = self._find_peak_layer(layer_predictions, final.top_tokens[0])
        expected_peak_layer = None
        expected_token = None
        if expected_token_id is not None:
            expected_peak_layer = self._find_peak_layer(layer_predictions, expected_token_id)
            expected_token = self.model.decode_token(expected_token_id)
        
        clear_memory()
        
        return AnalysisResult(
            prompt=prompt,
            formatted_prompt=formatted,
            tokens=token_strings,
            token_ids=token_ids_list,
            analysis_position=position,
            layer_predictions=layer_predictions,
            final_prediction=final.top_strings[0],
            final_prob=final.top_probs[0],
            peak_layer=peak_layer,
            expected_token=expected_token,
            expected_peak_layer=expected_peak_layer,
        )

    def analyze_positions_all_layers(
        self,
        prompt: str,
        positions: List[int],
        top_k: int = 1,
        use_chat_template: bool = True,
        input_ids: Optional[torch.Tensor] = None,
        formatted_prompt: Optional[str] = None,
        layers: Optional[List[int]] = None,
        expected_token_id: Optional[int] = None,
        steering: Optional[SteeringConfig] = None,
    ) -> SequenceAnalysis:
        """Run logit lens across all layers for multiple positions."""
        if not positions:
            raise ValueError("No positions provided for analysis.")

        if input_ids is None:
            if use_chat_template:
                formatted = self.model.apply_chat_template(prompt)
            else:
                formatted = prompt
            input_ids = self.model.tokenize(formatted)
        else:
            formatted = formatted_prompt or self.model.decode(input_ids[0])

        seq_len = input_ids.shape[1]
        token_ids_list = input_ids[0].tolist()
        token_strings = [self.model.decode_token(t) for t in token_ids_list]

        resolved_positions = []
        for pos in positions:
            pos_idx = pos if pos >= 0 else seq_len + pos
            if pos_idx < 0 or pos_idx >= seq_len:
                raise ValueError(f"Position {pos} is out of bounds for seq_len={seq_len}")
            resolved_positions.append(pos_idx)

        console.print(f"[bold]Analyzing{'(with chat template)' if use_chat_template else ''}:[/bold] '{prompt}'")
        console.print(f"[dim]Tokens ({seq_len}): {token_strings[:5]}...{token_strings[-3:]}[/dim]")
        layer_indices = self._resolve_layers(layers)
        if len(layer_indices) == self.model.num_layers:
            layer_display = f"[0, 1, 2, ..., {self.model.num_layers - 1}]"
        else:
            layer_display = f"{layer_indices}"
        console.print(f"[dim]Captured layers: {layer_display}[/dim]")
        if len(resolved_positions) <= 6:
            pos_display = f"{resolved_positions}"
        else:
            pos_display = f"{resolved_positions[0]}..{resolved_positions[-1]}"
        console.print(f"[dim]Analyzing positions: {pos_display} ({len(resolved_positions)} tokens)[/dim]")
        console.print()

        layer_predictions_by_pos = {pos: [] for pos in resolved_positions}
        hidden_states = None
        use_nnsight = True
        try:
            probe = self.model.get_layer_module(0)
            use_nnsight = hasattr(probe, "output")
        except Exception:
            use_nnsight = False
        if steering is not None:
            use_nnsight = False
        if not use_nnsight:
            console.print("[yellow]nnsight tracing unavailable; using HF hidden_states.[/yellow]")
            hidden_states = self._get_all_hidden_states(input_ids, steering=steering)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing layers...", total=len(layer_indices))

            for layer_idx in layer_indices:
                progress.update(task, description=f"Layer {layer_idx}/{self.model.num_layers-1}")

                if hidden_states is None:
                    hidden = self._get_hidden_states_at_layer_efficient(layer_idx, input_ids)
                else:
                    hidden = self._select_hidden_layer(hidden_states, layer_idx)

                hidden_at_pos = hidden[:, resolved_positions, :]
                logits = self._apply_logit_lens_efficient(hidden_at_pos)

                probs = F.softmax(logits[0], dim=-1)
                top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

                for idx, pos in enumerate(resolved_positions):
                    top_tokens = top_indices[idx].tolist()
                    top_probs_list = top_probs[idx].tolist()
                    top_strings = [self.model.decode_token(t) for t in top_tokens]
                    layer_predictions_by_pos[pos].append(LayerPrediction(
                        layer_idx=layer_idx,
                        top_tokens=top_tokens,
                        top_probs=top_probs_list,
                        top_strings=top_strings,
                    ))

                progress.advance(task)

        position_results = []
        for pos in resolved_positions:
            preds = layer_predictions_by_pos[pos]
            final = preds[-1]
            peak_layer = self._find_peak_layer(preds, final.top_tokens[0])
            expected_peak_layer = None
            if expected_token_id is not None:
                expected_peak_layer = self._find_peak_layer(preds, expected_token_id)
            position_results.append(PositionAnalysis(
                position=pos,
                token_id=token_ids_list[pos],
                token=token_strings[pos],
                layer_predictions=preds,
                final_prediction=final.top_strings[0],
                final_prob=final.top_probs[0],
                peak_layer=peak_layer,
                expected_peak_layer=expected_peak_layer,
            ))

        clear_memory()

        return SequenceAnalysis(
            prompt=prompt,
            formatted_prompt=formatted,
            tokens=token_strings,
            token_ids=token_ids_list,
            positions=resolved_positions,
            position_results=position_results,
            expected_token=self.model.decode_token(expected_token_id) if expected_token_id is not None else None,
            expected_token_id=expected_token_id,
        )

    def _generate_full_ids(
        self,
        prompt: str,
        generate_tokens: int,
        use_chat_template: bool,
        steering: Optional[SteeringConfig] = None,
        generate_kwargs: Optional[dict] = None,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        if use_chat_template:
            formatted = self.model.apply_chat_template(prompt)
        else:
            formatted = prompt

        input_ids = self.model.tokenize(formatted)
        hf_model = self.model.get_hf_causal_lm()
        attention_mask = torch.ones_like(input_ids)
        kwargs = dict(generate_kwargs or {})
        kwargs.setdefault("max_new_tokens", generate_tokens)
        kwargs.setdefault("pad_token_id", self.model.tokenizer.pad_token_id)
        with self._steering_context(steering):
            full_ids = hf_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        if full_ids.dim() == 1:
            full_ids = full_ids.unsqueeze(0)
        if full_ids.device != input_ids.device:
            full_ids = full_ids.to(input_ids.device)

        return formatted, input_ids, full_ids

    def generate_sequence(
        self,
        prompt: str,
        generate_tokens: int = 30,
        use_chat_template: bool = True,
        steering: Optional[SteeringConfig] = None,
        generate_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, str, str, int]:
        """Generate tokens and return the full sequence for analysis."""
        formatted, input_ids, full_ids = self._generate_full_ids(
            prompt=prompt,
            generate_tokens=generate_tokens,
            use_chat_template=use_chat_template,
            steering=steering,
            generate_kwargs=generate_kwargs,
        )
        prompt_len = input_ids.shape[1]
        full_text = self.model.decode(full_ids[0])
        generated_text = self.model.decode(full_ids[0, prompt_len:])
        return full_ids, full_text, generated_text, prompt_len

    def detect_answer_position(
        self,
        prompt: str,
        generate_tokens: int = 30,
        use_chat_template: bool = True,
        steering: Optional[SteeringConfig] = None,
        generate_kwargs: Optional[dict] = None,
    ) -> AnswerDetection:
        """Generate and detect the most likely answer start position."""
        formatted, input_ids, full_ids = self._generate_full_ids(
            prompt=prompt,
            generate_tokens=generate_tokens,
            use_chat_template=use_chat_template,
            steering=steering,
            generate_kwargs=generate_kwargs,
        )

        prompt_len = input_ids.shape[1]
        if full_ids.shape[1] <= prompt_len:
            return AnswerDetection(
                input_ids=input_ids,
                analysis_position=max(prompt_len - 1, 0),
                analysis_text=formatted,
                generated_text="",
                answer_token="",
            )

        generated_ids = full_ids[0, prompt_len:]
        answer_offset = None
        answer_token = ""
        for idx, token_id in enumerate(generated_ids.tolist()):
            token_str = self.model.decode_token(token_id)
            if token_str.strip():
                answer_offset = idx
                answer_token = token_str
                break

        if answer_offset is None:
            answer_offset = generated_ids.shape[0] - 1
            answer_token = self.model.decode_token(generated_ids[answer_offset].item())

        analysis_position = prompt_len + answer_offset
        full_text = self.model.decode(full_ids[0])
        generated_text = self.model.decode(generated_ids)

        return AnswerDetection(
            input_ids=full_ids,
            analysis_position=analysis_position,
            analysis_text=full_text,
            generated_text=generated_text,
            answer_token=answer_token,
        )
    
    def _get_hidden_states_at_layer_efficient(self, layer_idx: int, input_ids: torch.Tensor) -> torch.Tensor:
        """Efficient hidden state extraction using nnsight."""
        model_inner = self.model.model
        
        with model_inner.trace(input_ids):
            layer = self.model.get_layer_module(layer_idx)
            # Capture output - typically (hidden_states, attention, ...)
            hidden = layer.output[0].save()
        
        return hidden.value

    def _apply_logit_lens_efficient(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply logit lens without tracing (direct computation)."""
        # Apply final norm
        norm = self.model.get_final_norm()
        normed = norm(hidden_states)
        
        # Apply lm_head
        lm_head = self.model.get_lm_head()
        logits = lm_head(normed)
        
        return logits

    def _get_all_hidden_states(self, input_ids: torch.Tensor, steering: Optional[SteeringConfig] = None):
        """Fetch all hidden states via HF forward as a fallback."""
        hf_model = self.model.get_hf_base_model()
        attention_mask = torch.ones_like(input_ids)
        with self._steering_context(steering):
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

    def _resolve_layers(self, layers: Optional[List[int]]) -> List[int]:
        if not layers:
            return list(range(self.model.num_layers))
        resolved = []
        for layer_idx in layers:
            idx = layer_idx
            if idx < 0:
                idx = self.model.num_layers + idx
            if idx < 0 or idx >= self.model.num_layers:
                raise ValueError(f"Layer {layer_idx} is out of bounds for num_layers={self.model.num_layers}")
            resolved.append(idx)
        return resolved
    
    def _find_peak_layer(self, predictions: List[LayerPrediction], target_token: int) -> Optional[int]:
        """Find the layer where target token first appears with >0.9 confidence."""
        for pred in predictions:
            for token, prob in zip(pred.top_tokens, pred.top_probs):
                if token == target_token and prob > 0.9:
                    return pred.layer_idx
        return None


def print_analysis_result(result: AnalysisResult, show_all: bool = True):
    """Pretty print the analysis result like in the video."""
    console.print()
    console.print("[bold]=== Final Prediction ===[/bold]")
    bar_len = int(result.final_prob * 50)
    bar = "#" * bar_len
    console.print(f"{result.final_prob:.4f} {bar} '{result.final_prediction}'")
    if result.expected_token is not None:
        console.print(f"[dim]Expected token: '{result.expected_token}'[/dim]")
    console.print()
    
    console.print("[bold]=== Logit Lens (top prediction at each layer) ===[/bold]")
    
    for pred in result.layer_predictions:
        token_str = pred.top_strings[0]
        prob = pred.top_probs[0]
        
        # Mark peak layer
        peak_marker = ""
        if result.peak_layer is not None and pred.layer_idx == result.peak_layer:
            peak_marker = " <- peak"
        expected_marker = ""
        if result.expected_peak_layer is not None and pred.layer_idx == result.expected_peak_layer:
            expected_marker = " <- expected"
        
        # Colorize based on probability
        if prob > 0.9:
            style = "bold green"
        elif prob > 0.5:
            style = "yellow"
        else:
            style = "dim"
        
        console.print(
            f"Layer {pred.layer_idx:2d}: [{style}]'{token_str}' ({prob:.4f})[/{style}]{peak_marker}{expected_marker}"
        )


def print_sequence_analysis(
    result: SequenceAnalysis,
    show_all: bool = True,
    group_by: str = "position",
    timeline_probs: bool = False,
    timeline_columns: bool = False,
):
    """Pretty print analysis results for multiple positions."""
    console.print()
    console.print("[bold]=== Logit Lens Timeline ===[/bold]")
    if result.expected_token is not None:
        console.print(f"[dim]Expected token: '{result.expected_token}'[/dim]")
    if group_by == "layer":
        tokens_display = [pos.token.replace("\n", "\\n") for pos in result.position_results]
        console.print(f"[dim]Positions: {result.positions}[/dim]")
        console.print(f"[dim]Tokens: {tokens_display}[/dim]")
        console.print()

        if not result.position_results:
            return
        num_layers = len(result.position_results[0].layer_predictions)
        chunk_size = 5
        if timeline_columns:
            for start in range(0, len(result.position_results), chunk_size):
                chunk = result.position_results[start:start + chunk_size]
                positions_label = [str(pos.position) for pos in chunk]
                table = Table(title=f"Positions {positions_label[0]}..{positions_label[-1]}")
                table.add_column("Layer", style="bold")
                for pos_result in chunk:
                    table.add_column(str(pos_result.position))

                for layer_offset in range(num_layers):
                    row = [f"{result.position_results[0].layer_predictions[layer_offset].layer_idx}"]
                    for pos_result in chunk:
                        pred = pos_result.layer_predictions[layer_offset]
                        token_str = pred.top_strings[0].replace("\n", "\\n")
                        if token_str.strip() == "":
                            token_str = "<ws>"
                        prob = pred.top_probs[0]

                        marker = ""
                        if pos_result.expected_peak_layer is not None and pred.layer_idx == pos_result.expected_peak_layer:
                            marker = "*"

                        if timeline_probs:
                            cell = f"{token_str} ({prob:.2f}){marker}"
                        else:
                            cell = f"{token_str}{marker}"
                        row.append(cell)
                    table.add_row(*row)
                console.print(table)
            return

        for layer_offset in range(num_layers):
            layer_idx = result.position_results[0].layer_predictions[layer_offset].layer_idx
            console.print(f"[bold]Layer {layer_idx:2d}[/bold]")
            for start in range(0, len(result.position_results), chunk_size):
                parts = []
                for pos_result in result.position_results[start:start + chunk_size]:
                    pred = pos_result.layer_predictions[layer_offset]
                    token_str = pred.top_strings[0].replace("\n", "\\n")
                    if token_str.strip() == "":
                        token_str = "<ws>"
                    prob = pred.top_probs[0]

                    marker = ""
                    if pos_result.expected_peak_layer is not None and pred.layer_idx == pos_result.expected_peak_layer:
                        marker = "*"

                    if prob > 0.9:
                        style = "bold green"
                    elif prob > 0.5:
                        style = "yellow"
                    else:
                        style = "dim"
                    if timeline_probs:
                        parts.append(
                            f"[{style}]{pos_result.position}:'{token_str}' ({prob:.2f}){marker}[/{style}]"
                        )
                    else:
                        parts.append(
                            f"[{style}]{pos_result.position}:'{token_str}'{marker}[/{style}]"
                        )

                console.print("  " + " | ".join(parts))
        return

    for pos_result in result.position_results:
        token_display = pos_result.token.replace("\n", "\\n")
        console.print()
        console.print(f"[bold]Position {pos_result.position}[/bold] token '{token_display}'")
        for pred in pos_result.layer_predictions:
            token_str = pred.top_strings[0].replace("\n", "\\n")
            prob = pred.top_probs[0]

            peak_marker = ""
            if pos_result.peak_layer is not None and pred.layer_idx == pos_result.peak_layer:
                peak_marker = " <- peak"
            expected_marker = ""
            if pos_result.expected_peak_layer is not None and pred.layer_idx == pos_result.expected_peak_layer:
                expected_marker = " <- expected"

            if prob > 0.9:
                style = "bold green"
            elif prob > 0.5:
                style = "yellow"
            else:
                style = "dim"

            console.print(
                f"Layer {pred.layer_idx:2d}: [{style}]'{token_str}' ({prob:.4f})[/{style}]{peak_marker}{expected_marker}"
            )
