# Lens 🔍

Influenced by [The Eiffel Tower Llama](https://huggingface.co/spaces/dlouapre/eiffel-tower-llama).

## Features

- **Logit Lens**: See top token predictions at every layer
- **Timeline Analysis**: Track layer-wise predictions across generated tokens
- **Answer Detection**: Auto-find where the model produces the answer
- **Neuron Discovery**: Find neurons that distinguish concepts (pos vs neg prompts)
- **Steering**: Inject vectors or clamp neurons to change model behavior
- **Filtering + Sampling**: Analyze specific layers/positions and run multiple samples

## Installation

```bash
# Clone and install
git clone <repo>
cd lens
pip install -e .
```

## Usage

```bash
# Basic logit lens analysis
lens analyze -m Qwen/Qwen2.5-7B -p "10*10="

# With answer detection
lens analyze -m Qwen/Qwen2.5-7B -p "What is 5+5?" --find-answer

# Timeline across generated tokens (group by layer)
lens analyze -m Qwen/Qwen2.5-7B -p "10*10=" --trace-generation --generate 30 --group-by layer

# Layer-grouped timeline with probabilities
lens analyze -m Qwen/Qwen2.5-7B -p "10*10=" --trace-generation --generate 30 --group-by layer --timeline-probs

# Layer-grouped timeline as columns
lens analyze -m Qwen/Qwen2.5-7B -p "10*10=" --trace-generation --generate 30 --group-by layer --timeline-columns

# Analyze a range of positions and specific layers
lens analyze -m Qwen/Qwen2.5-7B -p "10*10=" --range 14:30 --layers 10,15,20

# Sample multiple runs
lens analyze -m Qwen/Qwen2.5-7B -p "10*10=" --trace-generation --generate 30 -n 3 --temperature 1.0

# Find concept neurons
lens diff -m Qwen/Qwen2.5-7B --pos "complex math" --neg "simple addition" --layer 15 --save math_direction.pt

# Steer the model (neuron)
lens steer -m Qwen/Qwen2.5-7B -p "The answer is" --layer 15 --neuron 808 --boost 5.0

# Steer the model (vector)
lens steer -m Qwen/Qwen2.5-7B -p "The answer is" --layer 15 --vector math_direction.pt --coeff 2.0

# Steer during analysis
lens analyze -m Qwen/Qwen2.5-7B -p "10*10=" --trace-generation --generate 30 \
  --steer-layer 15 --steer-neuron 808 --steer-strength 5
```

## Hardware

Designed for systems with large RAM (tested on Strix Halo with 128GB shared memory).
Supports quantization (Q4/Q8) for larger models.

## Tech Stack

- **nnsight**: Direct HuggingFace model access with hooks
- **transformers**: Model loading with chat templates
- **typer**: CLI interface
- **rich**: Pretty terminal output
