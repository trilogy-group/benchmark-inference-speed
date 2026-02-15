# LLM Inference Benchmark

A Python benchmarking suite for measuring LLM inference performance, focusing on **Time to First Token (TTFT)** and **Tokens Per Second (TPS)**.

## Features

- **Multi-Provider Support**: Benchmark models from Kimi, Z.ai, Minimax, and Fireworks
- **Dual API Formats**: Supports both OpenAI Chat API and Anthropic Messages API
- **Diverse Prompts**: MT-Bench inspired prompts across 8 categories
- **Multi-Turn Support**: Test conversation context handling with follow-up prompts
- **Variable Input Length**: Short, medium, and long prompts for different prefill loads
- **Advanced Metrics**: Measures TTFT, TPS, decoding latency, and token counts
- **Statistical Analysis**: Reports mean, std, min, max, and percentiles (P50, P95, P99)
- **Warmup Phase**: Built-in warmup requests to ensure accurate measurements
- **Robust Error Handling**: Automatic retries with detailed error messages
- **JSON Export**: Save results to JSON for further analysis
- **Beautiful CLI**: Rich-formatted tables and progress bars

## Prompt Categories

The benchmark includes 24 prompts across 8 categories inspired by MT-Bench:

| Category | Description | Example |
|----------|-------------|---------|
| coding | Programming and code tasks | Refactoring, bug fixes, SQL |
| reasoning | Logic puzzles and reasoning | Race positions, family trees |
| math | Mathematical problems | Geometry, probability, algebra |
| extraction | Information extraction | Sentiment, entities |
| writing | Creative and professional writing | Emails, blogs, stories |
| roleplay | Character roleplay | Elon Musk, Sheldon Cooper |
| stem | Science and technology | Physics, biology, chemistry |
| humanities | Social sciences and business | Economics, history, business |

## Supported Providers

| Provider | Default Model | API Styles | Env Variable |
|----------|---------------|------------|--------------|
| Kimi | K2.5 | OpenAI | `KIMI_API_KEY` |
| Z.ai | glm-5 | OpenAI, Anthropic | `ZAI_API_KEY` |
| Minimax | MiniMax-M2.5-highspeed | OpenAI, Anthropic | `MINIMAX_API_KEY` |
| Fireworks | (none - specify with `--model`) | OpenAI | `FIREWORKS_API_KEY` |

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure API keys**:

   Create a `.env` file in the project root:
   ```env
   # At least one of these is required:
   KIMI_API_KEY=your_kimi_key
   ZAI_API_KEY=your_z_key
   MINIMAX_API_KEY=your_minimax_key
   FIREWORKS_API_KEY=your_fireworks_key
   ```

## Usage

### Basic Usage

```bash
# Benchmark Kimi with default settings
uv run benchmark.py --provider kimi

# Benchmark Minimax
uv run benchmark.py --provider minimax

# Benchmark with Anthropic API format
uv run benchmark.py --provider minimax --api-style anthropic
```

### CLI Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--provider` | Model provider to use (required) | - |
| `--model` | Model name | Provider's default |
| `--api-style` | API format: `openai` or `anthropic` | Provider's default |
| `--iterations` | Number of benchmark iterations | 12 |
| `--tokens` | Max tokens to generate | 512 |
| `--retries` | Retries per failed request | 3 |
| `--warmup` | Number of warmup requests before benchmarking | 2 |
| `--temperature` | Sampling temperature (0.0 = deterministic) | 0.0 |
| `--output` | Output file path for JSON results | (none) |
| `--quiet` | Suppress progress bars | false |
| `--randomize` | Randomize prompt order | false |
| `--multi-turn` | Test multi-turn conversations with follow-up prompts | false |
| `--category` | Filter prompts by category | all categories |

### Examples

```bash
# Quick test with 3 iterations
uv run benchmark.py --provider kimi --iterations 3

# Benchmark with custom model
uv run benchmark.py --provider fireworks --model accounts/your-account/deployment-id

# Higher token count for more stable TPS measurements
uv run benchmark.py --provider minimax --tokens 1024 --iterations 20

# Save results to JSON
uv run benchmark.py --provider minimax --output results.json

# Quiet mode for CI/automation
uv run benchmark.py --provider kimi --quiet

# Randomize prompts to avoid caching effects
uv run benchmark.py --provider z --randomize

# Test multi-turn conversations (with follow-up prompts)
uv run benchmark.py --provider minimax --multi-turn

# Test only coding prompts
uv run benchmark.py --provider kimi --category coding

# Test only math prompts
uv run benchmark.py --provider minimax --category math

# Test reasoning category with multi-turn
uv run benchmark.py --provider kimi --category reasoning --multi-turn
```

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to First Token - measures prefill latency |
| **TPS** | Tokens Per Second - measures decoding throughput |
| **Decode** | Decoding duration (total_time - TTFT) |
| **Tokens** | Total tokens generated |

### Percentiles

- **P50 (Median)**: 50% of requests complete faster than this
- **P95**: 95% of requests complete faster than this (captures slow outliers)
- **P99**: 99% of requests complete faster than this (captures worst case)

## Output Example

```
LLM Performance Benchmark
Provider: Minimax (M2.5-highspeed)
Model: MiniMax-M2.5-highspeed
API Style: openai
Iterations: 12 | Max Tokens: 512 | Warmup: 2
Temperature: 0.0

--- OpenAI Chat API ---
Completed 12/12 iterations successfully.

┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ Metric      ┃     Mean ± Std ┃    Min ┃     Max ┃    P50 ┃     P95 ┃     P99 ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ TTFT (s)    │ 1.2345 ± 0.123 │ 1.1023 │ 1.5678  │ 1.2034 │  1.4567 │  1.5234 │
│ TPS (tok/s) │   95.4 ± 10.2  │  75.3  │ 112.4   │  98.2  │  108.9  │  111.2  │
│ Decode (s)  │  5.432 ± 1.234 │  3.210 │  8.901  │  5.123 │   7.654 │   8.432 │
│ Tokens      │      512 ± 0   │    512 │     512 │      - │       - │       - │
└─────────────┴────────────────┴────────┴─────────┴────────┴─────────┴─────────┘
```

## Development

Run tests or check syntax:
```bash
uv run python3 -m py_compile benchmark.py
```
