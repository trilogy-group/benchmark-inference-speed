import os
import time
import json
import random
import statistics
import argparse
import numpy as np
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

load_dotenv()
console = Console()

# --- Provider Configuration ---

PROVIDERS = {
    "kimi": {
        "name": "Kimi (K2.5)",
        "default_model": "K2.5",
        "api_style": "openai",
        "base_url": "https://api.kimi.com/coding/v1",
        "api_key_env": "KIMI_API_KEY",
        "endpoints": {
            "chat": "/chat/completions",
            "completion": "/completions"
        }
    },
    "z": {
        "name": "Z.ai (GLM-5)",
        "default_model": "glm-5",
        "api_style": "openai",
        "base_url": "https://api.z.ai/api/coding/paas/v4",
        "api_key_env": "ZAI_API_KEY",
        "anthropic_url": "https://api.z.ai/api/anthropic",
        "endpoints": {
            "chat": "/chat/completions",
            "completion": "/completions"
        }
    },
    "minimax": {
        "name": "Minimax (M2.5-highspeed)",
        "default_model": "MiniMax-M2.5-highspeed",
        "api_style": "openai",
        "base_url": "https://api.minimax.io/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "anthropic_url": "https://api.minimax.io/anthropic/v1",
        "endpoints": {
            "chat": "/chat/completions",
            "completion": "/completions"
        }
    },
    "fireworks": {
        "name": "Fireworks",
        "default_model": None,  # No default - must be provided
        "api_style": "openai",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "endpoints": {
            "chat": "/chat/completions",
            "completion": "/completions"
        }
    },
    "openrouter": {
        "name": "OpenRouter",
        "default_model": "qwen/qwen3.5-plus-02-15",
        "api_style": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "anthropic_url": "https://openrouter.ai/api",
        "endpoints": {
            "chat": "/chat/completions"
        }
    }
}


def get_api_key(provider_config):
    """Get API key from environment or .env, with .env taking precedence."""
    env_var = provider_config["api_key_env"]
    # load_dotenv already loaded .env, so os.getenv will check .env first, then env
    return os.getenv(env_var)


# --- Configuration & Prompts ---

# MT-Bench inspired prompts with variety across 8 categories
# Each prompt can have a follow-up for multi-turn testing
PROMPTS = [
    # === SHORT PROMPTS (coding-focused) ===
    {
        "name": "Code: Refactor",
        "category": "coding",
        "prompt": "Refactor this Python function to be more concise:\n\ndef find_max(numbers):\n    max_val = numbers[0]\n    for n in numbers:\n        if n > max_val:\n            max_val = n\n    return max_val",
        "follow_up": "Now use a lambda function."
    },
    {
        "name": "Code: FastAPI",
        "category": "coding",
        "prompt": "Write a complete FastAPI endpoint for handling user registration with Pydantic validation and password hashing using passlib.",
        "follow_up": "Add email verification functionality."
    },
    {
        "name": "Code: Bug Fix",
        "category": "coding",
        "prompt": "Find the bug in this code:\n\nasync def main():\n    workers = [worker(i) for i in range(5)]\n    await workers\n\nasyncio.run(main())",
        "follow_up": "How would you fix it using asyncio.gather?"
    },
    {
        "name": "Code: SQL Migration",
        "category": "coding",
        "prompt": "Write a migration script to add a 'deleted_at' column for soft deletes to these tables:\n\nCREATE TABLE users (id SERIAL PRIMARY KEY, email VARCHAR(255) UNIQUE);\nCREATE TABLE posts (id SERIAL PRIMARY KEY, user_id INT REFERENCES users(id));",
        "follow_up": "Also add an index on deleted_at."
    },
    
    # === MEDIUM PROMPTS (reasoning) ===
    {
        "name": "Reasoning: Race",
        "category": "reasoning",
        "prompt": "If you overtake the second person in a race, what's your position? Where is the person you overtook?",
        "follow_up": "What if you overtake the last person?"
    },
    {
        "name": "Reasoning: Family",
        "category": "reasoning",
        "prompt": "David has three sisters. Each of them has one brother. How many brothers does David have?",
        "follow_up": "If each sister has two brothers, how many brothers does David have?"
    },
    {
        "name": "Reasoning: Logic",
        "category": "reasoning",
        "prompt": "Oranges cost more than apples. Oranges cost less than bananas. Bananas cost more than apples. Which statement is true: Bananas cost more than oranges?",
        "follow_up": "If bananas cost exactly the same as oranges, what would be true?"
    },
    
    # === MATH PROMPTS ===
    {
        "name": "Math: Geometry",
        "category": "math",
        "prompt": "Calculate the area of a triangle with vertices at (0, 0), (-1, 1), and (3, 3).",
        "follow_up": "What is the area of the circumscribed circle?"
    },
    {
        "name": "Math: Probability",
        "category": "math",
        "prompt": "When rolling two dice, what is the probability of rolling a total of at least 3?",
        "follow_up": "What is the probability of rolling an even number or at least 3?"
    },
    {
        "name": "Math: Algebra",
        "category": "math",
        "prompt": "Given x+y = 4z and x*y = 4z^2, express x-y in terms of z.",
        "follow_up": "Express z-x in terms of y."
    },
    
    # === EXTRACTION PROMPTS ===
    {
        "name": "Extraction: Reviews",
        "category": "extraction",
        "prompt": "Extract the sentiment from these reviews as a JSON array (1-5 scale):\n1. 'Absolutely phenomenal! Best movie ever.'\n2. 'Terrible, waste of time.'\n3. 'It was okay, some good parts.'",
        "follow_up": "Return it as a Python list instead."
    },
    {
        "name": "Extraction: Entities",
        "category": "extraction",
        "prompt": "Extract all names, dates, and locations from: 'John Smith visited Paris on March 15, 2024 to meet with Dr. Sarah Johnson from Google.'",
        "follow_up": "Format as a JSON object with keys: names, dates, locations."
    },
    
    # === WRITING PROMPTS (long input) ===
    {
        "name": "Writing: Email",
        "category": "writing",
        "prompt": "Write a professional email to your supervisor seeking feedback on the Quarterly Financial Report. Ask specifically about data analysis, presentation style, and clarity of conclusions. Keep it concise.",
        "follow_up": "Now rewrite it with a more persuasive tone."
    },
    {
        "name": "Writing: Blog",
        "category": "writing",
        "prompt": "Compose a travel blog post about a trip to Hawaii, highlighting cultural experiences and must-see attractions. Include vivid sensory details about the food, music, and landscapes.",
        "follow_up": "Add a section about budget travel tips."
    },
    {
        "name": "Writing: Story",
        "category": "writing",
        "prompt": "Write a captivating short story beginning with: 'The old abandoned house at the end of the street held a secret that no one had ever discovered.' Make it at least three paragraphs.",
        "follow_up": "Rewrite it using only four-word sentences."
    },
    
    # === ROLEPLAY PROMPTS (long input) ===
    {
        "name": "Roleplay: Elon Musk",
        "category": "roleplay",
        "prompt": "Pretend you are Elon Musk. Answer: Why do we need to go to Mars?",
        "follow_up": "How do you like dancing? Can you teach me?"
    },
    {
        "name": "Roleplay: Sheldon",
        "category": "roleplay",
        "prompt": "Act as Sheldon from 'The Big Bang Theory'. Answer: What is your opinion on hand dryers?",
        "follow_up": "Would you like to grab dinner? Take the bus with me?"
    },
    {
        "name": "Roleplay: ML Engineer",
        "category": "roleplay",
        "prompt": "You are a machine learning engineer. Explain what a language model is in simple terms. Is it trained on labeled or unlabeled data?",
        "follow_up": "Is this true? I heard other companies use different approaches."
    },
    
    # === STEM PROMPTS (long input) ===
    {
        "name": "STEM: Physics",
        "category": "stem",
        "prompt": "Explain the orbital mechanics of satellites. If a satellite's orbital radius increases, what happens to its orbital period?",
        "follow_up": "What are some edge cases in this relationship?"
    },
    {
        "name": "STEM: Biology",
        "category": "stem",
        "prompt": "Outline the two main stages of photosynthesis, including where they occur in the chloroplast and the inputs/outputs for each stage.",
        "follow_up": "How much energy can a tree produce in its lifetime? Estimate with numerical values."
    },
    {
        "name": "STEM: Chemistry",
        "category": "stem",
        "prompt": "Describe the reaction between solid calcium carbonate and hydrochloric acid. Write the balanced equation and identify the reaction type.",
        "follow_up": "How can we reverse this process?"
    },
    
    # === HUMANITIES PROMPTS (long input) ===
    {
        "name": "Humanities: Economics",
        "category": "humanities",
        "prompt": "Explain the correlation between GDP, inflation, and unemployment. How do fiscal and monetary policies affect these indicators?",
        "follow_up": "Explain this like I'm five years old."
    },
    {
        "name": "Humanities: History",
        "category": "humanities",
        "prompt": "What methods did Socrates use to challenge prevailing thoughts of his time?",
        "follow_up": "Generate a conversation between Socrates and Bill Gates debating generative AI in education."
    },
    {
        "name": "Humanities: Business",
        "category": "humanities",
        "prompt": "Discuss antitrust laws and their impact on market competition. Compare US and China antitrust laws with case studies.",
        "follow_up": "Explain one case study in detail."
    },
]


def calculate_percentiles(data, percentiles=[50, 95, 99]):
    """Calculate percentiles for a list of values using numpy."""
    if not data:
        return {p: None for p in percentiles}
    result = {}
    for p in percentiles:
        result[p] = float(np.percentile(data, p))
    return result


def benchmark_stream(url, payload, api_key, max_retries=3, retry_delay=5):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    
    for attempt in range(max_retries):
        start_time = time.time()
        ttft = None
        total_tokens = 0
        chunks_received = 0
        first_token_time = None
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True, timeout=120)
            
            if response.status_code == 503:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {"error": "503 Service Unavailable"}
            
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * 2)
                    continue
                return {"error": "429 Rate Limited"}
            
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    current_time = time.time()
                    if ttft is None:
                        ttft = current_time - start_time
                        first_token_time = ttft
                    
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data_content)
                            chunks_received += 1
                            if "usage" in chunk and chunk.get("usage"):
                                total_tokens = chunk["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            decoding_duration = total_duration - ttft if ttft else 0
            
            tokens = total_tokens if total_tokens > 0 else max(chunks_received - 1, 1)
            
            tps = tokens / decoding_duration if decoding_duration > 0.001 else 0
            
            return {
                "ttft": ttft,
                "tps": tps,
                "total_tokens": tokens,
                "total_duration": total_duration,
                "decoding_duration": decoding_duration,
                "chunks": chunks_received
            }
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return {"error": "Request Timeout"}
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return {"error": str(e)}


def benchmark_stream_anthropic(url, payload, api_key, max_retries=3, retry_delay=5):
    """Benchmark using Anthropic API format."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    # Convert OpenAI-style payload to Anthropic format
    anthropic_payload = {
        "model": payload.get("model"),
        "max_tokens": payload.get("max_tokens", 512),
        "stream": True,
    }
    
    # Handle messages (OpenAI format)
    if "messages" in payload:
        # Extract content from messages
        messages = payload["messages"]
        # Convert to Anthropic format - use last user message as input
        for msg in reversed(messages):
            if msg.get("role") == "user":
                anthropic_payload["messages"] = [{"role": "user", "content": msg["content"]}]
                break
    elif "prompt" in payload:
        anthropic_payload["prompt"] = payload["prompt"]
    
    if "temperature" in payload:
        anthropic_payload["temperature"] = payload["temperature"]
    
    for attempt in range(max_retries):
        start_time = time.time()
        ttft = None
        total_tokens = 0
        chunks_received = 0
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(anthropic_payload), stream=True, timeout=120)
            
            if response.status_code == 503:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {"error": "503 Service Unavailable"}
            
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * 2)
                    continue
                return {"error": "429 Rate Limited"}
            
            if response.status_code >= 400:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_body = response.json()
                    if "error" in error_body:
                        error_msg = f"{error_msg}: {error_body['error']}"
                    elif "message" in error_body:
                        error_msg = f"{error_msg}: {error_body['message']}"
                except:
                    error_msg = f"{error_msg}: {response.text[:200]}"
                return {"error": error_msg}
            
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    if ttft is None:
                        ttft = time.time() - start_time
                    
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data_content)
                            chunks_received += 1
                            # Anthropic usage
                            if "usage" in chunk and chunk.get("usage"):
                                total_tokens = chunk["usage"].get("output_tokens", 0)
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            decoding_duration = total_duration - ttft if ttft else 0
            
            tokens = total_tokens if total_tokens > 0 else max(chunks_received - 1, 1)
            
            tps = tokens / decoding_duration if decoding_duration > 0.001 else 0
            
            return {
                "ttft": ttft,
                "tps": tps,
                "total_tokens": tokens,
                "total_duration": total_duration,
                "decoding_duration": decoding_duration,
                "chunks": chunks_received
            }
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return {"error": "Request Timeout"}
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return {"error": str(e)}


def run_warmup(url, api_key, model, warmup_count=2, max_tokens=64, use_anthropic=False):
    """Run warmup requests to prime the model."""
    console.print(f"[dim]Running {warmup_count} warmup requests...[/]")
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    
    if use_anthropic:
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        warmup_payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": "Say 'warmup' in one word."}],
            "stream": True
        }
    else:
        headers["Authorization"] = f"Bearer {api_key}"
        warmup_payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": "Say 'warmup' in one word."}],
            "stream": True
        }
    
    for i in range(warmup_count):
        try:
            requests.post(url, headers=headers, data=json.dumps(warmup_payload), stream=True, timeout=30)
        except Exception:
            pass


def run_benchmarks():
    parser = argparse.ArgumentParser(description="LLM Benchmark for TTFT, TPS, and Decoding Speed.")
    parser.add_argument("--provider", type=str, required=True, 
                        choices=list(PROVIDERS.keys()),
                        help="Model provider to benchmark.")
    parser.add_argument("--model", type=str, default=None, 
                        help="Model name (defaults to provider's default).")
    parser.add_argument("--api-style", type=str, default=None,
                        choices=["openai", "anthropic"],
                        help="API format to use (overrides provider default).")
    parser.add_argument("--iterations", type=int, default=12, help="Number of benchmark iterations.")
    parser.add_argument("--tokens", type=int, default=512, help="Max tokens to generate.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per request.")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup requests before benchmarking.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for deterministic).")
    parser.add_argument("--output", type=str, default=None, help="Output file path for results (JSON).")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bars.")
    parser.add_argument("--randomize", action="store_true", help="Randomize prompt order.")
    parser.add_argument("--multi-turn", action="store_true", help="Test multi-turn conversations with follow-up prompts.")
    parser.add_argument("--category", type=str, default=None, 
                        choices=["coding", "reasoning", "math", "extraction", "writing", "roleplay", "stem", "humanities"],
                        help="Filter prompts by category.")
    
    args = parser.parse_args()
    
    provider = PROVIDERS[args.provider]
    model = args.model or provider["default_model"]
    
    if not model:
        console.print(f"[bold red]Error:[/] No model specified and {args.provider} has no default model. Use --model.")
        return
    
    api_key = get_api_key(provider)
    if not api_key:
        console.print(f"[bold red]Error:[/] {provider['api_key_env']} not found in environment or .env")
        return

    # Determine API style (openai or anthropic)
    api_style = args.api_style or provider.get("api_style", "openai")
    
    # Build URLs based on API style
    if api_style == "anthropic":
        chat_url = provider.get("anthropic_url")
        if not chat_url:
            console.print(f"[bold red]Error:[/] {args.provider} does not support Anthropic API format.")
            return
        chat_url = chat_url + "/messages"
    else:
        chat_url = provider["base_url"] + provider["endpoints"]["chat"]

    console.print(f"\n[bold blue]LLM Performance Benchmark[/]")
    console.print(f"[dim]Provider:[/] {provider['name']}")
    console.print(f"[dim]Model:[/] {model}")
    console.print(f"[dim]API Style:[/] {api_style}")
    console.print(f"[dim]Iterations:[/] {args.iterations} | [dim]Max Tokens:[/] {args.tokens} | [dim]Warmup:[/] {args.warmup}")
    console.print(f"[dim]Temperature:[/] {args.temperature}\n")

    api_name = "Anthropic Messages API" if api_style == "anthropic" else "OpenAI Chat API"
    url = chat_url

    console.print(f"[dim]URL:[/] {url}")

    console.print(f"[bold cyan]--- {api_name} ---[/]")
    
    # Run warmup if requested
    if args.warmup > 0:
        run_warmup(url, api_key, model, warmup_count=args.warmup, max_tokens=32, use_anthropic=(api_style == "anthropic"))
    
    results = []
    prompt_indices = list(range(len(PROMPTS)))
    
    # Filter by category if specified
    if args.category:
        prompt_indices = [i for i in prompt_indices if PROMPTS[i].get("category") == args.category]
        if not prompt_indices:
            console.print(f"[bold red]Error:[/] No prompts found for category '{args.category}'")
            return
        console.print(f"[dim]Filtered to {len(prompt_indices)} prompts in category '{args.category}'[/]")
    
    if args.randomize:
        random.shuffle(prompt_indices)
    
    iteration_order = []
    for i in range(args.iterations):
        if args.randomize and i > 0:
            iteration_order.append(random.choice(prompt_indices))
        else:
            iteration_order.append(prompt_indices[i % len(PROMPTS)])
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=not args.quiet
    ) as progress:
        task = progress.add_task(f"Running {api_name}...", total=args.iterations)
        
        for i in range(args.iterations):
            prompt_idx = iteration_order[i]
            prompt_obj = PROMPTS[prompt_idx]
            
            # Build messages - handle multi-turn if enabled
            if args.multi_turn and prompt_obj.get("follow_up"):
                # Multi-turn: include initial prompt + follow-up as conversation
                messages = [
                    {"role": "user", "content": prompt_obj["prompt"]},
                    {"role": "assistant", "content": "This is a placeholder response for context."},
                    {"role": "user", "content": prompt_obj["follow_up"]}
                ]
                prompt_name = f"{prompt_obj['name']} (multi-turn)"
            else:
                messages = [{"role": "user", "content": prompt_obj["prompt"]}]
                prompt_name = prompt_obj["name"]
            
            payload = {
                "model": model,
                "max_tokens": args.tokens,
                "temperature": args.temperature,
                "messages": messages
            }

            # Use appropriate benchmark function based on API style
            if api_style == "anthropic":
                res = benchmark_stream_anthropic(url, payload, api_key, max_retries=args.retries)
            else:
                res = benchmark_stream(url, payload, api_key, max_retries=args.retries)
            
            if "error" not in res:
                results.append({
                    **res,
                    "prompt_name": prompt_name,
                    "category": prompt_obj.get("category", "unknown")
                })
            else:
                console.print(f"[dim]Iteration {i+1} failed: {res['error']}[/]")
            
            progress.update(task, advance=1)

    if results:
        ttfts = [r["ttft"] for r in results]
        tpss = [r["tps"] for r in results]
        decoding = [r["decoding_duration"] for r in results]
        tokens_list = [r["total_tokens"] for r in results]
        
        ttft_pcts = calculate_percentiles(ttfts)
        tps_pcts = calculate_percentiles(tpss)
        decoding_pcts = calculate_percentiles(decoding)
        
        benchmark_data = {
            "ttft": {
                "mean": np.mean(ttfts),
                "std": np.std(ttfts),
                "min": min(ttfts),
                "max": max(ttfts),
                "p50": ttft_pcts[50],
                "p95": ttft_pcts[95],
                "p99": ttft_pcts[99],
            },
            "tps": {
                "mean": np.mean(tpss),
                "std": np.std(tpss),
                "min": min(tpss),
                "max": max(tpss),
                "p50": tps_pcts[50],
                "p95": tps_pcts[95],
                "p99": tps_pcts[99],
            },
            "decoding_duration": {
                "mean": np.mean(decoding),
                "std": np.std(decoding),
                "min": min(decoding),
                "max": max(decoding),
                "p50": decoding_pcts[50],
                "p95": decoding_pcts[95],
                "p99": decoding_pcts[99],
            },
            "total_tokens": {
                "mean": np.mean(tokens_list),
                "std": np.std(tokens_list),
                "min": min(tokens_list),
                "max": max(tokens_list),
            },
            "count": len(results)
        }

        console.print(f"Completed {len(results)}/{args.iterations} iterations successfully.\n")

    if results:
        table = Table(title=f"Inference Performance ({provider['name']})", header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean ± Std", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("P50", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("P99", justify="right")

        table.add_row(
            "TTFT (s)",
            f"{benchmark_data['ttft']['mean']:.4f} ± {benchmark_data['ttft']['std']:.3f}",
            f"{benchmark_data['ttft']['min']:.4f}",
            f"{benchmark_data['ttft']['max']:.4f}",
            f"{benchmark_data['ttft']['p50']:.4f}",
            f"{benchmark_data['ttft']['p95']:.4f}",
            f"{benchmark_data['ttft']['p99']:.4f}"
        )
        table.add_row(
            "TPS (tok/s)",
            f"{benchmark_data['tps']['mean']:.1f} ± {benchmark_data['tps']['std']:.1f}",
            f"{benchmark_data['tps']['min']:.1f}",
            f"{benchmark_data['tps']['max']:.1f}",
            f"{benchmark_data['tps']['p50']:.1f}",
            f"{benchmark_data['tps']['p95']:.1f}",
            f"{benchmark_data['tps']['p99']:.1f}"
        )
        table.add_row(
            "Decode (s)",
            f"{benchmark_data['decoding_duration']['mean']:.3f} ± {benchmark_data['decoding_duration']['std']:.3f}",
            f"{benchmark_data['decoding_duration']['min']:.3f}",
            f"{benchmark_data['decoding_duration']['max']:.3f}",
            f"{benchmark_data['decoding_duration']['p50']:.3f}",
            f"{benchmark_data['decoding_duration']['p95']:.3f}",
            f"{benchmark_data['decoding_duration']['p99']:.3f}"
        )
        table.add_row(
            "Tokens",
            f"{benchmark_data['total_tokens']['mean']:.0f} ± {benchmark_data['total_tokens']['std']:.0f}",
            str(benchmark_data['total_tokens']['min']),
            str(benchmark_data['total_tokens']['max']),
            "-",
            "-",
            "-"
        )

        console.print(table)
        
        if args.output:
            output_data = {
                "config": {
                    "provider": args.provider,
                    "provider_name": provider["name"],
                    "model": model,
                    "api_style": api_style,
                    "iterations": args.iterations,
                    "max_tokens": args.tokens,
                    "warmup": args.warmup,
                    "temperature": args.temperature,
                },
                "results": benchmark_data
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            console.print(f"\n[green]Results saved to {args.output}[/]")
        
        console.print("\n[dim]Note: TTFT is measured from request start to first token.[/]")
    else:
        console.print("[bold red]No successful benchmark results to report.[/]")


if __name__ == "__main__":
    run_benchmarks()
