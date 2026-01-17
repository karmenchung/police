import argparse
import os
import sys
import torch

try:
    import accelerate
except Exception:
    accelerate = None

from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as hf_version


default_system = "你是一个可靠的翻译助手。翻译到中文，不需要额外解释。"
default_prompt = "I am a police, I am going to arrest you."


def main():
    parser = argparse.ArgumentParser(description="Run a simple LLM")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Model name or path")
    parser.add_argument("--prompt", default=default_prompt, help="User prompt text")
    parser.add_argument("--system", default=default_system, help="System prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU not available. Install a CUDA-enabled PyTorch and GPU driver.", file=sys.stderr)
        raise SystemExit(2)

    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    text = run_prompt(args.prompt, args.system, tokenizer, model, args.max_new_tokens)
    print("----------------")
    print(text)


def run_prompt(prompt, system, tokenizer, model, max_new_tokens):
    messages = [ 
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    input_len = inputs.shape[-1]
    new_tokens = output[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    main()
