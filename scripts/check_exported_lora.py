#!/usr/bin/env python3
"""Check that an exported skill LoRA changes a Hugging Face backbone correctly."""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--prompt", default="Say hello.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map={"": "cuda:0"},
        local_files_only=True,
    )
    model.eval()
    encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda:0")

    with torch.inference_mode():
        base_logits = model(**encoded).logits[:, -1].float().cpu()

    model = PeftModel.from_pretrained(
        model,
        args.adapter_path,
        is_trainable=False,
        autocast_adapter_dtype=False,
    )
    model.eval()
    lora_parameters = [
        parameter for name, parameter in model.named_parameters() if "lora_" in name
    ]
    print(
        f"adapter_parameters={len(lora_parameters)} "
        f"dtype={lora_parameters[0].dtype if lora_parameters else None}"
    )

    with torch.inference_mode():
        adapter_logits = model(**encoded).logits[:, -1].float().cpu()
        adapter_ids = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
        with model.disable_adapter():
            disabled_logits = model(**encoded).logits[:, -1].float().cpu()
            base_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )

    print(f"base_vs_disabled_max_abs={torch.max(torch.abs(base_logits - disabled_logits)).item():.8f}")
    print(f"base_vs_adapter_max_abs={torch.max(torch.abs(base_logits - adapter_logits)).item():.8f}")
    print(f"base_argmax={base_logits.argmax().item()} adapter_argmax={adapter_logits.argmax().item()}")
    prompt_length = encoded["input_ids"].shape[1]
    print("[base]")
    print(repr(tokenizer.decode(base_ids[0, prompt_length:], skip_special_tokens=False)))
    print("[adapter]")
    print(repr(tokenizer.decode(adapter_ids[0, prompt_length:], skip_special_tokens=False)))


if __name__ == "__main__":
    main()
