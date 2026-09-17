#!/usr/bin/env python3
"""Export LatentSkill-generated ALFWorld adapters in PEFT/vLLM format."""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from evals.alfworld.evaluate import load_model, load_skill_contexts


PROJECTIONS = {
    "attention": {
        "q": "self_attn.q_proj",
        "k": "self_attn.k_proj",
        "v": "self_attn.v_proj",
        "o": "self_attn.o_proj",
    },
    "mlp": {
        "gate": "mlp.gate_proj",
        "up": "mlp.up_proj",
        "down": "mlp.down_proj",
    },
}
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--skill_context_dir", default="evals/alfworld/skills")
    parser.add_argument("--output_dir", default="artifacts/alfworld_skill_loras")
    parser.add_argument("--config_name", default="models/qwen3_8b")
    parser.add_argument("--context_max_length", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--model_parallel", action="store_true")
    return parser.parse_args()


def to_peft_state(adapter_state):
    state = {}
    rank = None
    for layer_index, layer_state in adapter_state.items():
        for group, projections in PROJECTIONS.items():
            for short_name, module_name in projections.items():
                leaf = layer_state[group][short_name]
                a = leaf["A"]
                b = leaf["B"]
                c = leaf.get("C")
                if a.shape[0] != 1 or b.shape[0] != 1:
                    raise ValueError("Export expects one generated adapter per skill")
                if c is not None and torch.count_nonzero(c).item() != 0:
                    raise ValueError("PEFT export does not support LatentSkill C bias")
                current_rank = a.shape[-1]
                rank = current_rank if rank is None else rank
                if current_rank != rank or b.shape[1] != rank:
                    raise ValueError("Inconsistent LoRA rank")
                prefix = (
                    f"base_model.model.model.layers.{layer_index}."
                    f"{module_name}"
                )
                # LatentSkill: x @ A[in,r] @ B[r,out].
                # PEFT: linear(linear(x, A[r,in]), B[out,r]).
                state[f"{prefix}.lora_A.weight"] = (
                    a[0].transpose(0, 1).contiguous().cpu()
                )
                state[f"{prefix}.lora_B.weight"] = (
                    b[0].transpose(0, 1).contiguous().cpu()
                )
    return state, rank


def main():
    args = parse_args()
    # load_model resolves these paths through the existing OmegaConf model config.
    import os
    os.environ["MODEL_PATH"] = str(Path(args.model_path).resolve())
    load_args = SimpleNamespace(
        raw_base=False,
        device="cuda",
        dtype=args.dtype,
        config_name=args.config_name,
        model_path=args.model_path,
        checkpoint=args.checkpoint,
        model_parallel=args.model_parallel,
        vllm_base_url=None,
    )
    hypernet, metalora, tokenizer, device, cfg, _ = load_model(load_args)
    contexts = load_skill_contexts(args.skill_context_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_model": str(Path(args.model_path).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dtype": args.dtype,
        "adapters": {},
    }

    for skill, context in contexts.items():
        encoded = tokenizer(
            context,
            max_length=args.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )
        with torch.inference_mode():
            adapter_state = hypernet.build_adapter_state(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
                metalora,
            )
        peft_state, rank = to_peft_state(adapter_state)
        skill_dir = output_root / skill
        skill_dir.mkdir(parents=True, exist_ok=True)
        save_file(peft_state, skill_dir / "adapter_model.safetensors")
        # vLLM asks Hugging Face for a tokenizer at every absolute LoRA path.
        # Without tokenizer assets, AutoTokenizer can silently construct a
        # 256-entry byte tokenizer from an adapter-only directory instead of
        # failing back to the backbone tokenizer.  Save the exact backbone
        # tokenizer beside each adapter so prompt IDs and decoded output stay
        # identical across the base model and every skill LoRA.
        tokenizer.save_pretrained(skill_dir)
        adapter_config = {
            "base_model_name_or_path": str(Path(args.model_path).resolve()),
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            # Scaling is already embedded as sqrt(scale) in both generated
            # factors, so PEFT/vLLM must apply an additional scale of exactly 1.
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "peft_type": "LORA",
            "r": rank,
            "target_modules": TARGET_MODULES,
            "task_type": "CAUSAL_LM",
        }
        (skill_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8"
        )
        manifest["adapters"][skill] = {
            "path": str(skill_dir.resolve()),
            "rank": rank,
            "tensor_count": len(peft_state),
        }
        print(f"[exported] {skill}: rank={rank}, tensors={len(peft_state)}")
        del adapter_state, peft_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
