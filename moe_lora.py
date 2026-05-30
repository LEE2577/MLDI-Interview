"""Mixture-of-Experts LoRA (MoE-LoRA) implementation.

Each matching Linear layer gets num_experts LoRA adapter pairs (A_i, B_i)
plus a lightweight router that produces per-token mixing weights.

Forward:  out = W·x + scale * Σ_i( g_i(x) · B_i·A_i·x )
Merge:    avg expert delta folded back into W for inference.
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        num_experts: int,
        top_k: int = None,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.num_experts = num_experts
        self.top_k = top_k if top_k is not None else num_experts  # dense by default
        self.scale = lora_alpha / r

        d_in = base_layer.in_features
        d_out = base_layer.out_features

        # Expert matrices: lora_A [E, r, d_in], lora_B [E, d_out, r]
        self.lora_A = nn.Parameter(torch.zeros(num_experts, r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, d_out, r))
        self.router = nn.Linear(d_in, num_experts, bias=False)
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # kaiming for A, zeros for B (same init convention as standard LoRA)
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))

        for param in base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        # LoRA computation in fp32 for stability; cast result back to base dtype
        x_lora = x.to(self.lora_A.dtype)
        gate = F.softmax(self.router(x_lora), dim=-1)  # [..., E]

        if self.top_k < self.num_experts:
            topk_vals, topk_idx = torch.topk(gate, self.top_k, dim=-1)
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
            gate = torch.zeros_like(gate).scatter_(-1, topk_idx, topk_vals)

        orig_shape = x_lora.shape
        x_flat = self.dropout(x_lora).reshape(-1, orig_shape[-1])  # [N, d_in]
        gate_flat = gate.reshape(-1, self.num_experts)              # [N, E]

        ax = torch.einsum("ni,eri->enr", x_flat, self.lora_A)      # [E, N, r]
        bax = torch.einsum("enr,eor->eno", ax, self.lora_B)        # [E, N, d_out]
        lora_out = torch.einsum("ne,eno->no", gate_flat, bax)      # [N, d_out]
        lora_out = lora_out.reshape(*orig_shape[:-1], self.base_layer.out_features)

        return base_out + self.scale * lora_out.to(base_out.dtype)

    def merge(self) -> nn.Linear:
        """Average all expert deltas and fold into base weight. Returns the plain Linear."""
        with torch.no_grad():
            # [E, d_out, d_in]
            deltas = torch.bmm(self.lora_B.float(), self.lora_A.float())
            avg_delta = deltas.mean(0) * self.scale
            self.base_layer.weight.data += avg_delta.to(self.base_layer.weight.dtype)
        return self.base_layer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_parent_and_attr(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_moe_lora(
    model: nn.Module,
    target_modules: list,
    r: int,
    lora_alpha: float,
    lora_dropout: float,
    num_experts: int,
    top_k: int = None,
) -> int:
    """Replace all matching nn.Linear layers with MoELoRALinear. Returns count."""
    target_set = set(target_modules)
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf not in target_set:
            continue
        parent, attr = _get_parent_and_attr(model, name)
        setattr(
            parent,
            attr,
            MoELoRALinear(
                module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                num_experts=num_experts,
                top_k=top_k,
            ),
        )
        replaced += 1
    return replaced


def print_trainable_parameters(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {trainable:,} || "
        f"all params: {total:,} || "
        f"trainable%: {100 * trainable / total:.4f}"
    )


def save_moe_lora(model: nn.Module, save_dir: str, cfg: dict):
    os.makedirs(save_dir, exist_ok=True)
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, MoELoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            state[f"{name}.router.weight"] = module.router.weight.detach().cpu()
    torch.save(state, os.path.join(save_dir, "moe_lora_weights.pt"))

    meta = {
        "type": "moe_lora",
        "r": int(cfg["lora_r"]),
        "lora_alpha": float(cfg["lora_alpha"]),
        "lora_dropout": float(cfg.get("lora_dropout", 0.0)),
        "num_experts": int(cfg["moe_num_experts"]),
        "top_k": cfg.get("moe_top_k"),
        "target_modules": list(cfg["target_modules"]),
    }
    with open(os.path.join(save_dir, "moe_lora_config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] MoE-LoRA adapter saved to {save_dir}")


def load_and_merge_moe_lora(base_model: nn.Module, adapter_dir: str) -> nn.Module:
    with open(os.path.join(adapter_dir, "moe_lora_config.json")) as f:
        meta = json.load(f)
    state = torch.load(
        os.path.join(adapter_dir, "moe_lora_weights.pt"), map_location="cpu"
    )

    inject_moe_lora(
        base_model,
        target_modules=meta["target_modules"],
        r=meta["r"],
        lora_alpha=meta["lora_alpha"],
        lora_dropout=0.0,
        num_experts=meta["num_experts"],
        top_k=meta.get("top_k"),
    )

    for name, module in base_model.named_modules():
        if not isinstance(module, MoELoRALinear):
            continue
        module.lora_A.data.copy_(state[f"{name}.lora_A"])
        module.lora_B.data.copy_(state[f"{name}.lora_B"])
        module.router.weight.data.copy_(state[f"{name}.router.weight"])

    names_to_merge = [
        n for n, m in base_model.named_modules() if isinstance(m, MoELoRALinear)
    ]
    for name in names_to_merge:
        parent, attr = _get_parent_and_attr(base_model, name)
        setattr(parent, attr, getattr(parent, attr).merge())

    print(f"[INFO] Merged {len(names_to_merge)} MoE-LoRA layers into base weights")
    return base_model
