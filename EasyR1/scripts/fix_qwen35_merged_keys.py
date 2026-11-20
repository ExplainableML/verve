"""Normalize a merged qwen3_5 checkpoint's key names to the HF layout.

WHY. EasyR1's training-time state dict nests the vision tower inside the text
model -- `model.language_model.visual.*` -- while the HF class (and therefore
the official checkpoint, and therefore every vLLM loader) puts them side by
side as `model.visual.*`. model_merger.py writes the state dict through
`save_pretrained(state_dict=...)`, which stores keys verbatim, so the merged
checkpoint inherits the wrong path. vLLM then reports EVERY `visual.*` weight
as "not initialized from checkpoint" and refuses to start -- the tensors are
all present and correct, only their names are wrong (verified 2026-08-23 on
verify_asym20k/global_step_52: 760 keys, 333 of them the misplaced vision
subtree).

The rename is safe because it is checked, not assumed: the resulting key set
must equal the key set of a freshly instantiated model, exactly, or this exits
non-zero without writing anything.

    python3 scripts/fix_qwen35_merged_keys.py --hf_dir <ckpt>/actor/huggingface
"""

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForImageTextToText

RENAME = ("model.language_model.visual.", "model.visual.")


def expected_keys(hf_dir: str) -> set:
    cfg = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(cfg)
    return set(model.state_dict().keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dir", required=True)
    ap.add_argument("--backup", action="store_true", help="keep the original as *.orig")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.hf_dir, "*.safetensors")))
    if not shards:
        raise SystemExit(f"no safetensors in {args.hf_dir}")

    tensors, needs_fix = {}, 0
    for path in shards:
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                nk = k.replace(*RENAME) if k.startswith(RENAME[0]) else k
                needs_fix += nk != k
                tensors[nk] = f.get_tensor(k)

    if needs_fix == 0:
        print(f"[fix-keys] {args.hf_dir}: already in HF layout ({len(tensors)} keys), nothing to do")
        return

    want = expected_keys(args.hf_dir)
    got = set(tensors)
    if got != want:
        raise SystemExit(
            f"[fix-keys] REFUSING to write: renamed keys do not match the model.\n"
            f"  missing: {sorted(want - got)[:5]}\n  unexpected: {sorted(got - want)[:5]}"
        )

    if args.backup:
        for path in shards:
            shutil.copy2(path, path + ".orig")
    for path in shards:
        os.remove(path)
    idx = os.path.join(args.hf_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        os.remove(idx)

    out = os.path.join(args.hf_dir, "model.safetensors")
    save_file(tensors, out, metadata={"format": "pt"})
    print(f"[fix-keys] {args.hf_dir}: renamed {needs_fix} vision keys, "
          f"{len(tensors)} total, verified against the model -> {out}")


if __name__ == "__main__":
    main()
