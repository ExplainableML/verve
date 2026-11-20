# VERVE: Verified Visual Search

Code for **VERVE: Verified Visual Search**.

- Models: https://huggingface.co/collections/xiaorui638/verve-models
- Training data: https://huggingface.co/datasets/xiaorui638/VERVE-Data

## Training

Training lives in [`EasyR1/`](EasyR1), a fork of [hiyouga/EasyR1](https://github.com/hiyouga/EasyR1)
(GRPO on top of veRL) at upstream commit `dd71bbd` plus
[PR #11001](https://github.com/hiyouga/EasyR1/pull/11001) for Qwen3.5. One tree trains every backbone:

| `MODEL` | Backbone |
|---|---|
| `qwen25vl` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` |
| `qwen35` | `Qwen/Qwen3.5-9B` |
| `mimo` | `XiaomiMiMo/MiMo-VL-7B-SFT-2508` |

and both modes:

| `MODE` | Reward | Prompt |
|---|---|---|
| `plain` | `examples/reward_function/perception_multinode.py` | `examples/format_prompt/perception.jinja` |
| `opsv` | `examples/reward_function/perception_verify.py` | `examples/format_prompt/perception_verify2_1.jinja` |

### Setup

```bash
cd EasyR1
pip install -r requirements.txt     # vllm 0.18, torch 2.10
pip install transformers==5.5.3     # after vllm, which declares transformers<5
```

This is the environment we tested (Python 3.12). `verl` is imported from the `EasyR1/` directory, so
there is nothing else to install.

Download [VERVE-Data](https://huggingface.co/datasets/xiaorui638/VERVE-Data) and extract the image archives
as its card describes. The policy and judge models must be in the local Hugging Face cache (`HF_HOME`).

### Run

`train_verve.sh` is a SLURM launcher for 7 nodes with 4 GPUs each: 6 nodes train, 1 node serves the reward
judge (`Qwen/Qwen3-30B-A3B-Instruct-2507`). Set your account and partition in its header, then:

```bash
cd EasyR1
MODEL=qwen3vl MODE=opsv \
TRAIN_PARQUET=/data/VERVE-Data/train_s57_opsv.parquet \
IMAGE_DIR=/data/VERVE-Data/images \
HF_HOME=/path/to/hf_cache \
sbatch train_verve.sh
```

Use `train_s57_plain.parquet` with `MODE=plain`. The shared cluster logic (Ray cluster, judge servers,
resume across the walltime limit) is in `train_common.sh`; every hyperparameter is a config override there
or in `examples/perception_config.yaml`.

`ANNEAL="START END"` (OPSV only) retires the verification terms of the reward linearly between the two
steps, so training ends on the plain accuracy reward. It defaults to `"40 80"` for `qwen35` and is off for
the other backbones, which matches the released checkpoints.

### Merging a checkpoint

```bash
python scripts/model_merger.py --local_dir verl_exp/<run>/global_step_<N>/actor
```

For Qwen3.5, also run `python scripts/fix_qwen35_merged_keys.py --hf_dir <...>/actor/huggingface` afterwards.

## Acknowledgements

Built on [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl); the
training recipe starts from [Zooming without Zooming](https://github.com/inclusionAI/Zooming-without-Zooming).
