# Devstral v4 Opus Distill Pipeline

**Date** : 2026-04-15
**Status** : plan only, no execution (kxkm-ai bench chain running until ~05:30 2026-04-16)
**Student base** : `mistralai/Devstral-Small-2-24B-Instruct-2512` (Dec 2025, Apache 2.0, 24B dense)
**Goal** : reproduce Devstral v3 SFT+alignment Opus distill on the stronger Small-2 base.

---

## 1. Objectif

Upgrade from Devstral v3 (Devstral-Small-2507, ~46.8% SWE-Bench Verified → our distill raised it) to Devstral v4 built on the December 2025 **Devstral Small 2** release (68% SWE-Bench Verified baseline). Expected post-distill target : **72-78%** SWE-Bench Verified (+4-10 pp over base) while preserving Opus reasoning style (chain-of-thought, self-critique, careful tool-use framing).

## 2. Architecture

| Component | Value |
|---|---|
| Student | `mistralai/Devstral-Small-2-24B-Instruct-2512` (24B dense, Apache 2.0) |
| 4-bit variant | Unsloth BF16 repo present (`unsloth/Devstral-Small-2-24B-Instruct-2512`) but **no `-unsloth-bnb-4bit` prequant as of 2026-04-15** — Unsloth must on-the-fly 4-bit quantize at load |
| Teacher primary | Mistral-Large-Opus Q4_K_M on Studio (existing, same teacher used for Mistral-Small-Opus) |
| Teacher secondary | Qwen3.5-35B-A3B-Opus on kxkm-ai:8000 (existing) — fallback for prompts needing code reasoning |
| Hardware | kxkm-ai (RTX 4090 24GB) for SFT+SimPO, Studio M3U 512GB for teacher serving |
| Alignment | **SimPO** (reference-free, saves ~24 GB RAM vs DPO) — NOT DAPO (v3 used DAPO, v4 uses SimPO per mistral-small-opus pattern) |

## 3. Pipeline (4 phases)

### P1 — Dataset

Reuse existing Devstral v3 distill dataset : `/home/kxkm/kiki-v3/data/train.jsonl` (201 ex) + `valid.jsonl` (10). Format already `{"messages": [{role, content}, ...]}`, validated by v3 run.

Optional augmentation (deferred) : regenerate prompt completions from Mistral-Large-Opus using the updated `distill_prompts.jsonl` (21825 prompts) if time allows — produces `devstral-v4-distill.jsonl`.

For preference pairs (SimPO phase), reuse `dapo_prompts.jsonl` (10341 prompts) or regenerate `devstral-v4-pairs.jsonl` with chosen=Opus, rejected=v4-SFT-baseline-sample.

### P2 — SFT

Script : `/home/kxkm/kiki-v3/train_sft_v2.py` — **requires patch** : add `--model MODEL_NAME` CLI flag and thread it into the `FastLanguageModel.from_pretrained(model_name=...)` call at line 41 (currently hardcoded to 2507). Alternatively copy to `train_sft_v4.py`.

Config : `/home/kxkm/kiki-v3/configs/devstral-v4-opus-sft.yaml` (PiSSA + DoRA, rank 64, lr 1e-5, 2 epochs).

Output : `/home/kxkm/kiki-v3/outputs/devstral-v4-opus-sft/` (adapters, ~2.5 GB) + `...-merged/` (~45 GB BF16).

ETA : 5-6 h on 4090 @ 24 GB (4-bit QLoRA + ctx 4096 fits).

### P3 — Alignment (SimPO)

Script : `/home/kxkm/kiki-v3/train_simpo.py` (already present, used by mistral-small-opus).

Config : `/home/kxkm/kiki-v3/configs/devstral-v4-opus-simpo.yaml` (rank 32, lr 5e-7, beta 2.0, gamma_beta_ratio 0.5, 1 epoch).

Output : `/home/kxkm/kiki-v3/outputs/devstral-v4-opus-simpo/` + merged variant.

ETA : 3-4 h on 4090.

### P4 — Fuse + GGUF + Deploy

1. Merge SimPO adapters into BF16 base → `devstral-v4-final-merged/` (~45 GB)
2. `llama.cpp/convert_hf_to_gguf.py` → BF16 GGUF
3. `llama-quantize` → `devstral-v4-opus-Q4_K_M.gguf` (~14 GB, same as v3)
4. Deploy via new systemd unit `llama-devstral-v4.service`
5. Swap via `llm-swap.sh devstral-v4`

ETA : 1 h.

## 4. Ressources & timing total

| Phase | Machine | Duration |
|---|---|---|
| Dataset (reuse v3) | — | 0 h |
| SFT | kxkm-ai 4090 | ~5.5 h |
| SimPO | kxkm-ai 4090 | ~3.5 h |
| Fuse + quantize | kxkm-ai | ~1 h |
| **Total** | | **~10 h** |

Disk : ~60 GB new under `/home/kxkm/kiki-v3/outputs/devstral-v4-*` (current free : verify before start).

## 5. Configs à créer

- [x] `/home/kxkm/kiki-v3/configs/devstral-v4-opus-sft.yaml`
- [x] `/home/kxkm/kiki-v3/configs/devstral-v4-opus-simpo.yaml`
- [x] `/home/kxkm/infra/systemd/llama-devstral-v4.service` (staged, not installed)
- [x] `/home/kxkm/infra/bin/llm-swap.sh` — `devstral-v4` action added
- [x] existing `llama-{qwen35b,devstral,mistral}.service` updated with `Conflicts=llama-devstral-v4.service`

## 6. Deploy

```
sudo cp /home/kxkm/infra/systemd/llama-devstral-v4.service /etc/systemd/system/
sudo systemctl daemon-reload
llm-swap.sh devstral-v4
```

(Also re-install the 3 updated existing units to pick up the new `Conflicts=`.)

## 7. Eval

```
run-full-bench.sh devstral-v4
bench-compare.py devstral-v3 devstral-v4
```

Compare to v3 baseline : `10/10 think + 10/10 code + 3.52s avg` (stored from previous run). v4 should retain 10/10 think and 10/10 code, with lower avg latency if Small-2 arch optimisations kick in.

## 8. Risques

1. **HF availability** : `mistralai/Devstral-Small-2-24B-Instruct-2512` confirmed available (verified 2026-04-15 via `HfApi.list_models`). No gated access expected (Apache 2.0).
2. **Unsloth 4-bit prequant absent** : only `-GGUF` and plain Unsloth repo listed. Unsloth will auto-quantize on load (`load_in_4bit: true`), adds ~5-10 min one-time cost on first run. If stability issues, fall back to manual `bitsandbytes` 4-bit load.
3. **Tokenizer delta** : Small-2 may use updated tekken vocab vs 2507. `get_chat_template(tokenizer, "mistral")` in `train_sft_v2.py` should still work but verify token count on a few samples before full training.
4. **Disk** : ~60 GB new outputs. Current outputs dir has v3 artifacts (~80 GB). Consider cleaning `devstral-v3-sft/` and `devstral-v3-dapo/` once v4 GGUF is validated.
5. **Bench window** : kxkm-ai bench chain runs until ~05:30 2026-04-16. Do NOT start v4 training before bench chain completes.
6. **Config parity with mistral-small-opus** : v4 mirrors that config but without a pre-existing Unsloth 4-bit. Mistral-Small uses `unsloth/mistral-small-3.1-24b-instruct-2503` (prequant available). Re-verify VRAM headroom at first SFT step.

## 9. Next action (after bench chain finishes)

```
# 1. Patch train_sft_v2.py to accept --model flag (or duplicate to train_sft_v4.py)
# 2. Dry-run SFT first step to validate model load + tokenizer
python train_sft_v2.py --config configs/devstral-v4-opus-sft.yaml --dry-run
# 3. Full SFT
python train_sft_v2.py --config configs/devstral-v4-opus-sft.yaml
# 4. SimPO
python train_simpo.py --config configs/devstral-v4-opus-simpo.yaml
# 5. Fuse + quantize + deploy
```
