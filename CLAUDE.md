# CLAUDE.md - FluxFlow Training

See `AGENTS.md` for the canonical agent and coordinator conventions used in
this repo. This file holds Claude-specific notes layered on top of that.

## v0.10.0 Bezier-Coupled Redesign (in progress on feature/model-v0.10.0)

The v0.10.0 redesign coordinates with `fluxflow-core` v0.10.0 and ships the
training-side glue for the five locked decisions:

1. **Per-token text** — flow trainer consumes `(text_seq, text_mask)` from
   `BertTextEncoder` instead of pooled vectors.
2. **Conditional ctx coupling** — VAE trainer optimizes a new
   `ctx_shrinkage_weight` term to keep ctx informative without collapsing.
3. **Full flow modernization** — flow trainer wires 2D RoPE + dual FiLM
   downstream of `FluxFlowProcessor_v100`.
4. **Multi-scale SPADE** — VAE trainer respects the new decoder topology with
   no config changes required beyond the latent-dim drop.
5. **Clean Gaussian z** — `kl_z_weight` (cosine warmup) and a new
   `t_txt` (text-token max length, default 32) are exposed in the YAML config.

New config keys (see `config.example.yaml`):
- `kl_z_weight` — final KL weight on the clean Gaussian latent.
- `ctx_shrinkage_weight` — L2-like pressure on `ctx` to prevent runaway scale.
- `t_txt` — DistilBERT max text length (32 for v0.10.0, was 512).

CFG empty-prompt substitution at flow-train time goes through
`fluxflow_training.training.cfg_utils.apply_cfg_null_substitution`
(pairs with `fluxflow.utils.visualization.build_cfg_null_pair`).

Plans live in `fluxflow-core/docs/plans/2026-06-13-v0.10.0-redesign-design.md`
and `-implementation.md`. Per-repo milestone tag: `m5-training-pipeline`.

Acceptance criteria stubs: `tests/acceptance/test_redesign_done_criteria.py`
(marked `acceptance, gpu, slow`; filled in after the M8 retraining run).
