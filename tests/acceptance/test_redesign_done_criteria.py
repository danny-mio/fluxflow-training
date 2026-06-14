"""Acceptance criteria for the v0.10.0 bezier-coupled redesign.

These are placeholders to be filled in after the full retraining run (operational
phase M8 of the design plan). They run only on the reference GPU and gate the
PR to ``develop``.

See docs/plans/2026-06-13-v0.10.0-redesign-design.md §7.2 for the criteria.
"""

import pytest


@pytest.mark.acceptance
@pytest.mark.gpu
@pytest.mark.slow
def test_acceptance_a_vae_lpips_at_d32_matches_baseline_d128():
    """Reconstruction LPIPS at D=32 <= v0.10.0-pre LPIPS at D=128 +/-10%.

    To be implemented after M8 retraining. Reads baseline from
    tests/perf/baselines.json (also to be created).
    """
    pytest.skip("Acceptance test pending retraining; see design plan §7.2")


@pytest.mark.acceptance
@pytest.mark.gpu
@pytest.mark.slow
def test_acceptance_b_flow_text_understanding_clip():
    """Paired prompts CLIP delta >= 0.05 on a 50-prompt validation set.

    To be implemented after M8 retraining.
    """
    pytest.skip("Acceptance test pending retraining; see design plan §7.2")


@pytest.mark.acceptance
@pytest.mark.gpu
@pytest.mark.slow
def test_acceptance_c_compute_budget():
    """VAE+Flow throughput regression <= 25%, peak GPU memory <= 24 GB at 1024^2.

    To be implemented after M8 retraining.
    """
    pytest.skip("Acceptance test pending retraining; see design plan §7.2")
