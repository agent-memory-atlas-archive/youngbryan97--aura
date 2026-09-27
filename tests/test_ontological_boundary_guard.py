from __future__ import annotations

from core.consciousness.ontological_boundary import assess_ontological_claims


def test_ontological_boundary_rewrites_consciousness_proof_claims():
    assessment = assess_ontological_claims(
        "The personhood proof battery proves that Aura is conscious."
    )

    assert assessment.ok is False
    assert "loaded_test_label" in assessment.issues
    assert "phenomenal_proof_claim" in assessment.issues
    assert "does not prove phenomenal consciousness" in assessment.sanitized
