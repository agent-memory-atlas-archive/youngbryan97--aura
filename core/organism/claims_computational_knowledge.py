"""Register the measured boundary of scoped calculation cells."""


def _scoped_computation_estimates_stay_conditional():
    from core.reasoning.computational_knowledge import _estimate_is_not_exact
    receipt = _estimate_is_not_exact()
    return (receipt["hard_constraint"] is False
            and receipt["source_interpretation_proven"] is False
            and receipt["general_transfer_proven"] is False)


def install_computational_knowledge_claims(suite):
    from core.organism.model_validation import (
        Claim,
        Evidence,
        Observation,
        ValidationTest,
        boolean_score,
    )

    name = "scoped_computation_keeps_estimates_conditional"
    owner = "core/reasoning/computational_knowledge.py"
    suite.add_test(ValidationTest(
        name=name, description="an estimated speed yields a conditional enclosing distance interval",
        required_capability="scoped_computational_knowledge",
        observation=Observation("conditional_distance_interval", True,
            "tests/test_computational_knowledge.py; exact interval [9,11]*2=[18,22]"),
        predict=lambda _model: _scoped_computation_estimates_stay_conditional(),
        score=lambda value, observation: boolean_score(value, expected=observation.value, subject=name),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Scoped calculation cells keep estimated input consequences conditional.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Measured rational interval arithmetic and premise taint; no autonomous source grounding, "
                      "live answer improvement, or general-transfer claim follows.",
    ))
