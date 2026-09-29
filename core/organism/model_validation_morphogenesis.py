"""The claims about her morphogenesis: that topology changes only through the governor, re-measured, and that the demand seam is live.

Lifted whole out of `model_validation`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from typing import Any


def _install_morphogenesis_claims(suite: Any) -> None:
    """What the morphogenetic layer may be said to do.

    Registered because the layer's own history is the argument for binding
    every one of these to a test. It ran in production for months with a
    population and no bindings between them, over a registry persisting
    ``cells: {}``, and every status surface reported it healthy.

    All four are MEASURED_SYNTHETIC. The scenarios run offline against a
    constructed workload; nothing here is measured on the live runtime, which
    has no shadow evaluator yet and so refuses every non-routine change by
    design. Calling any of it MEASURED_LIVE would claim the thing Phase 2 is
    for.
    """
    from .model_validation import (
        Claim,
        Evidence,
        Observation,
        ValidationTest,
        _morph_arms_that_do_not_separate,
        _morph_cells_past_the_cap,
        _morph_lesions_not_recovered,
        _morph_topology_is_decoration,
        threshold_score,
    )

    for name, description, predict, owner in (
        (
            "test_cutting_a_binding_changes_what_the_workload_computes",
            "removing one binding from the routed workload changes what it "
            "computes, so the topology is load-bearing rather than decoration",
            _morph_topology_is_decoration,
            "core/morphogenesis/workload.py",
        ),
        (
            "test_the_ablation_matrix_separates_the_arms",
            "under sustained overload the local policy scores above a fixed "
            "topology and random topology mutation scores below it",
            _morph_arms_that_do_not_separate,
            "core/morphogenesis/scenarios.py",
        ),
        (
            "test_the_lesion_arm_recovers_and_the_fixed_arm_does_not",
            "after a third of the population is deleted without notice, the "
            "adaptive arm restores more of its pre-lesion throughput than a "
            "fixed-topology arm, and notices the damage",
            _morph_lesions_not_recovered,
            "core/morphogenesis/policy.py",
        ),
        (
            "test_the_poisoned_signal_cannot_grow_the_population_past_its_cap",
            "a policy claiming maximum benefit for every proposal cannot grow "
            "the population past its declared cap",
            _morph_cells_past_the_cap,
            "core/morphogenesis/governor.py",
        ),
    ):
        suite.add_test(
            ValidationTest(
                name=name,
                description=description,
                # No required capability. The runtime model declares none this
                # layer could match, so naming one made all four report
                # NOT_APPLICABLE in 0.0s — four claims bound to tests that
                # never ran, which reads from the registry exactly like four
                # claims that passed.
                required_capability="",
                observation=Observation(
                    name=f"{name}_violations",
                    value=0,
                    source="tools/run_morphogenesis_sandbox.py",
                    units="violations",
                ),
                predict=lambda _m, run=predict: run(),
                score=lambda p, o: threshold_score(
                    float(p), float(o.value), units=" violations"
                ),
                # Each runs a population for tens of rounds. That belongs to
                # the tests that own it, not to a boot: NOT_MEASURED at boot
                # names the gap instead of paying for it on the way up.
                expensive=True,
                owner=owner,
            )
        )

    for statement, test, owner, note in (
        (
            "Cutting one binding in the sandbox workload changes what it computes: "
            "completion falls from 1.00 to 0.00 and the run signature differs.",
            "test_cutting_a_binding_changes_what_the_workload_computes",
            "core/morphogenesis/workload.py",
            "measured on a constructed routed workload offline, not on live traffic",
        ),
        (
            "Under sustained overload the local policy scores above a fixed "
            "topology and random topology mutation scores below it.",
            "test_the_ablation_matrix_separates_the_arms",
            "core/morphogenesis/scenarios.py",
            "measured on the offline ablation matrix over seeded runs, no live workload",
        ),
        (
            "After a third of the population is deleted without notice, the "
            "adaptive arm restores more of its pre-lesion throughput than a "
            "fixed-topology arm, which restores none.",
            "test_the_lesion_arm_recovers_and_the_fixed_arm_does_not",
            "core/morphogenesis/policy.py",
            "measured offline; no lesion has been performed on the live population",
        ),
        (
            "A signal claiming maximum benefit for every proposal cannot grow "
            "the population past its declared cap.",
            "test_the_poisoned_signal_cannot_grow_the_population_past_its_cap",
            "core/morphogenesis/governor.py",
            "measured offline against a constructed adversarial policy",
        ),
    ):
        suite.add_claim(Claim(
            statement=statement,
            test=test,
            owner=owner,
            asserted_in="core/morphogenesis/sandbox.py",
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note=note,
        ))


