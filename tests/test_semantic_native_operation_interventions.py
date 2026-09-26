"""Operation interventions change the question's meaning without changing values."""

import hashlib
import json

from tools.semantic_native_operation_interventions import build_native_operation_interventions
from core.learning.semantic_program_corpus_natural import build_semantic_program_natural_request_corpus

_SEED = 2718283


def test_interventions_change_one_operation_and_preserve_public_grounding():
    pairs = build_native_operation_interventions(seed=_SEED)
    assert len(pairs) == 24
    assert len({pair.original.construction_id for pair in pairs}) == 24
    for pair in pairs:
        original, changed = pair.original, pair.changed
        assert pair.changed_instruction == 2
        assert original.inputs == changed.inputs
        assert original.program.instructions[:-1] == changed.program.instructions[:-1]
        assert original.program.instructions[-1].args == changed.program.instructions[-1].args
        assert original.program.instructions[-1].op != changed.program.instructions[-1].op
        assert original.program.run(original.inputs) != changed.program.run(changed.inputs)


def test_interventions_are_not_the_exposed_development_sources():
    exposed = {hashlib.sha256(example.source_text.encode()).hexdigest()
               for example in build_semantic_program_natural_request_corpus(examples_per_schema_domain=3)}
    pairs = build_native_operation_interventions(seed=_SEED)
    source_hashes = [hashlib.sha256(example.source_text.encode()).hexdigest()
                     for pair in pairs for example in (pair.original, pair.changed)]
    assert len(set(source_hashes)) == 48
    assert not set(source_hashes) & exposed
    assert hashlib.sha256(json.dumps(source_hashes, separators=(",", ":")).encode()).hexdigest() == (
        "a60ec5c814509c47824b009faead357c8439621e9ca51cd3e9ce71e7bfef618b"
    )


def test_interventions_are_repeatable_but_seed_sensitive():
    assert build_native_operation_interventions(seed=_SEED) == build_native_operation_interventions(seed=_SEED)
    assert build_native_operation_interventions(seed=_SEED) != build_native_operation_interventions(seed=_SEED + 2)


def test_null_effect_intervention_is_rejected_before_model_execution():
    import pytest

    with pytest.raises(ValueError, match="changed outcome"):
        build_native_operation_interventions(seed=2718281)
