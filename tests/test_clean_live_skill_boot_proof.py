from __future__ import annotations

from types import SimpleNamespace

from tools.closeout.prove_clean_live_skill_boot import (
    evaluate_skill_surfaces,
    is_competing_model_owner,
)


def _health() -> dict[str, object]:
    return {
        "backend": "python_ast",
        "digest": "a" * 64,
        "expected_live_count": 2,
        "live_count": 2,
        "missing_live": [],
        "parity_status": "matched",
        "quarantined": [],
        "quarantined_count": 0,
        "ready": True,
        "reason": "ready",
        "execution_preflight": {
            "complete": True,
            "failed": [],
            "ok": True,
        },
    }


def _surfaces() -> tuple[dict, dict, dict]:
    catalog = [{"name": "clock"}, {"name": "web_search"}]
    health = _health()
    return (
        {"tools": catalog, "count": 2, "health": health},
        {"catalog": catalog, "count": 2, "health": health},
        {
            "tools": catalog,
            "skill_catalog": health,
            "ui": {"status_flags": []},
        },
    )


def test_external_surface_evaluator_requires_one_ready_identical_catalog() -> None:
    result = evaluate_skill_surfaces(*_surfaces())

    assert result["passed"] is True
    assert result["catalog_count"] == 2
    assert all(result["checks"].values())


def test_external_surface_evaluator_rejects_hidden_quarantine_and_ui_blocker() -> None:
    tools, skills, bootstrap = _surfaces()
    bootstrap["skill_catalog"] = {
        **bootstrap["skill_catalog"],
        "quarantined": [{"name": "web_search"}],
        "quarantined_count": 1,
        "ready": False,
    }
    bootstrap["ui"]["status_flags"] = ["skill_quarantined"]

    result = evaluate_skill_surfaces(tools, skills, bootstrap)

    assert result["passed"] is False
    assert result["checks"]["health_identical"] is False
    assert result["checks"]["ui_has_no_skill_blocker"] is False


def test_external_surface_evaluator_rejects_catalog_divergence() -> None:
    tools, skills, bootstrap = _surfaces()
    bootstrap["tools"] = [{"name": "clock"}]

    result = evaluate_skill_surfaces(tools, skills, bootstrap)

    assert result["passed"] is False
    assert result["checks"]["catalogs_identical"] is False


def test_model_owner_classifier_matches_argv_not_unrelated_python() -> None:
    evaluator = SimpleNamespace(
        pid=1,
        create_time=1.0,
        cmdline=("python", "/repo/tools/evaluate_unified_intrinsic_decoding.py"),
    )
    audit = SimpleNamespace(
        pid=2,
        create_time=1.0,
        cmdline=("python", "/repo/tools/closeout/audit_skill_catalog.py"),
    )

    assert is_competing_model_owner(evaluator) is True
    assert is_competing_model_owner(audit) is False


class TestTheParityCheckAsksForAValueTheCatalogCanReport:
    """`catalog_parity_ready` asked for "ready" and nothing ever says it.

    `core.skills.discovery` reports one of six words and "ready" is not among
    them, so the check was False on every boot whatever the Rust and Python
    catalogs agreed about — and it was the one check failing the clean-live
    boot proof. This test passed throughout, because the fixture above wrote
    "ready" into a health dict by hand: green because of the drift, not
    despite it.
    """

    def test_every_state_the_proof_accepts_is_one_the_catalog_reports(self) -> None:
        from core.skills.discovery import PARITY_STATES, PARITY_STATES_THAT_AGREE

        assert PARITY_STATES_THAT_AGREE <= PARITY_STATES

    def test_the_named_states_are_the_ones_the_catalog_reports(self, tmp_path, monkeypatch) -> None:
        """Every outcome the catalog builder has, driven, and what each reports.

        The set that comes back must be the named set exactly: a state no
        path reports is a name nothing can satisfy, and a reported state
        nobody named is the drift this class exists for. A new outcome in
        the builder needs a case here to be counted.
        """
        import core.skills.discovery as discovery

        root = tmp_path / "skills"
        root.mkdir()
        (root / "one.py").write_text(
            "from core.skills.base_skill import BaseSkill\n"
            "class OneSkill(BaseSkill):\n"
            "    name = 'one'\n"
            "    description = 'One.'\n"
            "    effect_scope = 'pure_compute'\n"
            "    async def execute(self, params, context): return {'ok': True}\n",
            encoding="utf-8",
        )
        roots = (discovery.SkillSourceRoot(root, "fixture", "project"),)
        monkeypatch.setattr(discovery, "_load_rust_builder", lambda: None)
        monkeypatch.setattr(discovery, "_load_rust_discoverer", lambda: None)

        def broken(_payload):
            raise RuntimeError("the rust catalog fell over")

        reported = {
            discovery.build_skill_catalog(roots, try_rust=False).parity_status,
            discovery.build_skill_catalog(roots).parity_status,
            discovery.build_skill_catalog(roots, rust_builder=broken).parity_status,
            discovery.build_skill_catalog(
                roots, rust_builder=lambda _p: '{"accepted":[],"duplicates":[]}'
            ).parity_status,
            discovery.build_skill_catalog(
                roots, rust_builder=discovery.canonicalize_skill_candidates
            ).parity_status,
        }
        # A filesystem discovery that agrees with Python's.
        monkeypatch.setattr(discovery, "_filesystem_parity_projection", lambda _payload: "same")
        reported.add(
            discovery.build_skill_catalog(
                roots,
                rust_builder=discovery.canonicalize_skill_candidates,
                rust_discoverer=lambda _roots: "{}",
            ).parity_status
        )

        assert reported == discovery.PARITY_STATES, (
            f"reported but unnamed: {sorted(reported - discovery.PARITY_STATES)}; "
            f"named but never reported: {sorted(discovery.PARITY_STATES - reported)}"
        )

    def test_the_proof_refuses_a_diverged_catalog(self) -> None:
        from tools.closeout.prove_clean_live_skill_boot import evaluate_skill_surfaces

        tools_payload, skills_payload, bootstrap = _surfaces()
        for payload in (tools_payload, skills_payload):
            payload["health"]["parity_status"] = "diverged"
        bootstrap["skill_catalog"]["parity_status"] = "diverged"

        report = evaluate_skill_surfaces(tools_payload, skills_payload, bootstrap)

        assert report["checks"]["catalog_parity_ready"] is False
