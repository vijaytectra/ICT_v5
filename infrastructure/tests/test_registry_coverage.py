"""
Registry coverage guardrail.

Enforces that the colocated markdown specs (under ``concepts/`` and
``models/``) and the code detectors stay in sync. Importing the
``concepts`` package and the model packages triggers every ``@concept``
registration as a side effect, so by the time these tests run the registry
reflects all implemented detectors.
"""
import concepts  # noqa: F401  (side effect: registers all concept detectors)
import models.ict  # noqa: F401  (registers model-2022, sniper-model, daily-bias)
import models.intermediate  # noqa: F401  (registers daily-bias)
from infrastructure.registry import load_concepts, detectors


def test_implemented_concepts_have_a_detector():
    """No concept may claim `detection: implemented` without a registered detector."""
    concepts = load_concepts()
    reg = detectors()
    missing = sorted(
        slug for slug, meta in concepts.items()
        if meta.get("detection") == "implemented" and slug not in reg
    )
    assert not missing, (
        "concepts marked `detection: implemented` with no registered detector: "
        + ", ".join(missing)
    )


def test_registered_detectors_are_marked_implemented_or_partial():
    """No detector may be registered against a concept the docs still call unbuilt."""
    concepts = load_concepts()
    problems = []
    for slug in detectors():
        meta = concepts.get(slug)
        if meta is None:
            problems.append(f"{slug}: detector registered but no knowledge file")
        elif meta.get("detection") not in {"implemented", "partial"}:
            problems.append(
                f"{slug}: detector registered but flag is {meta.get('detection')!r}"
            )
    assert not problems, "; ".join(problems)


def test_knowledge_frontmatter_is_well_formed():
    """Every concept file must carry a category and a valid detection flag."""
    concepts = load_concepts()
    assert len(concepts) > 50, f"only {len(concepts)} concepts loaded — spec dirs not found?"
    valid = {"implemented", "partial", "not-implemented"}
    for slug, meta in concepts.items():
        assert meta.get("category"), f"{slug} ({meta.get('_path')}) missing category"
        assert meta.get("detection") in valid, (
            f"{slug} ({meta.get('_path')}) bad detection flag: {meta.get('detection')!r}"
        )
