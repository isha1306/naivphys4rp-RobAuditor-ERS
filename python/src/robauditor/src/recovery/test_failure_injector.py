import pytest

from .failure_injector import (
    FAILURE_STAGES,
    FailureInjector,
    InjectedFailure,
)


def test_disabled_injector_does_not_fail():
    injector = FailureInjector()

    for stage in FAILURE_STAGES:
        assert injector.check(stage) is False

    assert injector.triggered is False


def test_injects_failure_at_configured_stage():
    injector = FailureInjector(
        enabled=True,
        stage="grasp",
    )

    assert injector.check("pre_grasp") is False

    with pytest.raises(InjectedFailure) as exc_info:
        injector.check("grasp")

    assert exc_info.value.stage == "grasp"
    assert exc_info.value.failure_id == "injected_grasp"
    assert injector.triggered is True
    assert injector.triggered_stage == "grasp"


def test_failure_is_injected_only_once():
    injector = FailureInjector(
        enabled=True,
        stage="transfer",
    )

    with pytest.raises(InjectedFailure):
        injector.check("transfer")

    assert injector.check("transfer") is False
    assert injector.triggered is True


def test_invalid_stage_is_rejected():
    with pytest.raises(ValueError):
        FailureInjector(
            enabled=True,
            stage="invalid_stage",
        )


def test_enabled_injector_requires_stage():
    with pytest.raises(ValueError):
        FailureInjector(enabled=True)


def test_reset_allows_another_injection():
    injector = FailureInjector(
        enabled=True,
        stage="lift",
    )

    with pytest.raises(InjectedFailure):
        injector.check("lift")

    injector.reset()

    assert injector.triggered is False
    assert injector.triggered_stage is None

    with pytest.raises(InjectedFailure):
        injector.check("lift")
