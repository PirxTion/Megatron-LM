import pytest
import torch

from megatron.core.optimizer import OptimizerConfig, _get_megatron_optimizer_based_on_param_groups
from megatron.core.optimizer.ademamix import AdEMAMix


def _run_ademamix_step(
    *,
    param_value: float,
    grad_value: float,
    lr: float = 0.1,
    weight_decay: float = 0.5,
    cautious: bool,
) -> float:
    """Utility that runs a single AdEMAMix step and returns the updated parameter value."""
    param = torch.nn.Parameter(torch.tensor([param_value], dtype=torch.float32))
    optimizer = AdEMAMix(
        [param],
        lr=lr,
        betas=(0.0, 0.0, 0.0),
        alpha=0.0,
        eps=0.0,
        weight_decay=weight_decay,
        cautious_weight_decay=cautious,
    )
    param.grad = torch.tensor([grad_value], dtype=torch.float32)
    optimizer.step()
    return param.item()


def test_ademamix_cautious_weight_decay_applies_when_update_shrinks_parameter():
    """When update direction matches the parameter sign, decay should be applied."""
    updated_value = _run_ademamix_step(param_value=1.0, grad_value=1.0, cautious=True)
    assert updated_value == pytest.approx(0.85, abs=1e-6)


def test_ademamix_cautious_weight_decay_skips_when_update_increases_magnitude():
    """When update grows the parameter magnitude, decay should be disabled."""
    cautious_value = _run_ademamix_step(param_value=1.0, grad_value=-1.0, cautious=True)
    baseline_value = _run_ademamix_step(param_value=1.0, grad_value=-1.0, cautious=False)

    assert cautious_value == pytest.approx(1.10, abs=1e-6)
    assert baseline_value == pytest.approx(1.05, abs=1e-6)
    assert cautious_value > baseline_value


def test_cautious_weight_decay_restricted_to_ademamix():
    """Using cautious weight decay with non-AdEMAMix optimizers should raise immediately."""
    config = OptimizerConfig(
        optimizer='adam',
        lr=0.01,
        weight_decay=0.1,
        use_cautious_weight_decay=True,
    )
    with pytest.raises(ValueError, match="only supported for AdEMAMix"):
        _get_megatron_optimizer_based_on_param_groups(config, model_chunks=[], param_groups=[])
