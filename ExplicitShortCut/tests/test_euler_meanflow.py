import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss.euler_meanflow_loss import EulerMeanFlowLoss
from sampler import cfg_sampler
from utils.scheduler import LinearFlowScheduler


class EndpointZero(torch.nn.Module):
    num_classes = 1000

    def forward(self, z, r, t, y=None):
        return torch.zeros_like(z)


class VelocityToZero(torch.nn.Module):
    num_classes = 1000

    def forward(self, z, r, t, y=None):
        return z / t.view(-1, 1, 1, 1).clamp_min(1e-6)


class TinyModel(torch.nn.Module):
    num_classes = 1000

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, z, r, t, y=None):
        return z * self.bias


def test_velocity_and_endpoint_samplers_agree():
    z = torch.randn(4, 4, 8, 8)
    y = torch.zeros(4, dtype=torch.long)
    scheduler = LinearFlowScheduler()

    for steps in (1, 2, 4):
        velocity_output = cfg_sampler(
            VelocityToZero(),
            z.clone(),
            y=y,
            scheduler=scheduler,
            num_steps=steps,
            prediction_type="velocity",
        )
        endpoint_output = cfg_sampler(
            EndpointZero(),
            z.clone(),
            y=y,
            scheduler=scheduler,
            num_steps=steps,
            prediction_type="endpoint",
        )
        torch.testing.assert_close(velocity_output, endpoint_output)


def test_euler_loss_modes_are_differentiable():
    images = torch.randn(8, 4, 8, 8)
    labels = torch.arange(8)

    for prediction_type, loss_time_weight in (
        ("velocity", "none"),
        ("velocity", "endpoint"),
        ("endpoint", "none"),
    ):
        model = TinyModel()
        target = copy.deepcopy(model)
        loss_fn = EulerMeanFlowLoss(
            prediction_type=prediction_type,
            loss_time_weight=loss_time_weight,
            label_dropout_prob=0.0,
        )
        loss, reference_loss = loss_fn(
            model,
            target,
            images,
            {"y": labels, "global_step": 0, "step_r": 0.0},
        )

        assert loss.shape == (8,)
        assert torch.isfinite(loss).all()
        assert torch.isfinite(reference_loss)

        loss.mean().backward()
        assert torch.isfinite(model.bias.grad)
