from collections import namedtuple

import pytest
import torch
from sklearn.metrics import max_error as sk_max_error

from pytorch_lightning.metrics.regression import MaxError
from pytorch_lightning.metrics.functional import max_error

from tests.metrics.utils import BATCH_SIZE, NUM_BATCHES, MetricTester

torch.manual_seed(42)

extra_dim = 5

Input = namedtuple('Input', ["preds", "target"])

_single_dim_inputs = Input(preds=torch.rand(NUM_BATCHES, BATCH_SIZE), target=torch.rand(NUM_BATCHES, BATCH_SIZE),)

_multi_dim_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, extra_dim), target=torch.rand(NUM_BATCHES, BATCH_SIZE, extra_dim),
)


def _sk_metric(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return sk_max_error(sk_preds, sk_target)

@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_dim_inputs.preds, _single_dim_inputs.target),
        (_multi_dim_inputs.preds, _multi_dim_inputs.target),
    ],
)
class TestMeanError(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_mean_error_class(self, preds, target, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MaxError,
            sk_metric=_sk_metric,
            dist_sync_on_step=dist_sync_on_step,
        )

    def test_mean_error_functional(self, preds, target):
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=max_error,
            sk_metric=_sk_metric,
        )


def test_error_on_different_shape():
    metric = MaxError()
    with pytest.raises(RuntimeError, match='Predictions and targets are expected to have the same shape'):
        metric(torch.randn(100,), torch.randn(50,))
