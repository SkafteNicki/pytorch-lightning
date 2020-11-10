# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from pytorch_lightning.metrics.utils import _check_same_shape


def _max_error_update(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _check_same_shape(preds, target)
    max_error = torch.max(torch.abs(preds - target))
    return max_error


def _max_error_compute(max_error: torch.Tensor) -> torch.Tensor:
    return max_error


def max_error(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes maximum absolute error

    Args:
        pred: estimated labels
        target: ground truth labels

    Return:
        Tensor with max error

    Example:

        >>> pred = torch.tensor([0., 1., 2., 3.])
        >>> target = torch.tensor([1., 3., 5., 7.])
        >>> max_error(pred, target)
        tensor(4.)

    """
    max_error = _max_error_update(preds, target)
    return _max_error_compute(max_error)
