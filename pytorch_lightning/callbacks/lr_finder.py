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
import numpy as np

from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback

class LearningRateFinder(Callback):
    def __init__(self, 
        min_lr: float = 1e-8,
        max_lr: float = 1,
        num_training: int = 100,
        mode: str = 'exponential',
        early_stop_threshold: float = 4.0
    ) -> None:
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_training = num_training
        self.mode = mode
        self.early_stop_threshold = early_stop_threshold
        
        self.result = {}
        self.suggested_val = None

    def _construct_new_trainer(self, trainer):
        from pytorch_lightning.trainer import Trainer # prevent circular import
        init_call = trainer._init_call.copy()
        # remove non args
        init_call.pop('self')
        init_call.pop('__class__')
        
        # update with customs
        temp_trainer = Trainer()
        
        return temp_trainer
        
    def lr_find(self, trainer):
        pass
    
    def plot(self, suggest: bool = False, show: bool = False):
        import matplotlib.pyplot as plt

        lrs = self.results["lr"]
        losses = self.results["loss"]

        fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)
        if self.mode == 'exponential':
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")

        if suggest:
            _ = self.suggestion()
            if self._optimal_idx:
                ax.plot(lrs[self._optimal_idx], losses[self._optimal_idx],
                        markersize=10, marker='o', color='red')

        if show:
            plt.show()

        return fig
    
    def suggestion(self, skip_begin: int = 10, skip_end: int = 1):
        try:
            loss = np.array(self.results["loss"][skip_begin:-skip_end])
            loss = loss[np.isfinite(loss)]
            min_grad = np.gradient(loss).argmin()
            self._optimal_idx = min_grad + skip_begin
            return self.results["lr"][self._optimal_idx]
        # todo: specify the possible exception
        except Exception:
            log.exception('Failed to compute suggesting for `lr`. There might not be enough points.')
            self._optimal_idx = None
        
    def on_before_accelerator_backend_setup(self, trainer, pl_module):
        self.lr_find(trainer, pl_module)
        self.suggested_val = self.suggestion()
        
    def on_fit_setup(self, trainer, pl_module):
        pl_module.auto_lr = self.suggested_val
        

class _LRCallback(Callback):
    """ Special callback used by the learning rate finder. This callbacks log
    the learning rate before each batch and log the corresponding loss after
    each batch.

    Args:
        num_training: number of iterations done by the learning rate finder
        early_stop_threshold: threshold for stopping the search. If the
            loss at any point is larger than ``early_stop_threshold*best_loss``
            then the search is stopped. To disable, set to ``None``.
        progress_bar_refresh_rate: rate to refresh the progress bar for
            the learning rate finder
        beta: smoothing value, the loss being logged is a running average of
            loss values logged until now. ``beta`` controls the forget rate i.e.
            if ``beta=0`` all past information is ignored.

    """
    def __init__(self, num_training: int,
                 early_stop_threshold: float = 4.0,
                 progress_bar_refresh_rate: int = 0,
                 beta: float = 0.98):
        self.num_training = num_training
        self.early_stop_threshold = early_stop_threshold
        self.beta = beta
        self.losses = []
        self.lrs = []
        self.avg_loss = 0.0
        self.best_loss = 0.0
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.progress_bar = None

    def on_batch_start(self, trainer, pl_module):
        """ Called before each training batch, logs the lr that will be used """
        if (trainer.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return

        if self.progress_bar_refresh_rate and self.progress_bar is None:
            self.progress_bar = tqdm(desc='Finding best initial lr', total=self.num_training)

        self.lrs.append(trainer.lr_schedulers[0]['scheduler'].lr[0])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """ Called when the training batch ends, logs the calculated loss """
        if (trainer.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return

        if self.progress_bar:
            self.progress_bar.update()

        current_loss = trainer.train_loop.running_loss.last().item()
        current_step = trainer.global_step + 1  # remove the +1 in 1.0

        # Avg loss (loss with momentum) + smoothing
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * current_loss
        smoothed_loss = self.avg_loss / (1 - self.beta**current_step)

        # Check if we diverging
        if self.early_stop_threshold is not None:
            if current_step > 1 and smoothed_loss > self.early_stop_threshold * self.best_loss:
                trainer.max_steps = current_step  # stop signal
                if self.progress_bar:
                    self.progress_bar.close()

        # Save best loss for diverging checking
        if smoothed_loss < self.best_loss or current_step == 1:
            self.best_loss = smoothed_loss

        self.losses.append(smoothed_loss)


class _LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries
    over a number of iterations.
    Arguments:

        optimizer: wrapped optimizer.

        end_lr: the final learning rate.

        num_iter: the number of iterations over which the test occurs.

        last_epoch: the index of last epoch. Default: -1.
    """
    last_epoch: int
    base_lrs: Sequence

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 end_lr: float,
                 num_iter: int,
                 last_epoch: int = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter

        if self.last_epoch > 0:
            val = [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]
        else:
            val = [base_lr for base_lr in self.base_lrs]
        self._lr = val
        return val

    @property
    def lr(self):
        return self._lr


class _ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries
    over a number of iterations.

    Arguments:

        optimizer: wrapped optimizer.

        end_lr: the final learning rate.

        num_iter: the number of iterations over which the test occurs.

        last_epoch: the index of last epoch. Default: -1.
    """
    last_epoch: int
    base_lrs: Sequence

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 end_lr: float,
                 num_iter: int,
                 last_epoch: int = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter

        if self.last_epoch > 0:
            val = [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]
        else:
            val = [base_lr for base_lr in self.base_lrs]
        self._lr = val
        return val

    @property
    def lr(self):
        return self._lr