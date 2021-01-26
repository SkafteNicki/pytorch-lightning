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
# limitations under the License
from typing import Optional, Tuple

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.data import has_len
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_hasattr, lightning_setattr
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error


class BatchSizeScaler(Callback):
    def __init__(self,  
                 mode: str = 'power',
                 steps_per_trial: int = 3,
                 init_val: int = 2,
                 max_trials: int = 25,
                 batch_arg_name: str = 'batch_size',
    ) -> None:
        self.mode = mode
        self.steps_per_trial = steps_per_trial
        self.init_val = init_val
        self.max_trials = max_trials
        self.batch_arg_name = batch_arg_name
    
    def _construct_new_trainer(self, trainer):
        from pytorch_lightning.trainer import Trainer # prevent circular import
        init_call = trainer._init_call.copy()
        # remove non args
        init_call.pop('self')
        init_call.pop('__class__')
        
        # update with customs
        init_call['logger'] = DummyLogger() # disable logging
        init_call['callbacks'] = [] # disable all callbacks
        init_call['checkpoint_callback'] = False # disable checkpoints
        init_call['progress_bar_refresh_rate'] = 0 # disable progress bar
        init_call['weights_summary'] = None # disable weights summary
        init_call['max_steps'] =self.steps_per_trial
        init_call['limit_train_batches'] = 1.0
                
        temp_trainer = Trainer(**init_call)
        return temp_trainer
    
    def scale_batch_size(self, trainer, model, train_dataloader=None, val_dataloaders=None, datamodule=None):
        if trainer.fast_dev_run:
            rank_zero_warn('Skipping batch size scaler since fast_dev_run is enabled.', UserWarning)
            return
        
        if not lightning_hasattr(model, self.batch_arg_name):
            raise MisconfigurationException(
                f'Field {self.batch_arg_name} not found in both `model`,`model.hparams` or `datamodule`')
        
        if hasattr(model, self.batch_arg_name) and hasattr(model, "hparams") and self.batch_arg_name in model.hparams:
            rank_zero_warn(
                f'Field `model.{self.batch_arg_name}` and `model.hparams.{self.batch_arg_name}` are mutually exclusive!'
                f' `model.{self.batch_arg_name}` will be used as the initial batch size for scaling.'
                f' If this is not the intended behavior, please remove either one.'
            )
        
        if hasattr(model.train_dataloader, 'patch_loader_code'):
            raise MisconfigurationException('The batch scaling feature cannot be used with dataloaders'
                                            ' passed directly to `.fit()`. Please disable the feature or'
                                            ' incorporate the dataloader into the model.')
        
        
        temp_trainer = self._construct_new_trainer(trainer)
        temp_trainer.train_loop.setup_fit(model, train_dataloader=train_dataloader,
                                          val_dataloaders=val_dataloaders,
                                          datamodule=datamodule)
        
        # Save model for loading afterwards
        temp_trainer.save_checkpoint('scale_batch_size_temp_model.ckpt', weights_only=True)
        
        # Scale batch size
        new_size = _adjust_batch_size(trainer, self.batch_arg_name, value=self.init_val)  # initially set to init_val
        if self.mode == 'power':
            new_size = _run_power_scaling(trainer, model, new_size, self.batch_arg_name, self.max_trials, **fit_kwargs)
        elif self.mode == 'binsearch':
            new_size = _run_binsearch_scaling(trainer, model, new_size, self.batch_arg_name, self.max_trials, **fit_kwargs)
        else:
            raise ValueError('mode in method `scale_batch_size` can only be `power` or `binsearch')
    
        garbage_collection_cuda()
        log.info(f'Finished batch size finder, will continue with full run using batch size {new_size}')
        
        # Load initial saved model
        temp_trainer.load_from_checkpoint(temp_trainer.default_root_dir + 'scale_batch_size_temp_model.ckpt')
        
    def on_before_accelerator_backend_setup(self, trainer, pl_module):
        results = self.scale_batch_size(trainer, pl_module, 
                                        train_dataloader=pl_module.train_dataloader,
                                        val_dataloaders=pl_module.val_dataloader,
                                        datamodule=trainer.datamodule)
        self.suggested_val = results.suggestion()
        
    def on_fit_setup(self, trainer, pl_module):
        pl_module.auto_batch_scale = self.suggested_val
        
        
def _run_power_scaling(trainer, model, new_size, batch_arg_name, max_trials, **fit_kwargs):
    """ Batch scaling mode where the size is doubled at each iteration until an
        OOM error is encountered. """
    for _ in range(max_trials):
        garbage_collection_cuda()
        trainer.global_step = 0  # reset after each try
        try:
            # Try fit
            trainer.fit(model, **fit_kwargs)
            # Double in size
            new_size, changed = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc='succeeded')
        except RuntimeError as exception:
            # Only these errors should trigger an adjustment
            if is_oom_error(exception):
                # If we fail in power mode, half the size and return
                garbage_collection_cuda()
                new_size, _ = _adjust_batch_size(trainer, batch_arg_name, factor=0.5, desc='failed')
                break
            else:
                raise  # some other error not memory related

        if not changed:
            break
    return new_size


def _run_binsearch_scaling(trainer, model, new_size, batch_arg_name, max_trials, **fit_kwargs):
    """ Batch scaling mode where the size is initially is doubled at each iteration
        until an OOM error is encountered. Hereafter, the batch size is further
        refined using a binary search """
    high = None
    count = 0
    while True:
        garbage_collection_cuda()
        trainer.global_step = 0  # reset after each try
        try:
            # Try fit
            trainer.fit(model, **fit_kwargs)
            count += 1
            if count > max_trials:
                break
            # Double in size
            low = new_size
            if high:
                if high - low <= 1:
                    break
                midval = (high + low) // 2
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc='succeeded')
            else:
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc='succeeded')

            if not changed:
                break

        except RuntimeError as exception:
            # Only these errors should trigger an adjustment
            if is_oom_error(exception):
                # If we fail in power mode, half the size and return
                garbage_collection_cuda()
                high = new_size
                midval = (high + low) // 2
                new_size, _ = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc='failed')
                if high - low <= 1:
                    break
            else:
                raise  # some other error not memory related

    return new_size


def _adjust_batch_size(trainer,
                       batch_arg_name: str = 'batch_size',
                       factor: float = 1.0,
                       value: Optional[int] = None,
                       desc: Optional[str] = None) -> Tuple[int, bool]:
    """ Helper function for adjusting the batch size.

    Args:
        trainer: instance of pytorch_lightning.Trainer

        batch_arg_name: name of the field where batch_size is stored.

        factor: value which the old batch size is multiplied by to get the
            new batch size

        value: if a value is given, will override the batch size with this value.
            Note that the value of `factor` will not have an effect in this case

        desc: either `succeeded` or `failed`. Used purely for logging

    Returns:
        The new batch size for the next trial and a bool that signals whether the
        new value is different than the previous batch size.
    """
    model = trainer.get_model()
    batch_size = lightning_getattr(model, batch_arg_name)
    new_size = value if value is not None else int(batch_size * factor)
    if desc:
        log.info(f'Batch size {batch_size} {desc}, trying batch size {new_size}')

    if not _is_valid_batch_size(new_size, trainer.train_dataloader):
        new_size = min(new_size, len(trainer.train_dataloader.dataset))

    changed = new_size != batch_size
    lightning_setattr(model, batch_arg_name, new_size)
    return new_size, changed


def _is_valid_batch_size(current_size, dataloader):
    return not has_len(dataloader) or current_size <= len(dataloader)

        
        