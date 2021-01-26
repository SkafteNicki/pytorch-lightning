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
        
    def on_before_accelerator_backend_setup(self, trainer, pl_module):
        results = self.lr_find(trainer, pl_module)
        self.suggested_val = results.suggestion()
        
    def on_fit_setup(self, trainer, pl_module):
        pl_module.auto_lr = self.suggested_val