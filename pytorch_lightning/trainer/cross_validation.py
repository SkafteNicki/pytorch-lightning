r""" CROSS VALIDATION !!!! """

from abc import ABC, abstractmethod
from typing import Optional

from torch.utils.data import DataLoader, Subset
import os
import numpy as np

from pytorch_lightning import _logger as log
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.data_loading import _has_len


class CrossValidationMixin(ABC):
    @abstractmethod
    def _attach_dataloaders(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""
        
    @abstractmethod
    def save_checkpoint(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def restore(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""
    
    def cross_validation_fit(self, model: LightningModule,
                             K: int = 5,
                             train_dataloader: Optional[DataLoader] = None,
                             val_dataloaders: Optional[DataLoader] = None):
        
        # Required for saving the model, will be reset on fit
        checkpoint_callback = self.checkpoint_callback
        self.checkpoint_callback = False
        self.optimizers, self.schedulers = [], [],
        self.model = model
        
        # Save initial model
        save_path = os.path.join(self.default_root_dir, 'cv_init_model.ckpt')
        self.save_checkpoint(str(save_path))
        
        # Reset
        
        
        # _init_cv_dataloaderAttach dataloaders
        self._attach_dataloaders(model,
                                 train_dataloader=train_dataloader,
                                 val_dataloaders=val_dataloaders)
        
        # Initialize cv dataloaders
        dataloaders_generator = self._init_cv_dataloaders(K)
        
        result = []
        for k, (train_dataloader, test_dataloader) in enumerate(dataloaders_generator):
            log.info(f'Running Cross-Validation Fold {k+1}/{K}')
            self.checkpoint_callback = checkpoint_callback
            self.fit(model,
                     train_dataloader = train_dataloader,
                     val_dataloaders = val_dataloaders)
        
            r = self.test(test_dataloaders = test_dataloader)
            result.append(result)
        
            # Reload initial config
            self.checkpoint_callback = False
            self.restore(str(save_path), on_gpu=self.on_gpu)
            
        # Remove tempoary file
        os.remove(save_path)
        
        return result
        
    def _init_cv_dataloaders(self, K: int):
        
        
        orig_train_dataloader = self.model.train_dataloader()
        
        # Get parameters for the new dataloaders (copy of original setting)
        dataloader_params = {'batch_size': orig_train_dataloader.batch_size,
                             'num_workers': orig_train_dataloader.num_workers,
                             'collate_fn': orig_train_dataloader.collate_fn,
                             'pin_memory': orig_train_dataloader.pin_memory,
                             'drop_last': orig_train_dataloader.drop_last,
                             'timeout': orig_train_dataloader.timeout,
                             'worker_init_fn': orig_train_dataloader.worker_init_fn}
        
        # Check that we can divide into pieces
        _has_len(orig_train_dataloader)
        N = len(orig_train_dataloader)
        
        try:
            from sklearn.model_selection import KFold
        except ImportError as error:
            raise ValueError('install sklearn')
            
        cv = KFold(n_splits=K, shuffle=True)
        dataset = orig_train_dataloader.dataset
        for train_idx, test_idx in cv.split(np.arange(N)):
            subset = Subset(dataset, train_idx)
            train_dataloader = DataLoader(subset, **dataloader_params,
                                          shuffle=True)
                
            subset = Subset(dataset, test_idx)
            test_dataloader = DataLoader(subset, **dataloader_params,
                                         shuffle=False)
            
            yield train_dataloader, test_dataloader
                    
        