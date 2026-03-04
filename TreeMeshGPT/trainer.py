from pathlib import Path
from functools import partial
from packaging import version
from contextlib import nullcontext, contextmanager

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from pytorch_custom_utils import (
    get_adam_optimizer,
    OptimizerWithWarmupSchedule,
    add_wandb_tracker_contextmanager
)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Type, List


#from meshgpt_pytorch.data import custom_collate
#from meshgpt_pytorch.version import __version__


# constants

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cycle(dl):
    while True:
        for data in dl:
            yield data

def maybe_del(d: dict, *keys):
    for key in keys:
        if key not in d:
            continue

        del d[key]


@contextmanager
def trackers(
    trainer,
    project_name: str,
    run_name = None,
    hps = None
):
    assert trainer.use_wandb_tracking

    trainer.accelerator.init_trackers(project_name, config = hps)

    if run_name is not None and len(trainer.accelerator.trackers) > 0:
        trainer.accelerator.trackers[0].run.name = run_name

    yield
    trainer.accelerator.end_training()


@add_wandb_tracker_contextmanager()
class MeshTransformerTrainer(Module):
    @beartype
    def __init__(
        self,
        model,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int,
        grad_accum_every: int,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.,
        max_grad_norm: Optional[float] = 0.5,
        val_dataset: Optional[Dataset] = None,
        val_every = 1,
        val_num_batches = 5,
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        checkpoint_every = 1000,
        checkpoint_folder = './checkpoints',
        data_kwargs: Tuple[str, ...] = ['vertices', 'faces', 'face_edges', 'texts'],
        warmup_steps = 1000,
        use_wandb_tracking = False
    ):
        super().__init__()

        # experiment tracker

        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'

        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        self.accelerator = Accelerator(**accelerator_kwargs)

        self.model = model

        optimizer = get_adam_optimizer(
            model.parameters(),
            lr = learning_rate * warmup_steps,
            wd = weight_decay,
            filter_by_requires_grad = True,
            **optimizer_kwargs
        )

        self.optimizer = OptimizerWithWarmupSchedule(
            accelerator = self.accelerator,
            optimizer = optimizer,
            scheduler = scheduler,
            scheduler_kwargs = scheduler_kwargs,
            warmup_steps = warmup_steps,
            max_grad_norm = max_grad_norm
        )

        pad_id = -1
        self.dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            num_workers = 16,
            shuffle = True,
            drop_last = True,
            #collate_fn = partial(custom_collate, pad_id = pad_id)
        )

        self.should_validate = exists(val_dataset)

        if self.should_validate:
            assert len(val_dataset) > 0, 'your validation dataset is empty'

            self.val_every = val_every
            self.val_num_batches = val_num_batches

            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size = batch_size,
                num_workers = 16,
                shuffle = True,
                drop_last = True,
                #collate_fn = partial(custom_collate, pad_id = pad_id)
            )

        if hasattr(dataset, 'data_kwargs') and exists(dataset.data_kwargs):
            assert is_bearable(dataset.data_kwargs, List[str])
            self.data_kwargs = dataset.data_kwargs
        else:
            self.data_kwargs = data_kwargs

        (
            self.model,
            self.dataloader,
            self.val_dataloader,
            self.optimizer.optimizer, 
            self.optimizer.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.dataloader,
            self.val_dataloader,
            self.optimizer.optimizer, 
            self.optimizer.scheduler,
        )

        self.grad_accum_every = grad_accum_every
        self.num_train_steps = num_train_steps
        self.register_buffer('step', torch.tensor(0))

        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step = self.step.item())

    @property
    def device(self):
        return self.unwrapped_model.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        data = next(dl_iter)

        if isinstance(data, tuple):
            forward_kwargs = dict(zip(self.data_kwargs, data))

        elif isinstance(data, dict):
            forward_kwargs = data

        return forward_kwargs

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.unwrapped_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            step = self.step.item(),
            #version = __version__
        )

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location="cpu")

        #if version.parse(__version__) != version.parse(pkg['version']):
        #    self.print(f'loading saved mesh transformer at version {pkg["version"]}, but current package version is {__version__}')

        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(pkg['model'])
        else:
            self.model.load_state_dict(pkg['model'])
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.step.copy_(pkg['step'])

    def forward(self):
        step = self.step.item()
        dl_iter = cycle(self.dataloader)

        if self.should_validate:
            val_dl_iter = cycle(self.val_dataloader)

        while step < self.num_train_steps:

            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)

                with self.accelerator.autocast(), maybe_no_sync():
                    loss_ce, l2_loss = self.model(**forward_kwargs)
                    
                    loss = loss_ce + 0.001 * l2_loss
                    self.accelerator.backward(loss / self.grad_accum_every)

            self.print(f'step: {step} | loss_ce: {loss_ce.item():.4f} | | loss_l2: {l2_loss.item():.4f} lr: {self.optimizer.optimizer.param_groups[0]["lr"]}')

            self.log(
                loss_ce = loss_ce.item(), 
                loss_l2 = l2_loss.item(), 
                lr = self.optimizer.optimizer.param_groups[0]['lr'],
            )

            self.optimizer.step()
            self.optimizer.zero_grad()

            step += 1
            self.step.add_(1)

            self.wait()

            if self.should_validate and divisible_by(step, self.val_every):

                total_val_loss_ce = 0.
                total_val_loss_l2 = 0.
                self.unwrapped_model.eval()
                """
                # Loop through the entire validation DataLoader
                for forward_kwargs in self.val_dataloader:
                    with self.accelerator.autocast(), torch.no_grad():
                        # Perform forward pass and accumulate loss
                        val_loss = self.unwrapped_model(**forward_kwargs)
                        total_val_loss += val_loss.item()
                
                total_val_loss /= len(self.val_dataloader)
                """
                num_val_batches = self.val_num_batches * self.grad_accum_every
                
                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():

                        forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)

                        val_loss_ce, val_loss_l2 = self.unwrapped_model(**forward_kwargs)

                        #val_loss = val_loss_ce + val_loss_l2 * 0.001
                        total_val_loss_ce += (val_loss_ce / num_val_batches)
                        total_val_loss_l2 += (val_loss_l2 / num_val_batches)
            

                self.print(f'valid loss CE: {total_val_loss_ce:.4f} | valid loss L2: {total_val_loss_l2:.4f} ')

                self.log(val_loss_ce = total_val_loss_ce,
                         val_loss_l2 = total_val_loss_l2)
                
                self.unwrapped_model.train()

            self.wait()

            if self.is_main and divisible_by(step, self.checkpoint_every):
                checkpoint_num = step // self.checkpoint_every
                self.save(self.checkpoint_folder / f'model.ckpt.{checkpoint_num}.pt')
                
            if self.is_main and divisible_by(step, 200):
                self.save(self.checkpoint_folder / f'model.ckpt.last.pt')

            self.wait()

        self.print('training complete')
