from trainer import MeshTransformerTrainer, trackers
from model.treemeshgpt_train import TreeMeshGPT
from train_dataloader import EdgesDataset
from accelerate.utils import DistributedDataParallelKwargs
import os
import yaml
import torch

torch._dynamo.config.optimize_ddp = False

with open("configs/tmgpt.yaml","r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

TRAIN_PATH = "dummy_data/pkl"
VAL_PATH = "dummy_data/pkl"

quant_bit = config["quant_bit"]

train_dataset = EdgesDataset(TRAIN_PATH, quant_bit=quant_bit)
val_dataset = EdgesDataset(VAL_PATH, quant_bit=quant_bit)


transformer = TreeMeshGPT(
    dim = config['dim'],
    attn_depth = config['depth'],
    dropout = config['dropout'],
    quant_bit = quant_bit
)

trainer = MeshTransformerTrainer(
    model = transformer,
    dataset = train_dataset,
    val_dataset = val_dataset,
    val_every = config['val_every'],
    val_num_batches = 1,
    num_train_steps = int(config['num_train_steps']),
    batch_size = config['batch_size'],
    learning_rate = config['learning_rate'],
    grad_accum_every = config['grad_accum_every'],
    warmup_steps = config['warmup_steps'], 
    weight_decay = config['weight_decay'],
    use_wandb_tracking = True,
    checkpoint_every = config['checkpoint_every'],
    checkpoint_folder = f'./checkpoints/{config["exp_name"]}',
    ema_kwargs = {
        "allow_different_devices": True,
    },
    accelerator_kwargs = {
        'kwargs_handlers': [
            DistributedDataParallelKwargs(find_unused_parameters=False)
        ]
    }
)

# uncomment to continue training from checkpoint
#ckpt = "checkpoints/tmgpt/model.ckpt.last.pt"
#trainer.load(ckpt)

#trainer() # Train without wandb logging

with trackers(trainer, project_name='TreeMeshGPT', run_name=config["exp_name"], hps=config):
    trainer()
