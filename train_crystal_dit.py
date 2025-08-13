import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from tqdm import tqdm
import json
import random

from crystal_representation import CrystalDataset, preprocess_dataset
from crystal_dit import CrystalDiT
from crystal_diffusion import CrystalGaussianDiffusion

# 添加种子工作器初始化函数
def seed_worker(worker_id):
    """Set seed for DataLoader workers"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed, rank=0):
    """Set all random seeds for reproducibility"""
    actual_seed = seed + rank
    
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(actual_seed)
        torch.cuda.manual_seed_all(actual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(actual_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def setup_logging(rank, args):
    """Setup logging configuration"""
    log_file = os.path.join(args.output_dir, 'train.log')
    
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    return logging.getLogger()

def setup(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29527'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training environment"""
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, args, rank, is_best=False, is_periodic=False):
    """Save model checkpoint"""
    if rank != 0:
        return
    
    checkpoint = {
        'model': model.module.state_dict(),
    }
    
    if not is_periodic:
        checkpoint.update({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'global_step': global_step,
            'args': args,
            'random_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        })
    
    if not is_periodic:
        latest_path = os.path.join(args.output_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        logging.info(f"Saved latest checkpoint: {latest_path}")
    
    if is_periodic:
        epoch_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, epoch_path)
        logging.info(f"Saved periodic checkpoint: {epoch_path} (weights only)")
    
    if is_best:
        best_path = os.path.join(args.output_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best model: {best_path}")

def load_checkpoint(model, optimizer, scheduler, args, rank):
    """Load model checkpoint"""
    if not os.path.exists(args.resume):
        if rank == 0:
            logging.info(f"Checkpoint not found: {args.resume}, starting from scratch.")
        return 0, 0
    
    if rank == 0:
        logging.info(f"Loading checkpoint: {args.resume}")
    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(args.resume, map_location=map_location)
    
    model.module.load_state_dict(checkpoint['model'])
    
    if 'optimizer' in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if rank == 0:
                logging.info("Successfully loaded optimizer state")
        except:
            if rank == 0:
                logging.info("Optimizer state not found or incompatible, using new optimizer")
    
    if 'scheduler' in checkpoint and scheduler is not None and checkpoint.get('scheduler') is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
            if rank == 0:
                logging.info("Successfully loaded scheduler state")
        except:
            if rank == 0:
                logging.info("Scheduler state not found or incompatible, using new scheduler")
    
    if 'random_state' in checkpoint:
        random_state = checkpoint['random_state']
        if random_state.get('python') is not None:
            random.setstate(random_state['python'])
        if random_state.get('numpy') is not None:
            np.random.set_state(random_state['numpy'])
        if random_state.get('torch') is not None:
            torch.set_rng_state(random_state['torch'])
        if random_state.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(random_state['cuda'])
        if rank == 0:
            logging.info("Successfully restored random state")
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    
    if rank == 0:
        if epoch > 0:
            logging.info(f"Resuming training from epoch {epoch} and step {global_step}")
        else:
            logging.info(f"Loaded model weights, but starting from epoch 0 (simplified checkpoint)")
    
    return epoch, global_step

def train_epoch(model, diffusion, dataloader, optimizer, scheduler, device, epoch, global_step, args, rank):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    epoch_start_time = time.time()
    
    iterator = dataloader
    if rank == 0:
        iterator = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(iterator):
        lattice_vectors = batch["lattice_vectors"].to(device)
        atom_features = batch["atom_features"].to(device)
        
        optimizer.zero_grad()
        loss_dict = diffusion.training_losses(model, lattice_vectors, atom_features)
        loss = loss_dict["loss"]
        
        loss.backward()
        
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        global_step += 1
        
        if rank == 0 and global_step % args.log_every == 0:
            if isinstance(iterator, tqdm):
                iterator.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
            
            logging.info(
                f"Epoch {epoch} | Step {global_step} | "
                f"Loss: {loss.item():.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
    
    avg_loss = total_loss / len(dataloader)
    
    if rank == 0:
        epoch_time = time.time() - epoch_start_time
        logging.info(
            f"Epoch {epoch} completed | Time: {epoch_time:.2f}s | "
            f"Average loss: {avg_loss:.6f}"
        )
    
    return global_step, avg_loss

def train(rank, world_size, args):
    """Main training function"""
    setup(rank, world_size)
    set_seed(args.seed, rank)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    logger = setup_logging(rank, args)
    
    if rank == 0:
        logging.info(f"Training parameters: {args}")
        logging.info(f"World size: {world_size}, Device: {device}")
        logging.info(f"Random seed: {args.seed}")
        logging.info(f"Model config: hidden_size={args.hidden_size}, num_heads={args.num_heads}, depth={args.depth}")
    
    logging.info(f"Process {rank}: Preparing to load dataset...")
    
    if args.preprocessed_dir is None:
        preprocessed_dir = os.path.join(args.data_dir, "preprocessed")
    else:
        preprocessed_dir = args.preprocessed_dir

    if rank == 0:
        os.makedirs(preprocessed_dir, exist_ok=True)
        train_preprocessed = os.path.join(preprocessed_dir, f"train_preprocessed_ma{args.max_atoms}.pkl")
        
        if args.force_preprocess or not os.path.exists(train_preprocessed):
            preprocess_dataset(
                os.path.join(args.data_dir, "train.csv"), 
                train_preprocessed, 
                args.max_atoms
            )
    
    dist.barrier()
    
    train_preprocessed = os.path.join(preprocessed_dir, f"train_preprocessed_ma{args.max_atoms}.pkl")
    logging.info(f"Process {rank}: Loading dataset...")
    
    try:
        train_dataset = CrystalDataset(
            os.path.join(args.data_dir, "train.csv"), 
            max_atoms=args.max_atoms,
            preprocessed_file=train_preprocessed,
            force_preprocess=False
        )
    except Exception as e:
        logging.error(f"Process {rank}: Failed to load dataset: {e}")
        cleanup()
        raise e
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed
    )
    
    g = torch.Generator()
    g.manual_seed(args.seed + rank)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    logging.info(f"Process {rank}: Creating model (hidden_size={args.hidden_size}, num_heads={args.num_heads}, depth={args.depth})...")
    
    model = CrystalDiT(
        max_atoms=args.max_atoms,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0
    )
    
    if rank == 0:
        from crystal_dit import count_parameters
        total_params = count_parameters(model)
        logging.info(f"Total model parameters: {total_params:,}")
    
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    diffusion = CrystalGaussianDiffusion(
        timesteps=args.diffusion_steps,
        beta_schedule=args.beta_schedule
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader),
        eta_min=args.min_learning_rate
    )
    
    start_epoch, global_step = 0, 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, args, rank)
    
    best_train_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        
        global_step, train_loss = train_epoch(
            model, diffusion, train_loader, optimizer, scheduler,
            device, epoch, global_step, args, rank
        )
        
        is_best = train_loss < best_train_loss
        if is_best:
            best_train_loss = train_loss
            if rank == 0:
                logging.info(f"Found new best model, training loss: {train_loss:.6f}")
        
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, global_step,
                args, rank, is_best=False, is_periodic=True
            )
        
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1, global_step,
            args, rank, is_best=is_best, is_periodic=False
        )
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='Train Crystal DiT model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./datasets/mp_20', help='Dataset directory')
    parser.add_argument('--max_atoms', type=int, default=20, help='Maximum number of atoms')
    parser.add_argument('--preprocessed_dir', type=str, default=None, help='Preprocessed data directory')
    parser.add_argument('--force_preprocess', action='store_true', help='Force reprocessing even if preprocessed files exist')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--depth', type=int, default=18, help='Number of DiT layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    
    # Diffusion parameters
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--beta_schedule', type=str, default='linear', help='Beta schedule type')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping threshold')
    parser.add_argument('--epochs', type=int, default=50000, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers per GPU')
    
    # Checkpoint parameters
    parser.add_argument('--output_dir', type=str, default='./output/crystal_dit', help='Output directory')
    parser.add_argument('--resume', type=str, default='', help='Resume training from checkpoint')
    parser.add_argument('--save_every', type=int, default=250, help='Save checkpoint every N epochs')
    
    # Logging parameters
    parser.add_argument('--log_every', type=int, default=10, help='Log every N steps')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    assert world_size >= 1, "At least one GPU is required for training"
    
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
