import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
import math

from diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType, get_named_beta_schedule
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.timestep_sampler import create_named_schedule_sampler

class CrystalGaussianDiffusion:
    """Gaussian diffusion process wrapper for crystal structures"""
    
    def __init__(
        self,
        *,
        timesteps=1000,
        beta_schedule="linear",
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    ):
        self.timesteps = timesteps
        
        betas = get_named_beta_schedule(beta_schedule, timesteps)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=loss_type,
        )
        
        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)
    
    def q_sample(
        self, 
        lattice_vectors: torch.Tensor, 
        atom_features: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to crystal structure data (forward diffusion)"""
        if noise is None:
            lattice_noise = torch.randn_like(lattice_vectors)
            atom_noise = torch.randn_like(atom_features)
        else:
            lattice_noise, atom_noise = noise
        
        noisy_lattice = self.diffusion.q_sample(lattice_vectors, t, noise=lattice_noise)
        noisy_atoms = self.diffusion.q_sample(atom_features, t, noise=atom_noise)
        
        return noisy_lattice, noisy_atoms


    def training_losses(
        self, 
        model: nn.Module, 
        lattice_vectors: torch.Tensor, 
        atom_features: torch.Tensor, 
        t: Optional[torch.Tensor] = None, 
        noise: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses"""
        batch_size = lattice_vectors.shape[0]
        
        if t is None:
            t, weights = self.schedule_sampler.sample(batch_size, device=lattice_vectors.device)
        else:
            weights = torch.ones_like(t, dtype=torch.float)
        
        if noise is None:
            lattice_noise = torch.randn_like(lattice_vectors)
            atom_noise = torch.randn_like(atom_features)
        else:
            lattice_noise, atom_noise = noise
        
        noisy_lattice, noisy_atoms = self.q_sample(
            lattice_vectors, atom_features, t, noise=(lattice_noise, atom_noise)
        )
        
        pred_lattice_noise, pred_atom_noise = model(noisy_lattice, noisy_atoms, t)
        
        # Compute losses
        lattice_loss = torch.mean((lattice_noise - pred_lattice_noise) ** 2, dim=(1, 2))
        
        # Weighted atom feature loss (higher weight for period/group features)
        feature_weights = torch.tensor([1.5, 2.0, 1.0, 1.0, 1.0], device=atom_noise.device)
        weighted_sq_diff = (atom_noise - pred_atom_noise) ** 2
        weighted_sq_diff = weighted_sq_diff * feature_weights.view(1, 1, -1)
        atom_loss = torch.mean(weighted_sq_diff, dim=(1, 2))
        
        # Total loss = lattice_loss + 100 * atom_loss
        total_loss = lattice_loss + 100 * atom_loss
        weighted_loss = total_loss * weights
        
        return {
            "loss": weighted_loss.mean(),
            "lattice_loss": lattice_loss.mean(),
            "atom_loss": atom_loss.mean(),
            "weighted_loss": weighted_loss.mean(),
        }
    
    def p_sample_loop(
        self,
        model: nn.Module,
        batch_size: int,
        max_atoms: int,
        clip_denoised: bool = True,
        progress: bool = False,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate crystal structures from noise (sampling process)"""
        if device is None:
            device = next(model.parameters()).device
        
        lattice_shape = (batch_size, 3, 3)
        atom_shape = (batch_size, max_atoms, 5)
        
        lattice_noise = torch.randn(lattice_shape, device=device)
        atom_noise = torch.randn(atom_shape, device=device)
        
        indices = list(range(self.diffusion.num_timesteps))[::-1]
        
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        
        lattice_vectors = lattice_noise
        atom_features = atom_noise
        
        for i in indices:
            t = torch.tensor([i] * batch_size, device=device)
            with torch.no_grad():
                pred_lattice_noise, pred_atom_noise = model(lattice_vectors, atom_features, t)
                
                pred_lattice_x0 = self._predict_x0_from_eps(lattice_vectors, t, pred_lattice_noise)
                pred_atom_x0 = self._predict_x0_from_eps(atom_features, t, pred_atom_noise)
                
                if clip_denoised:
                    pred_lattice_x0 = torch.clamp(pred_lattice_x0, -1, 1)
                    pred_atom_x0 = torch.clamp(pred_atom_x0, -1, 1)
                
                posterior_mean_lattice = self._q_posterior_mean(
                    pred_lattice_x0, lattice_vectors, t
                )
                posterior_mean_atom = self._q_posterior_mean(
                    pred_atom_x0, atom_features, t
                )
                
                lattice_posterior_log_variance = self._extract_into_tensor(
                    self.diffusion.posterior_log_variance_clipped, t, lattice_vectors.shape
                )
                
                atom_posterior_log_variance = self._extract_into_tensor(
                    self.diffusion.posterior_log_variance_clipped, t, atom_features.shape
                )
                
                if i == 0:
                    lattice_vectors = posterior_mean_lattice
                    atom_features = posterior_mean_atom
                else:
                    lattice_noise = torch.randn_like(lattice_vectors)
                    lattice_vectors = posterior_mean_lattice + torch.exp(0.5 * lattice_posterior_log_variance) * lattice_noise
                    
                    atom_noise = torch.randn_like(atom_features)
                    atom_features = posterior_mean_atom + torch.exp(0.5 * atom_posterior_log_variance) * atom_noise
                                
        return lattice_vectors, atom_features
    
    def _predict_x0_from_eps(self, x_t, t, eps):
        """从噪声预测原始输入"""
        return (
            self._extract_into_tensor(self.diffusion.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract_into_tensor(self.diffusion.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _q_posterior_mean(self, x_0, x_t, t):
        """计算后验均值 q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            self._extract_into_tensor(self.diffusion.posterior_mean_coef1, t, x_t.shape) * x_0 +
            self._extract_into_tensor(self.diffusion.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return posterior_mean
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """从1-D numpy数组中提取值"""
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
