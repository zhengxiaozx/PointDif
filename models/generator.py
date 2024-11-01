import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from utils import misc
from utils.logger import *
from SoftPool import soft_pool2d, SoftPool2d

class VarianceSchedule(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.num_steps = self.config.generator_config.time_schedule.num_steps
        self.beta_start = self.config.generator_config.time_schedule.beta_start
        self.beta_end = self.config.generator_config.time_schedule.beta_end
        self.mode = self.config.generator_config.time_schedule.mode
        
        if self.mode == 'linear':
            betas = torch.linspace(self.beta_start, self.beta_end, steps=self.num_steps)
            
        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding
        alphas = 1 - betas
        
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    # original sampling strategy
    # def uniform_sampling(self, batch_size):
    #     ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
    #     return ts.tolist()

    # Recurrent Uniform Sampling Strategy
    def recurrent_uniform_sampling(self, batch_size, interval_nums):
        interval_size = self.num_steps / interval_nums
        sampled_intervals = []
        for i in range(interval_nums):
            start = int(i * interval_size) + 1
            end = int((i + 1) * interval_size)
            sampled_interval = np.random.choice(np.arange(start, end + 1), batch_size)
            sampled_intervals.append(sampled_interval)
        ts = np.vstack(sampled_intervals)
        ts = torch.tensor(ts)
        ts = torch.stack([ts[:, i][torch.randperm(interval_nums)] for i in range(batch_size)], dim=1)
        return ts


# Condition Aggregation Network
class CANet(nn.Module): 
    def __init__(self, encoder_dims, cond_dims):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.cond_dims = cond_dims

        self.mlp1 = nn.Sequential(
            nn.Conv2d(self.encoder_dims, 512, kernel_size=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=1, bias=True),
            nn.ReLU(True),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, self.cond_dims, kernel_size=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, patch_fea):
        '''
            patch_feature : B G 384
            -----------------
            point_condition : B 384
        '''
        
        patch_fea = patch_fea.transpose(1, 2)     # B 384 G
        patch_fea = patch_fea.unsqueeze(-1)       # B 384 G 1
        patch_fea = self.mlp1(patch_fea)          # B 512 G 1
        # soft_pool2d
        global_fea = soft_pool2d(patch_fea, kernel_size=[patch_fea.size(2), 1])  # B 512 1 1
        global_fea = global_fea.expand(-1, -1, patch_fea.size(2), -1)            # B 512 G 1
        combined_fea = torch.cat([patch_fea, global_fea], dim=1)                 # B 1024 G 1
        combined_fea = self.mlp2(combined_fea)                                       # B F G 1
        condition_fea = soft_pool2d(combined_fea, kernel_size=[combined_fea.size(2), 1])  # B F 1 1
        condition_fea = condition_fea.squeeze(-1).squeeze(-1)                          #  B F
        return condition_fea

# Point Condition Network 
class PCNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_cond):
        super(PCNet, self).__init__()
        self.fea_layer = nn.Linear(dim_in, dim_out)
        self.cond_bias = nn.Linear(dim_cond, dim_out, bias=False)
        self.cond_gate = nn.Linear(dim_cond, dim_out)

    def forward(self, fea, cond):
        gate = torch.sigmoid(self.cond_gate(cond))
        bias = self.cond_bias(cond)
        out = self.fea_layer(fea) * gate + bias
        return out

# Point Denoising Network
class DenoisingNet(nn.Module):

    def __init__(self, point_dim, cond_dims, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = nn.ModuleList([
            PCNet(3, 128, cond_dims+3),
            PCNet(128, 256, cond_dims+3),
            PCNet(256, 512, cond_dims+3),
            PCNet(512, 256, cond_dims+3),
            PCNet(256, 128, cond_dims+3),
            PCNet(128, 3, cond_dims+3)
        ])

    def forward(self, coords, beta, cond):
        """
        Args:
            coords:   Noise point clouds at timestep t, (B, N, 3).
            beta:     Time. (B, ).
            cond:     Condition. (B, F).
        """

        batch_size = coords.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        cond = cond.view(batch_size, 1, -1)         # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        cond_emb = torch.cat([time_emb, cond], dim=-1)    # (B, 1, F+3)
        
        out = coords
        for i, layer in enumerate(self.layers):
            out = layer(fea=out, cond=cond_emb)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return coords + out
        else:
            return out


# Conditional Point Diffusion Model
class CPDM(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cond_dims = self.config.generator_config.cond_dims 
        self.net = DenoisingNet(point_dim=3, cond_dims=self.cond_dims, residual=True)
        self.var_sched = VarianceSchedule(config)
        self.interval_nums = self.config.generator_config.interval_nums
    
    def get_loss(self, coords, cond, ts=None):
        """
        Args:
            coords:   point cloud, (B, N, 3).
            cond:     condition (B, F).
        """

        batch_size, _, point_dim = coords.size()

        if ts == None:
            ts = self.var_sched.recurrent_uniform_sampling(batch_size, self.interval_nums)

        total_loss = 0

        for i in range(self.interval_nums):
            t = ts[i].tolist()
            
            alphas_cumprod = self.var_sched.alphas_cumprod[t]
            beta = self.var_sched.betas[t]
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod).view(-1, 1, 1)       # (B, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod).view(-1, 1, 1)   # (B, 1, 1)
            
            noise = torch.randn_like(coords)  # (B, N, d)
            pred_noise = self.net(sqrt_alphas_cumprod_t * coords + sqrt_one_minus_alphas_cumprod_t * noise, beta=beta, cond=cond)
            loss = F.mse_loss(noise.view(-1, point_dim), pred_noise.view(-1, point_dim), reduction='mean')
            total_loss += (loss * (1.0 / self.interval_nums))

        return total_loss