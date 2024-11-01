import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from utils import misc
from .mask_encoder import Mask_Encoder, Group, Encoder, TransformerEncoder
from .generator import CPDM, CANet

@MODELS.register_module()
class PointDif(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointDif] ', logger ='PointDif')
        self.config = config
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.trans_dim = config.encoder_config.trans_dim
        self.mask_encoder = Mask_Encoder(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.drop_path_rate = config.encoder_config.drop_path_rate

        self.encoder_dims = config.encoder_config.encoder_dims
        self.cond_dims =  config.generator_config.cond_dims
        self.ca_net = CANet(self.encoder_dims, self.cond_dims)
        self.point_diffusion = CPDM(config)
    
        print_log(f'[PointDif] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointDif')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        trunc_normal_(self.mask_token, std=.02)

    def forward(self, pts, noaug = False, vis = False, **kwargs):
        B,_,_ = pts.shape
        # get patch
        neighborhood, center = self.group_divider(pts)
        # mask and encoder
        encoder_token, mask = self.mask_encoder(neighborhood, center)
        _,N,_ = (center[mask].reshape(B,-1,3)).shape
        # learnable masked token
        mask_token = self.mask_token.expand(B, N, -1)
        encoder_token[mask] = mask_token.reshape(-1, self.trans_dim)
        point_condition = self.ca_net(encoder_token)

        loss = self.point_diffusion.get_loss(pts, point_condition)
        

        if vis: #visualization
            noise_points, recon_points = self.point_diffusion.sample(1024, point_condition)
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - N), -1, 3)
            vis_points = vis_points + center[~mask].unsqueeze(1)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            mask_center = misc.fps(pts, 256)
            vis_points = vis_points.reshape(-1, 3).unsqueeze(0)
            return noise_points, recon_points, vis_points, mask_center
        else:
            return loss


# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.mask_encoder.", ""): v for k, v in ckpt['pointdif'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('pointdif'):
                    base_ckpt[k[len('pointdif.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)

        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        # A.max(1)
        
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret
