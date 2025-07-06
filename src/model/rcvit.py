"""
Code for CAS-ViT
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

import numpy as np
from einops import rearrange, repeat
import itertools
import os
import copy

from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from timm.models.registry import register_model

#EVANDRO: LOSS CALCULATION
import torch.nn.functional as F 

# ======================================================================================================================
def stem(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )

class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class LocalIntegration(nn.Module):
    """
    """
    def __init__(self, dim, ratio=1, act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        mid_dim = round(ratio * dim)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            norm_layer(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.network(x)


class AdditiveTokenMixer(nn.Module):
    """
    改变了proj函数的输入，不对q+k卷积，而是对融合之后的结果proj
    """
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out


NUM_DAP_TOKENS = 10
TASK_EMB = 256 #Use TASK_EMB = 64 or 128 (same as DAP default). ??? Antes estava eu coloquei 16
class AdditiveBlock(nn.Module):
    """
    Additive block with support for Domain-Adaptive Prompts (DAP).
    """
    def __init__(self, dim, mlp_ratio=4., attn_bias=False, drop=0., drop_path=0.,
                 act_layer=nn.ReLU, norm_layer=nn.GELU, dap_config=None):
        super().__init__()

        self.local_perception = LocalIntegration(dim, ratio=1, act_layer=act_layer, norm_layer=norm_layer)
        self.norm1 = norm_layer(dim)
        self.attn = AdditiveTokenMixer(dim, attn_bias=attn_bias, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=self.mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # DAP components       
        # self.dap_config = dap_config #OK 
        # self.dap_downsample = nn.Linear(dim, NUM_DAP_TOKENS) #NOK self.dap_config.NUM_DAP_TOKENS
        # nn.init.zeros_(self.dap_downsample.weight) #OK
        # nn.init.zeros_(self.dap_downsample.bias)   #OK
        # self.dap_film = nn.Linear(TASK_EMB, self.mlp_hidden_dim * 2)   #NOK    dap_config.TASK_EMB  dap_config.NUM_DAP_TOKENS
        # self.dap_norm = nn.LayerNorm(self.mlp_hidden_dim, eps=1e-6) #NOK
        # # down_proj to match shape for concat    
        # self.dim = dim  # Make dim available to inject_prompts
        # self.down_proj = nn.Linear(NUM_DAP_TOKENS, dim) #dap_config.NUM_DAP_TOKENS

        self.dap_config = dap_config
        self.task_emb_dim = TASK_EMB
        #self.dap_norm = nn.LayerNorm(dim)
        #self.dap_downsample = nn.Linear(dim, dim)  # Preserve hidden dim

        # self.dap_downsample = nn.Linear(dim, self.mlp_hidden_dim)
        # self.dap_norm = nn.LayerNorm(dim)  # still use dim here              
        # self.dap_film = nn.Linear(self.task_emb_dim, self.mlp_hidden_dim * 2)
        # #self.down_proj = nn.Identity()  # No dimension change        
        # self.down_proj = nn.Linear(self.mlp_hidden_dim, dim)

        self.dap_downsample = nn.Linear(dim, self.mlp_hidden_dim)
        self.dap_film = nn.Linear(self.task_emb_dim, self.mlp_hidden_dim * 2)
        self.dap_norm = nn.LayerNorm(dim)
        self.down_proj = nn.Linear(self.mlp_hidden_dim, dim)        

        # domain-adaptive prompts
        # self.dap_config = dap_config
        # self.dap_downsample = nn.Linear(197, dap_config.NUM_DAP_TOKENS)
        # nn.init.zeros_(self.dap_downsample.weight)
        # nn.init.zeros_(self.dap_downsample.bias)
        # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
        # self.dap_norm = LayerNorm(config.hidden_size, eps=1e-6)        

    def inject_prompts(self, x, task_id_estimated_emb):
        B, C, H, W = x.shape
        L = H * W
        # x_flat = x.permute(0, 2, 3, 1).reshape(B, L, C)  # [B, L, C]

        # x_norm = self.dap_norm(x_flat)
        # down = self.dap_downsample(x_norm)  # [B, L, C]

        # film = self.dap_film(task_id_estimated_emb)
        # gamma, beta = film[:, :self.mlp_hidden_dim], film[:, self.mlp_hidden_dim:]
        # gamma = gamma.unsqueeze(1)  # [B, 1, mlp_hidden_dim]
        # beta = beta.unsqueeze(1)

        # down = gamma * down + beta  # [B, L, C]

        # # (Optional projection)
        # down = self.down_proj(down)

        # x_concat = torch.cat((x_norm[:, :1, :], down, x_norm[:, 1:, :]), dim=1)  # [B, L+L_prompts, C]

        # if x_concat.shape[1] != L:
        #     x_concat = x_concat[:, :L, :]

        #x_out = x_concat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x_flat = x.permute(0, 2, 3, 1).reshape(B, L, C)
        x_norm = self.dap_norm(x_flat)
        down = self.dap_downsample(x_norm)  # [B, L, 192]

        film = self.dap_film(task_id_estimated_emb)  # [B, 384]
        gamma, beta = film[:, :self.mlp_hidden_dim], film[:, self.mlp_hidden_dim:]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        down = gamma * down + beta  # [B, L, 192]
        down = self.down_proj(down)  # [B, L, 48]

        x_concat = torch.cat((x_norm[:, :1, :], down, x_norm[:, 1:, :]), dim=1)
        if x_concat.shape[1] != L:
            x_concat = x_concat[:, :L, :]

        x_out = x_concat.reshape(B, H, W, C).permute(0, 3, 1, 2)        
        return x_out

    # def inject_prompts(self, x, task_id_estimated_emb): #, layer_index=None, cfg=None):
    #     B, C, H, W = x.shape
    #     L = H * W
    #     x_flat = x.permute(0, 2, 3, 1).reshape(B, L, C)  # [B, L, C]

    #     x_norm = self.dap_norm(x_flat)  # [B, L, C]
    #     down = self.dap_downsample(x_norm)  # [B, L, NUM_DAP_TOKENS]

    #     film = self.dap_film(task_id_estimated_emb) 
    #     gamma4 = film[:, :self.mlp_hidden_dim].unsqueeze(1) #self.dap_config.NUM_DAP_TOKENS
    #     beta4 = film[:, self.mlp_hidden_dim:].unsqueeze(1) #self.dap_config.NUM_DAP_TOKENS
    #     down = gamma4 * down + beta4  # [B, L, NUM_DAP_TOKENS]

    #     # Project down to match dim
    #     down = self.down_proj(down)  # [B, L, C]

    #     # Concatenate prompts
    #     x_concat = torch.cat((x_norm[:, :1, :], down, x_norm[:, 1:, :]), dim=1)  # [B, L+L_prompts, C]

    #     # No reshape to spatial unless known it's safe.
    #     # If must reshape: check x_concat.shape[1] == H*W or infer correctly.

    #     # For now, project back to spatial shape of original x: 
    #     # => Drop or adjust number of tokens if necessary!
    #     if x_concat.shape[1] != L:
    #         # Example fix: truncate or interpolate
    #         x_concat = x_concat[:, :L, :]  # Simple truncate

    #     x_out = x_concat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
    #     return x_out


    def forward(self, x, task_id_emb=None): #, layer_index=None, cfg=None
        """
        Forward pass with optional DAP prompt injection.
        """
        x = x + self.local_perception(x)

        # if task_id_emb is not None:
        #     x = self.inject_prompts(x, task_id_emb) #, layer_index=layer_index, cfg=cfg
        # else:
        #     dummy_task_id_emb = torch.zeros((x.size(0), TASK_EMB), device=x.device) #self.dap_config.TASK_EMB
        #     x = self.inject_prompts(x, dummy_task_id_emb) #, layer_index=layer_index, cfg=cfg

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
   
#EVANDRO - Adaptado ao original do CAS-ViT - OK 
def Stage(dim, index, layers, mlp_ratio=4., act_layer=nn.GELU, attn_bias=False, drop=0., drop_path_rate=0.): #, dap_config=None
    """
    Create a stage for RCViT with support for DAP.
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)

        blocks.append(
            AdditiveBlock(
                dim, mlp_ratio=mlp_ratio, attn_bias=attn_bias, drop=drop, drop_path=block_dpr,
                act_layer=act_layer, norm_layer=nn.BatchNorm2d) #, dap_config=dap_config
        )
    blocks = nn.Sequential(*blocks)
    return blocks
 

# #ORIGINAL FROM CAS-VIT
# class AdditiveBlock(nn.Module):
#     """
#     """
#     def __init__(self, dim, mlp_ratio=4., attn_bias=False, drop=0., drop_path=0.,
#                  act_layer=nn.ReLU, norm_layer=nn.GELU):
#         super().__init__()
#         self.local_perception = LocalIntegration(dim, ratio=1, act_layer=act_layer, norm_layer=norm_layer)
#         self.norm1 = norm_layer(dim)
#         self.attn = AdditiveTokenMixer(dim, attn_bias=attn_bias, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x):
#         x = x + self.local_perception(x)
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

# def Stage(dim, index, layers, mlp_ratio=4., act_layer=nn.GELU, attn_bias=False, drop=0., drop_path_rate=0.):
#     """
#     """
#     blocks = []
#     for block_idx in range(layers[index]):
#         block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)

#         blocks.append(
#             AdditiveBlock(
#                 dim, mlp_ratio=mlp_ratio, attn_bias=attn_bias, drop=drop, drop_path=block_dpr,
#                 act_layer=act_layer, norm_layer=nn.BatchNorm2d)
#         )
#     blocks = nn.Sequential(*blocks)
#     return blocks



#====================================


class RCViT(nn.Module):
    def __init__(self, layers, embed_dims, mlp_ratios=4, downsamples=[True, True, True, True], norm_layer=nn.BatchNorm2d, attn_bias=False,
                 act_layer=nn.GELU, num_classes=1000, drop_rate=0., drop_path_rate=0., fork_feat=False,
                 init_cfg=None, pretrained=None, distillation=True, **kwargs):
        super().__init__()


        self.task_emb_dim = TASK_EMB #Use TASK_EMB = 64 or 128 (same as DAP default).
        self.prompt_pool = 20 # 10 or 20
        self.top_k = 5 #Set top_k = 5 or 10 (DAP uses prompt pool size = 10). Antes eu tinha coloca 1

        # DAP projection layers
        # self.dap_cls_proj = nn.Linear(embed_dims[-1], self.task_emb_dim)
        # self.reduce_sim_proj = nn.Linear(embed_dims[-1], self.task_emb_dim)
        # nn.init.xavier_uniform_(self.dap_cls_proj.weight)
        # nn.init.zeros_(self.dap_cls_proj.bias)
        # nn.init.xavier_uniform_(self.reduce_sim_proj.weight)
        # nn.init.zeros_(self.reduce_sim_proj.bias)

        self.dap_key_embeddings = nn.Parameter(torch.randn(self.prompt_pool, self.task_emb_dim))
        nn.init.xavier_uniform_(self.dap_key_embeddings)

        #TASK_EMB = 256  # or 128/768 depending on your model
        self.reduce_sim_proj = nn.Linear(embed_dims[-1], self.task_emb_dim, bias=False)
        nn.init.normal_(self.reduce_sim_proj.weight, std=0.1)
        self.dap_key_embeddings = nn.Parameter(torch.randn(self.prompt_pool, self.task_emb_dim) * 0.1)



        # DAP projection layers
        #=================================================================================================        
        # #self.dap_config = cfg.MODEL.DAP
        # self.task_emb_dim = TASK_EMB #self.dap_config.TASK_EMB
        # self.prompt_pool = 10 #self.dap_config.PROMPT_POOL
        # self.top_k = 1 #getattr(self.dap_config, "TOP_K", 1)

        # self.dap_cls_proj = nn.Linear(embed_dims[-1], self.task_emb_dim)
        # self.reduce_sim_proj = nn.Linear(embed_dims[-1], self.task_emb_dim)
        # nn.init.xavier_uniform_(self.dap_cls_proj.weight)
        # nn.init.zeros_(self.dap_cls_proj.bias)
        # nn.init.xavier_uniform_(self.reduce_sim_proj.weight)
        # nn.init.zeros_(self.reduce_sim_proj.bias)

        # self.dap_key_embeddings = nn.Parameter(torch.randn(self.prompt_pool, self.task_emb_dim))
        # nn.init.xavier_uniform_(self.dap_key_embeddings)        

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = stem(3, embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = Stage(embed_dims[i], i, layers, mlp_ratio=mlp_ratios, act_layer=act_layer,
                          attn_bias=attn_bias, drop=drop_rate, drop_path_rate=drop_path_rate)

            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    Embedding(
                        patch_size=3, stride=2, padding=1, in_chans=embed_dims[i],
                        embed_dim=embed_dims[i+1], norm_layer=nn.BatchNorm2d)
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(
                    embed_dims[-1], num_classes) if num_classes > 0 \
                    else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        # logger = get_root_logger()
        # if self.init_cfg is None and pretrained is None:
        #     logger.warn(f'No pre-trained weights for '
        #                 f'{self.__class__.__name__}, '
        #                 f'training start from scratch')
        #     pass
        # else:
        #     assert 'checkpoint' in self.init_cfg, f'Only support ' \
        #                                           f'specify `Pretrained` in ' \
        #                                           f'`init_cfg` in ' \
        #                                           f'{self.__class__.__name__} '
        #     if self.init_cfg is not None:
        #         ckpt_path = self.init_cfg['checkpoint']
        #     elif pretrained is not None:
        #         ckpt_path = pretrained
        #
        #     ckpt = _load_checkpoint(
        #         ckpt_path, logger=logger, map_location='cpu')
        #     if 'state_dict' in ckpt:
        #         _state_dict = ckpt['state_dict']
        #     elif 'model' in ckpt:
        #         _state_dict = ckpt['model']
        #     else:
        #         _state_dict = ckpt
        #
        #     state_dict = _state_dict
        #     missing_keys, unexpected_keys = \
        #         self.load_state_dict(state_dict, False)
        pass    


    def reducesim_calculation(self, x):
        """
        DAP-style ReduceSim computation for CAS-ViT:
        - Computes pulling force between projected CLS token and prompt keys.
        - Output: scalar ReduceSim (~0.5 to 5.0+), similar to DAP.
        """
        # x: [B, C, H, W] or [B, N, D]
        if x.dim() == 4:
            cls_embed = x.mean(dim=[2, 3])  # [B, C]
        elif x.dim() == 3:
            cls_embed = x.mean(dim=1)       # [B, D]
        else:
            raise ValueError("Unexpected shape for x")

        # Apply controlled noise (improves generalization and sim spread)
        cls_embed = cls_embed + 0.1 * torch.randn_like(cls_embed)

        cls_embed = x.flatten(2).mean(-1)  # [B, C] #EVANDRO TO IMPROVE ACC

        # Project to TASK_EMB dimension
        cls_proj = self.reduce_sim_proj(cls_embed)               # [B, TASK_EMB]
        cls_proj_norm = F.normalize(cls_proj, dim=-1)            # [B, TASK_EMB]
        prompt_keys_norm = F.normalize(self.dap_key_embeddings, dim=-1)  # [P, TASK_EMB]

        # Cosine similarity between each sample and prompt keys: [B, P]
        sim = torch.matmul(cls_proj_norm, prompt_keys_norm.T)    # [B, P]

        # Select top-k similarities
        topk_sim, topk_idx = torch.topk(sim, self.top_k, dim=-1) # [B, K]

        # Pull selected prompt keys
        selected_keys = prompt_keys_norm[topk_idx]               # [B, K, D]
        cls_exp = cls_proj_norm.unsqueeze(1)                     # [B, 1, D]

        # Pulling force = cosine sim * cls alignment
        sim_pull = (selected_keys * cls_exp).sum(dim=-1)         # [B, K]

        # Final ReduceSim = average pulling force across top-k, then batch
        reduce_sim = sim_pull.mean(dim=1).mean() * self.top_k    # scale for magnitude
        #reduce_sim = topk_sim.mean(dim=1).mean()  # scalar

        return reduce_sim, cls_embed

    # def reducesim_calculation(self, x): #no improvements
    #     """
    #     DAP-style ReduceSim computation:
    #     Produces values like 0.5 ~ 5.0+ (not capped at 1.0).
    #     """
    #     # x: [B, C, H, W] or [B, N, D]
    #     if x.dim() == 4:
    #         cls_embed = x.mean(dim=[2, 3])  # [B, C]
    #     elif x.dim() == 3:
    #         cls_embed = x.mean(dim=1)       # [B, D]
    #     else:
    #         raise ValueError("Unexpected shape for x")

    #     # Project CLS to TASK_EMB
    #     cls_proj = self.reduce_sim_proj(cls_embed)   # [B, TASK_EMB]

    #     # Normalize both cls_proj and prompt keys
    #     cls_proj_norm = F.normalize(cls_proj, dim=-1)                          # [B, D]
    #     prompt_keys_norm = F.normalize(self.dap_key_embeddings, dim=-1)        # [P, D]

    #     # Cosine similarity matrix [B, P]
    #     sim = torch.matmul(cls_proj_norm, prompt_keys_norm.T)

    #     # Top-k similarities [B, K]
    #     topk_sim, _ = torch.topk(sim, self.top_k, dim=-1)

    #     # Average top-k sim per sample, then mean over batch
    #     reduce_sim = topk_sim.mean(dim=1).mean()

    #     return reduce_sim, cls_embed


#     # def reducesim_calculation(self, x): #valores na faixa de 0 a 0.99
#     #     # CLS embedding
#     #     cls_embed = x.flatten(2).mean(-1)  # [B, embed_dim]

#     #     # Project to TASK_EMB dimension
#     #     cls_embed_proj = self.reduce_sim_proj(cls_embed)  # [B, TASK_EMB]

#     #     # Normalize
#     #     cls_embed_norm = F.normalize(cls_embed_proj, dim=-1, eps=1e-6)
#     #     dap_prompt_key_norm = F.normalize(self.dap_key_embeddings, dim=-1, eps=1e-6)

#     #     # Similarity computation
#     #     sim = torch.matmul(cls_embed_norm, dap_prompt_key_norm.T)  # [B, P]

#     #     # Top-k prompt selection
#     #     _, idx = torch.topk(sim, self.top_k, dim=-1)  # [B, K]
#     #     selected_prompt_key = dap_prompt_key_norm[idx]  # [B, K, TASK_EMB]

#     #     # Expand cls_embed for broadcast
#     #     cls_embed_exp = cls_embed_norm.unsqueeze(1)  # [B, 1, TASK_EMB]
#     #     sim_pull = selected_prompt_key * cls_embed_exp  # [B, K, TASK_EMB]

#     #     reduce_sim = sim_pull.sum() / cls_embed.shape[0]  # scalar
#     #     reduce_sim = torch.clamp(reduce_sim, -1e2, 1e2)

#     #     return reduce_sim, cls_embed  # cls_embed is used for classification head
        
#     # def reducesim_calculation(self, x): #valores na faixa 0.06xxx
#     #     # CLS embedding
#     #     cls_embed = x.flatten(2).mean(-1)  # [B, embed_dim]

#     #     # Project to TASK_EMB dimension
#     #     cls_embed_proj = self.reduce_sim_proj(cls_embed)  # [B, TASK_EMB]

#     #     # Normalize
#     #     cls_embed_norm = F.normalize(cls_embed_proj, dim=-1, eps=1e-6)
#     #     dap_prompt_key_norm = F.normalize(self.dap_key_embeddings, dim=-1, eps=1e-6)

#     #     # Similarity computation
#     #     sim = torch.matmul(cls_embed_norm, dap_prompt_key_norm.T)  # [B, P]

#     #     # Top-k prompt selection
#     #     _, idx = torch.topk(sim, self.top_k, dim=-1)  # [B, K]
#     #     selected_prompt_key = dap_prompt_key_norm[idx]  # [B, K, TASK_EMB]

#     #     # Expand cls_embed for broadcasting
#     #     cls_embed_exp = cls_embed_norm.unsqueeze(1)  # [B, 1, TASK_EMB]

#     #     # Element-wise similarity pull
#     #     sim_pull = selected_prompt_key * cls_embed_exp  # [B, K, TASK_EMB]

#     #     # Corrected reduce_sim calculation (use mean instead of sum)
#     #     reduce_sim = sim_pull.mean()  # scalar

#     #     reduce_sim = torch.clamp(reduce_sim, -1e2, 1e2)
#     #     return reduce_sim, cls_embed

    

    def forward_tokens(self, x, task_id_emb=None):
        outs = []
        for idx, block in enumerate(self.network):
            if isinstance(block, AdditiveBlock):
                x = block(x, task_id_emb=task_id_emb)
            else:
                x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x, task_id_emb=None):
        x = self.patch_embed(x)
        x = self.forward_tokens(x, task_id_emb=task_id_emb)
        if self.fork_feat:
            return x
        x = self.norm(x)
        reduce_sim, x_flat = self.reducesim_calculation(x) 
        if self.dist:
            cls_out = self.head(x_flat), self.dist_head(x_flat)  
            #cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2 
        else:
            cls_out = self.head(x_flat) 
            #cls_out = self.head(x.flatten(2).mean(-1))            

        return cls_out, reduce_sim

    #original from CAS-ViT
    # def forward(self, x):
    #     x = self.patch_embed(x)
    #     x = self.forward_tokens(x)
    #     if self.fork_feat:
    #         # otuput features of four stages for dense prediction
    #         return x
    #     x = self.norm(x)
    #     if self.dist:
    #         cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
    #         if not self.training:
    #             cls_out = (cls_out[0] + cls_out[1]) / 2
    #     else:
    #         cls_out = self.head(x.flatten(2).mean(-1))
    #     # for image classification
    #     return cls_out        
 
# ======================================================================================================================

@register_model
def rcvit_xs(**kwargs): #embed_dims = [64, 128, 256, 512] use larger CAS-ViT depth: [3, 4, 6, 3] or [3, 6, 9, 3]
    model = RCViT(
        layers=[2, 2, 4, 2], embed_dims=[48, 56, 112, 220], mlp_ratios=4, downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0.,
        fork_feat=False, init_cfg=None, **kwargs)
    return model

@register_model
def rcvit_s(**kwargs):
    model = RCViT(
        layers=[3, 3, 6, 3], embed_dims=[48, 64, 128, 256], mlp_ratios=4, downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0.,
        fork_feat=False, init_cfg=None, **kwargs)
    return model

@register_model
def rcvit_m(**kwargs):
    model = RCViT(
        layers=[3, 3, 6, 3], embed_dims=[64, 96, 192, 384], mlp_ratios=4, downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0.,
        fork_feat=False, init_cfg=None, **kwargs)
    return model

@register_model
def rcvit_t(**kwargs):
    model = RCViT(
        layers=[3, 3, 6, 3], embed_dims=[96, 128, 256, 512], mlp_ratios=4, downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0.,
        fork_feat=False, init_cfg=None, **kwargs)
    return model


# ======================================================================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    net = rcvit_xs()
    x = torch.rand((1, 3, 224, 224))
    out = net(x)

    print('Net Params: {:d}'.format(int(count_parameters(net))))

