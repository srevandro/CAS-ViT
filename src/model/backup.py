
    # def reducesim_calculation(self, x):
    #     # x is the transformer token output before norm and head

    #     # CLS embedding
    #     cls_embed = x.flatten(2).mean(-1)  # [B, embed_dim]

    #     # Project to TASK_EMB dimension
    #     cls_embed_proj = self.reduce_sim_proj(cls_embed)  # [B, TASK_EMB]

    #     # Normalize
    #     cls_embed_norm = F.normalize(cls_embed_proj, dim=-1, eps=1e-6)
    #     dap_prompt_key_norm = F.normalize(self.dap_key_embeddings, dim=-1, eps=1e-6)

    #     # Similarity computation
    #     sim = torch.matmul(cls_embed_norm, dap_prompt_key_norm.T)  # [B, P]

    #     # Top-k prompt selection
    #     _, idx = torch.topk(sim, self.top_k, dim=-1)  # [B, K]
    #     selected_prompt_key = dap_prompt_key_norm[idx]  # [B, K, TASK_EMB]

    #     # Expand cls_embed for broadcast
    #     cls_embed_exp = cls_embed_norm.unsqueeze(1)  # [B, 1, TASK_EMB]
    #     sim_pull = selected_prompt_key * cls_embed_exp  # [B, K, TASK_EMB]

    #     reduce_sim = sim_pull.sum() / cls_embed.shape[0]  # scalar
    #     reduce_sim = torch.clamp(reduce_sim, -1e2, 1e2)

    #     return reduce_sim, cls_embed  # cls_embed is used for classification head

    #def forward_tokens(self, x):
    # def forward_tokens(self, x, task_id_emb=None):    
    #     outs = []
    #     for idx, block in enumerate(self.network):
    #     #    x = block(x)
    #         if isinstance(block, AdditiveBlock):
    #             x = block(x, task_id_emb=task_id_emb)
    #         else:
    #             x = block(x)        
    #         if self.fork_feat and idx in self.out_indices:
    #             norm_layer = getattr(self, f'norm{idx}')
    #             x_out = norm_layer(x)
    #             outs.append(x_out)
    #     if self.fork_feat:
    #         return outs
    #     return x 

    # #def forward(self, x):
    # def forward(self, x, task_id_emb=None): #, is_train=None, cfg=None     
    #     x = self.patch_embed(x)
    #     x = self.forward_tokens(x, task_id_emb=task_id_emb)
    #     if self.fork_feat:
    #         # output features of four stages for dense prediction
    #         return x
    #     x = self.norm(x)

    #     # Compute reduce_sim and x_flat for classification
    #     reduce_sim, x_flat = self.reducesim_calculation(x)

    #     if self.dist:
    #         cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
    #         if not self.training:
    #             cls_out = (cls_out[0] + cls_out[1]) / 2
    #     else:
    #         cls_out = self.head(x.flatten(2).mean(-1))
    #     # for image classification
    #     return cls_out, reduce_sim
