import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from RetrievalNet.models import *
from RetrievalNet.models.rp_clip.clip_encoder import CLIPTextEncoder
from RetrievalNet.models.rp_clip.trans_ae import TransEncoder

# MotionEncoder()
loss_ce = nn.CrossEntropyLoss()

# 这个InterCLIP_RP主要是为了测R precision
class InterCLIP_RP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.clip_training = "cond_"
        self.latent_dim = self.latent_dim
        
        self.motion_encoder = TransEncoder(cfg.TransEncoder)
        self.cond_encoder = TransEncoder(cfg.CondEncoder)

        self.l1_criterion = torch.nn.L1Loss(reduction='mean')
        # self.latent_scale = nn.Parameter(torch.Tensor([1]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def compute_loss(self, batch):
        losses = {}
        losses["total"] = 0

        batch = self.encode_motion(batch)
        batch = self.encode_cond(batch)

        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)
        losses.update(clip_losses)
        losses["total"] += mixed_clip_loss

        return losses["total"], losses

    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.
        clip_losses = {}

        if 1:
            for d in self.clip_training.split('_')[:1]:
                if d == 'image':
                    features = self.clip_model.encode_image(batch['images']).float()  # preprocess is done in dataloader
                elif d == 'text':
                    features = batch['text_emb']
                elif d == 'cond':
                    features = batch['cond_emb']
                motion_features = batch['motion_emb']
                # normalized features
                # features_norm = features / features.norm(dim=-1, keepdim=True)
                # motion_features_norm = motion_features / motion_features.norm(dim=-1, keepdim=True)

                # logit_scale = self.latent_scale ** 2
                # logits_per_motion = motion_features @ features.t()
                
                # logit_scale = self.clip_model.clip.logit_scale.exp()
                logit_scale = self.logit_scale.exp()
                logits_per_motion = logit_scale * motion_features @ features.t()
                
                logits_per_d = logits_per_motion.t()

                batch_size = motion_features.shape[0]
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=motion_features.device)

                ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

                clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss
                clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss
                clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss
                mixed_clip_loss += clip_mixed_loss

        return mixed_clip_loss, clip_losses

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def encode_motion(self, batch):
        device = next(self.parameters()).device
        # batch["mask"] = self.generate_src_mask(batch["motions"].shape[1], batch["motions"].shape[1]).to(batch["motions"].device)
        batch["mask"] = torch.ones(batch["motions"].shape[0], batch["motions"].shape[1]).to(device)
        batch["motions"] = batch["motions"].to(device)

        # batch.update(self.motion_encoder(batch))
        batch["motion_emb"]  = self.motion_encoder(batch["motions"], batch["mask"])
        batch["motion_emb"] = batch["motion_emb"] / batch["motion_emb"].norm(dim=-1, keepdim=True) 
        return batch
    
    def encode_cond(self, batch):
        device = next(self.parameters()).device
        
        audio = batch["audio"]
        joints = batch["joints"]
        condition = torch.cat((audio, joints), dim=-1).to(device)
    
        batch["mask"] = torch.ones(condition.shape[0], condition.shape[1]).to(device)
        batch["cond_emb"] = self.cond_encoder(condition, batch["mask"])
        batch["cond_emb"] = batch["cond_emb"] / batch["cond_emb"].norm(dim=-1, keepdim=True)
        return batch

    def encode_text(self, batch):
        device = next(self.parameters()).device
        raw_text = batch["text"]
        out = self.clip_model(raw_text, device, None)
        
        batch['text_emb'] = out
        batch["text_emb"] = batch["text_emb"] / batch["text_emb"].norm(dim=-1, keepdim=True)
        return batch




class InterCLIP_AudioJoints(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.clip_training = "cond_"
        self.latent_dim = self.latent_dim
        
        self.motion_encoder = TransEncoder(cfg.TransEncoder)
        self.cond_encoder = TransEncoder(cfg.CondEncoder)

        self.l1_criterion = torch.nn.L1Loss(reduction='mean')
        # self.latent_scale = nn.Parameter(torch.Tensor([1]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def compute_loss(self, batch):
        losses = {}
        losses["total"] = 0

        batch = self.encode_motion(batch)
        batch = self.encode_cond(batch)

        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)
        losses.update(clip_losses)
        losses["total"] += mixed_clip_loss

        return losses["total"], losses

    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.
        clip_losses = {}

        if 1:
            for d in self.clip_training.split('_')[:1]:
                if d == 'image':
                    features = self.clip_model.encode_image(batch['images']).float()  # preprocess is done in dataloader
                elif d == 'text':
                    features = batch['text_emb']
                elif d == 'cond':
                    features = batch['cond_emb']
                motion_features = batch['motion_emb']
                # normalized features
                # features_norm = features / features.norm(dim=-1, keepdim=True)
                # motion_features_norm = motion_features / motion_features.norm(dim=-1, keepdim=True)

                # logit_scale = self.latent_scale ** 2
                # logits_per_motion = motion_features @ features.t()
                
                # logit_scale = self.clip_model.clip.logit_scale.exp()
                logit_scale = self.logit_scale.exp()
                logits_per_motion = logit_scale * motion_features @ features.t()
                
                logits_per_d = logits_per_motion.t()

                batch_size = motion_features.shape[0]
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=motion_features.device)

                ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

                clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss
                clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss
                clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss
                mixed_clip_loss += clip_mixed_loss

        return mixed_clip_loss, clip_losses

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def encode_motion(self, batch):
        device = next(self.parameters()).device
        batch["mask"] = torch.ones(batch["joints"].shape[0], batch["joints"].shape[1]).to(device)
        batch["joints"] = batch["joints"].to(device)
        
        batch["motion_emb"]  = self.motion_encoder(batch["joints"], batch["mask"])
        batch["motion_emb"] = batch["motion_emb"] / batch["motion_emb"].norm(dim=-1, keepdim=True) 
        return batch
    
    def encode_cond(self, batch):
        device = next(self.parameters()).device
        
        condition = batch["audio"].to(device)
        batch["mask"] = torch.ones(condition.shape[0], condition.shape[1]).to(device)
        batch["cond_emb"] = self.cond_encoder(condition, batch["mask"])
        batch["cond_emb"] = batch["cond_emb"] / batch["cond_emb"].norm(dim=-1, keepdim=True)
        return batch
