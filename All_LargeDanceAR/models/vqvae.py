
'''
RVQVAE from MOMASK for 
'''

import random
from random import randrange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize
from einops import pack, rearrange, repeat, unpack
from models.encdec import Decoder, Encoder
from torch import nn


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    dim = -1,
    training = True
):

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)

    return ind
class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super(QuantizeEMAReset, self).__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu  ##TO_DO
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def quantize(self, x, sample_codebook_temp=0.):
        # N X C -> C X N
        k_w = self.codebook.t()
        # x: NT X C
        # NT X N
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - \
                   2 * torch.matmul(x, k_w) + \
                   torch.sum(k_w ** 2, dim=0, keepdim=True)  # (N * L, b)

        # code_idx = torch.argmin(distance, dim=-1)

        code_idx = gumbel_sample(-distance, dim = -1, temperature = sample_codebook_temp, stochastic=True, training = self.training)

        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x
    
    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device) # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x) # nb_code, c
        code_count = code_onehot.sum(dim=-1) # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1-usage) * code_rand


        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        # x = x.permute(0, 2, 1).contiguous()
        # x = x.view(-1, x.shape[-1])
        x = rearrange(x, 'n c t -> (n t) c')
        return x

    def forward(self, x, return_idx=False, temperature=0.):
        N, width, T = x.shape

        x = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(x)

        code_idx = self.quantize(x, temperature)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)
        if self.training:
            # code_idx 的形状是 [N * T]，统计每个码的使用次数
            code_usage = torch.bincount(code_idx, minlength=self.nb_code).float()  # [nb_code]
            code_usage_ratio = code_usage / code_usage.sum()  # 归一化到 [0, 1]
        commit_loss = F.mse_loss(x, x_d.detach()) # It's right. the t2m-gpt paper is wrong on embed loss and commitment loss.

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()
        # print(code_idx[0])
        if return_idx:
            return x_d, code_idx, commit_loss, perplexity
        return x_d, commit_loss, perplexity
    

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        num_quantizers,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        **kwargs
    ):
        super().__init__()

        self.num_quantizers = num_quantizers

        # self.layers = nn.ModuleList([VectorQuantize(accept_image_fmap = accept_image_fmap, **kwargs) for _ in range(num_quantizers)])
        if shared_codebook:
            layer = QuantizeEMAReset(**kwargs)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])
        # self.layers = nn.ModuleList([QuantizeEMA(**kwargs) for _ in range(num_quantizers)])

        # self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob

            
    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks # 'q c d'
    
    def get_codes_from_indices(self, indices): #indices shape 'b n q' # dequantize

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # get ready for gathering

        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b = batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d = codebooks.shape[-1])

        # take care of quantizer dropout

        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        # print(gather_indices.max(), gather_indices.min())
        all_codes = codebooks.gather(2, gather_indices) # gather all codes

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(mask, 0.)

        return all_codes # 'q b n d'

    def get_codebook_entry(self, indices): #indices shape 'b n q'
        all_codes = self.get_codes_from_indices(indices) #'q b n d'
        latent = torch.sum(all_codes, dim=0) #'b n d'
        latent = latent.permute(0, 2, 1)
        return latent

    def forward(self, x, return_all_codes=False, sample_codebook_temp=None, force_dropout_index=-1):
            num_quant, quant_dropout_prob, device = self.num_quantizers, self.quantize_dropout_prob, x.device

            quantized_out = 0.
            residual = x

            all_losses = []
            all_indices = []
            all_perplexity = []
            utilization_stats = []  # 新增：存储每个量化器的利用率统计

            should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob

            start_drop_quantize_index = num_quant
            if should_quantize_dropout:
                start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant)
                null_indices_shape = [x.shape[0], x.shape[-1]]
                null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)

            if force_dropout_index >= 0:
                should_quantize_dropout = True
                start_drop_quantize_index = force_dropout_index
                null_indices_shape = [x.shape[0], x.shape[-1]]
                null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)

            # 遍历每一层并计算利用率
            for quantizer_index, layer in enumerate(self.layers):
                if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                    all_indices.append(null_indices)
                    # 对于丢弃的层，添加占位符统计信息
                    utilization_stats.append({
                        'mean': 0.0,
                        'std': 0.0,
                        'non_zero_codes': 0,
                        'total_codes': layer.nb_code
                    })
                    continue

                quantized, *rest = layer(residual, return_idx=True, temperature=sample_codebook_temp)

                residual -= quantized.detach()
                quantized_out += quantized

                embed_indices, loss, perplexity = rest
                all_indices.append(embed_indices)
                all_losses.append(loss)
                all_perplexity.append(perplexity)

                # 计算码本利用率
                if self.training:
                    code_usage = torch.bincount(embed_indices.view(-1), minlength=layer.nb_code).float()
                    code_usage_ratio = code_usage / code_usage.sum()
                    non_zero_codes = (code_usage > 0).sum().item()
                    utilization_stats.append({
                        'mean': code_usage_ratio.mean().item(),
                        'std': code_usage_ratio.std().item(),
                        'non_zero_codes': non_zero_codes,
                        'total_codes': layer.nb_code
                    })

            # stack all losses and indices
            all_indices = torch.stack(all_indices, dim=-1)
            all_losses = sum(all_losses) / len(all_losses) if all_losses else torch.tensor(0.0, device=device)
            all_perplexity = sum(all_perplexity) / len(all_perplexity) if all_perplexity else torch.tensor(0.0, device=device)

            ret = (quantized_out, all_indices, all_losses, all_perplexity, utilization_stats)  # 添加 utilization_stats

            if return_all_codes:
                all_codes = self.get_codes_from_indices(all_indices)
                ret = (*ret, all_codes)

            return ret
    
    def quantize(self, x, return_latent=False):
        all_indices = []
        quantized_out = 0.
        residual = x
        all_codes = []
        for quantizer_index, layer in enumerate(self.layers):

            quantized, *rest = layer(residual, return_idx=True) #single quantizer

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            # print(quantizer_index, embed_indices[0])
            # print(quantizer_index, quantized[0])
            # break
            all_codes.append(quantized)

        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        if return_latent:
            return code_idx, all_codes
        return code_idx       
        
        
class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=264,#dim_pose
                 nb_code=1024,#nb_code
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim
        # breakpoint()
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
 
        self.vel_decoder = Decoder(2, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)  # new vel_decoder
        self.rot_decoder = Decoder(128, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)  # new rot_decoder 1roty+rootyvec+23*6D
        if args.vel_decoder:
            for name, param in self.named_parameters():
                if 'vel_decoder' in name :
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif args.rot_decoder:
            for name, param in self.named_parameters():
                if "rot_decoder" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif not args.vel_decoder and not args.rot_decoder:
            for name, param in self.named_parameters():
                if 'vel_decoder' not in name and "rot_decoder" not in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    # print(f"Training parameter: {name}")
        #change
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print("x_encoder",x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print("code_idx",code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        x_quantized, code_idx, commit_loss, perplexity, utilization_stats = self.quantizer(
            x_encoder, sample_codebook_temp=0.5
        )

        ## decoder
        x_out = self.decoder(x_quantized)
        x_xz_vel_out = self.vel_decoder(x_quantized)  # 预测 XZ 速度
        x_rot_out=self.rot_decoder(x_quantized)

        return x_out,x_xz_vel_out,x_rot_out, commit_loss, perplexity, utilization_stats
    def forward_quantizer(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_quantized, code_idx, commit_loss, perplexity, utilization_stats = self.quantizer(
            x_encoder, sample_codebook_temp=0.5
        )
        return x_quantized

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)