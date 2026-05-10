"""
train_v2/train_main.py — 简化的 fresh-train 入口，集成:
  - bf16 / fp16 (autocast + model 转换)
  - gradient accumulation
  - 全量 LLaMA 微调（默认 freeze_layers=0，可改）
  - DDP via mp.spawn

不支持从旧 ckpt resume（旧 ckpt 跟新 bridge 名字对不上）。
保存格式: rank-0 torch.save(module.state_dict()) → 单文件 ckpt，跟现有 inference 兼容。
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from models.dancellama import Music2DanceLlamaModel, MusicDanceDataset, variable_length_collate_fn
from models.motion import load_vqvae_model
from models.train_utils import _topk_correct


GENRES = ["Ballet", "Popular", "Latin", "Modern", "Folk", "Classic"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--world_size", type=int, default=2)
    p.add_argument("--MASTER_PORT", type=str, default="17795")
    # 数据
    p.add_argument("--music_dir", required=True)
    p.add_argument("--dance_dir", required=True)
    p.add_argument("--data_split_dir", required=True)
    p.add_argument("--style_dir", required=True)
    p.add_argument("--dancedata", default="All")
    p.add_argument("--vqvae_checkpoint_path", required=True)
    p.add_argument("--mean_path", required=True)
    p.add_argument("--std_path", required=True)
    p.add_argument("--llama_config_path", default="models/Llama3.2-1B/config.json")
    p.add_argument("--pretrained_model_name", default="models/Llama3.2-1B")
    # 模型形状
    p.add_argument("--style_embedding_dim", type=int, default=128)
    p.add_argument("--n_bins", type=int, default=2)
    p.add_argument("--music_length", type=int, default=320)
    p.add_argument("--dance_length", type=int, default=288)
    p.add_argument("--window_stride", type=int, default=144)
    p.add_argument("--dance_token_start", type=int, default=4096)
    p.add_argument("--dance_token_increment", type=int, default=512 * 3)
    p.add_argument("--dance_ranges_count", type=int, default=1)
    p.add_argument("--special_tokens_count", type=int, default=256)
    p.add_argument("--use_music_cross_attn", action="store_true", default=False)
    # 优化
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--learning_rate2", type=float, default=None,
                   help="Stage 2 LR;不指定则沿用 learning_rate")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--llama_dropout", type=float, default=0.05)
    p.add_argument("--cond_drop_prob", type=float, default=0.0,
                   help="probability of dropping music conditioning during training")
    p.add_argument("--early_token_weight", type=float, default=3.0,
                   help="前 N 帧 dance loss 加权(逼无 dance 历史时用 music)")
    p.add_argument("--early_token_count", type=int, default=60,
                   help="前 N 帧加权(对应 retrieval prefix 长度)")
    p.add_argument("--freeze_llama_layers", type=int, default=0,
                   help="冻底 N 层 LLaMA。默认 0 = 全量微调")
    # 精度
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    # 数据加载
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    # 训练流程
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="path to a previous stage-2 checkpoint to warm-start from")
    # === Stage 1 / Stage 2 配置 ===
    p.add_argument("--stage1_epoch", type=int, default=2,
                   help="Stage 1: 只训 bridge + style + cond_proj + classifier,LLaMA 全冻")
    p.add_argument("--stage2_epoch", type=int, default=20,
                   help="Stage 2: LLaMA 解冻(底 N 层可继续冻,见 freeze_llama_layers),全量微调")
    p.add_argument("--target_train_acc", type=float, default=0.90,
                   help="train_acc 达到此阈值就早停(每 epoch 末检查)")
    return p.parse_args()


def setup_ddp(rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def precision_dtype(name):
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def maybe_freeze(model_module, n_freeze):
    """冻底 N 层 LLaMA (model.module.llama.model.layers[0..n-1])"""
    if n_freeze <= 0:
        return
    layers = model_module.llama.model.layers
    n_freeze = min(n_freeze, len(layers))
    for i in range(n_freeze):
        for p in layers[i].parameters():
            p.requires_grad = False


def set_stage1_freeze(model_module):
    """Stage 1: 只训 bridge + style + cond_proj + classifier,llama 全冻(包括 embed_tokens / lm_head)"""
    for p in model_module.llama.parameters():
        p.requires_grad = False


def set_stage2_unfreeze(model_module, n_freeze_llama):
    """Stage 2: 整个 llama 解冻,然后按 freeze_llama_layers 冻底 N 层"""
    for p in model_module.llama.parameters():
        p.requires_grad = True
    if n_freeze_llama > 0:
        layers = model_module.llama.model.layers
        n = min(n_freeze_llama, len(layers))
        for i in range(n):
            for p in layers[i].parameters():
                p.requires_grad = False


def count_params(model_module):
    trainable = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model_module.parameters() if not p.requires_grad)
    return trainable, frozen


def get_lr_scheduler(optimizer, num_steps, warmup, kind):
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
    if kind == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup, num_steps)
    return get_linear_schedule_with_warmup(optimizer, warmup, num_steps)


def save_state_dict(path, model_module, ema_unused=False):
    """rank-0 only: torch.save module.state_dict()"""
    state_dict = {k: v.detach().cpu() for k, v in model_module.state_dict().items()}
    torch.save(state_dict, path)


def train_worker(rank, world_size, args, codebooks_arg):
    setup_ddp(rank, world_size, args.MASTER_PORT)
    is_main = rank == 0
    device = torch.device("cuda", rank)
    dtype = precision_dtype(args.precision)
    use_amp = args.precision in ("bf16", "fp16")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"[rank0] precision={args.precision} ws={world_size} bs/gpu={args.batch_size} grad_accum={args.gradient_accumulation_steps} effective_batch={args.batch_size * world_size * args.gradient_accumulation_steps}")

    # 数据
    if is_main:
        print("Loading datasets...")
    train_dataset = MusicDanceDataset(
        music_dir=args.music_dir, dance_dir=args.dance_dir,
        data_split_dir=args.data_split_dir, style_dir=args.style_dir,
        split="train", music_length=args.music_length, dance_length=args.dance_length,
        max_samples=args.max_train_samples, verbose=is_main,
        window_stride=args.window_stride,
        dance_token_start=args.dance_token_start,
        dance_token_end=args.dance_token_start + args.dance_token_increment - 1,
        dancedata=args.dancedata, vqvae_ckpt_path=args.vqvae_checkpoint_path,
    )
    val_dataset = MusicDanceDataset(
        music_dir=args.music_dir, dance_dir=args.dance_dir,
        data_split_dir=args.data_split_dir, style_dir=args.style_dir,
        split="eval", music_length=args.music_length, dance_length=args.dance_length,
        max_samples=args.max_val_samples, verbose=is_main,
        window_stride=args.window_stride,
        dance_token_start=args.dance_token_start,
        dance_token_end=args.dance_token_start + args.dance_token_increment - 1,
        dancedata=args.dancedata, vqvae_ckpt_path=args.vqvae_checkpoint_path,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=variable_length_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=variable_length_collate_fn)

    # 模型
    if is_main:
        print(f"[rank0] building model ... (style_dim={args.style_embedding_dim}, n_bins={args.n_bins})")
    vocab_size = args.dance_token_start + args.dance_token_increment * args.dance_ranges_count + args.special_tokens_count
    model = Music2DanceLlamaModel(
        pretrained_model_name=args.pretrained_model_name,
        vocab_size=vocab_size, dance_token_start=args.dance_token_start,
        dance_token_increment=args.dance_token_increment,
        dance_ranges_count=args.dance_ranges_count,
        num_styles=len(GENRES), style_embedding_dim=args.style_embedding_dim,
        music_enc_dim=1024, dance_enc_dim=1024,
        llama_hidden_dim=2048, bridge_hidden_dim=2048, bridge_num_heads=8,
        dance_retrieval_cond_dim=264, music_len=args.music_length,
        retrieval_dance_len=384, codebooks=codebooks_arg,
        llama_config_path=args.llama_config_path,
    )

    # === optional warm-start from a previous checkpoint ===
    if getattr(args, 'resume_from_checkpoint', None):
        if is_main:
            print(f"[rank0] resuming from {args.resume_from_checkpoint}")
        ckpt = torch.load(args.resume_from_checkpoint, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        if ckpt and next(iter(ckpt.keys())).startswith('module.'):
            ckpt = {k.replace('module.', '', 1): v for k, v in ckpt.items()}
        ms = model.state_dict()
        filtered = {k: v for k, v in ckpt.items() if k in ms and ms[k].shape == v.shape}
        skipped_shape = [k for k in ckpt.keys() if k in ms and ms[k].shape != ckpt[k].shape]
        skipped_missing = [k for k in ckpt.keys() if k not in ms]
        if is_main:
            print(f"[rank0] partial resume: matched={len(filtered)}, shape_filtered={len(skipped_shape)}, missing_in_model={len(skipped_missing)}")
            for k in skipped_shape[:5]:
                print(f"  filtered (shape mismatch): {k}: model={ms[k].shape} ckpt={ckpt[k].shape}")
            print(f"  skipped (n_bins=4 only specialized_experts.2/3 etc): {len(skipped_missing)} keys")
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if is_main:
            print(f"[rank0] after load: missing in model={len(missing)} (will be re-init), unexpected={len(unexpected)}")

        # partial-init music_bridge.fusion_proj when input width differs from ckpt
        # (dance_bridge is an MLP and does not need fusion_proj)
        with torch.no_grad():
            for bridge_name in ['music_bridge']:
                if f'{bridge_name}.fusion_proj.weight' not in ckpt:
                    continue
                if f'{bridge_name}.fusion_proj.weight' not in model.state_dict():
                    continue
                ck_w = ckpt[f'{bridge_name}.fusion_proj.weight']
                ck_b = ckpt[f'{bridge_name}.fusion_proj.bias']
                target_w_shape = model.state_dict()[f'{bridge_name}.fusion_proj.weight'].shape
                w_subset = ck_w[:, :target_w_shape[1]].contiguous()
                model.state_dict()[f'{bridge_name}.fusion_proj.weight'].copy_(w_subset)
                model.state_dict()[f'{bridge_name}.fusion_proj.bias'].copy_(ck_b)
                if is_main:
                    print(f"  fusion_proj.{bridge_name}: partial-init from v4 first {target_w_shape[1]}/{ck_w.shape[1]} cols")

        # === warm-init dance_bridge MLP fc2 ===
        # 让 fresh-init MLP 在真实 motion features 输入下输出 std ≈ LLaMA dance embedding std
        # 这样 LLaMA 收到的 hidden 跟训练时见过的 dance token embedding 同 magnitude
        with torch.no_grad():
            emb_w = model.llama.model.embed_tokens.weight
            ds = args.dance_token_start
            de = ds + args.dance_token_increment
            target_std = emb_w[ds:de].std().item()

            # 用 codebook 中心当 sample input 测 raw output std(没有真实 batch 时这个最方便)
            try:
                cb0 = codebooks[0]  # (512, 1024) 第一个频段
                # 拼成 (288, in_dim) 模拟一个完整 dance 段的 features
                idx = torch.arange(288, device=cb0.device) % cb0.size(0)
                sample_in = cb0[idx].to(next(model.dance_bridge.parameters()).device)
                # 多个 sample 平均
                sample_in = sample_in + torch.randn_like(sample_in) * 0.5  # 加扰动近似不同 token
                raw_out = model.dance_bridge(sample_in)
                actual_std = raw_out.std().item()
            except Exception as e:
                # fallback: 直接用 random input
                rand_in = torch.randn(288, model.dance_bridge.fc1.in_features,
                                       device=next(model.dance_bridge.parameters()).device) * 2.0
                raw_out = model.dance_bridge(rand_in)
                actual_std = raw_out.std().item()

            scale = target_std / max(actual_std, 1e-6)
            model.dance_bridge.fc2.weight.data *= scale
            if model.dance_bridge.fc2.bias is not None:
                model.dance_bridge.fc2.bias.data *= scale  # bias 也跟着 scale
            # 验证
            with_scaled = model.dance_bridge(sample_in if 'sample_in' in dir() else rand_in)
            verified_std = with_scaled.std().item()
            if is_main:
                print(f"  dance_bridge MLP warm-init: target_std={target_std:.5f}  raw_std={actual_std:.5f}  scale={scale:.5f}  verified_std={verified_std:.5f}")

    # === Stage 1 起步:冻 LLaMA,只训 bridge + style + cond_proj + classifier ===
    # (Stage 2 切换在训练循环里手动 unfreeze)
    set_stage1_freeze(model)
    if is_main:
        t1, f1 = count_params(model)
        print(f"[rank0] Stage 1 init: trainable={t1:,} frozen={f1:,}")

    # 转精度
    if dtype != torch.float32:
        model = model.to(dtype)
    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    if is_main:
        trainable, frozen = count_params(model.module)
        print(f"[rank0] trainable={trainable:,}  frozen={frozen:,}")

    # 优化器
    no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
    grouped_params = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

    update_steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
    # 总 epoch = stage1 + stage2;num_train_epochs 仅作 fallback(若 stage1+2 都为 0)
    if args.stage1_epoch + args.stage2_epoch > 0:
        total_epochs = args.stage1_epoch + args.stage2_epoch
    else:
        total_epochs = args.num_train_epochs
    total_update_steps = update_steps_per_epoch * total_epochs
    scheduler = get_lr_scheduler(optimizer, total_update_steps, args.warmup_steps, args.lr_scheduler_type)

    scaler = torch.cuda.amp.GradScaler() if args.precision == "fp16" else None

    if is_main:
        print(f"[rank0] update_steps_per_epoch={update_steps_per_epoch} total_update_steps={total_update_steps}")
        print(f"[rank0] stage1_epoch={args.stage1_epoch} stage2_epoch={args.stage2_epoch} target_train_acc={args.target_train_acc}")

    # ============================ TRAIN ============================
    global_update_step = 0
    best_val_loss = float("inf")
    early_stop = 0
    current_stage = 1
    target_acc_reached = False

    for epoch in range(total_epochs):
        # === stage 切换 ===
        if epoch == args.stage1_epoch and current_stage == 1 and args.stage2_epoch > 0:
            current_stage = 2
            set_stage2_unfreeze(model.module, args.freeze_llama_layers)
            if is_main:
                t2, f2 = count_params(model.module)
                print(f"\n[rank0] *** Switching to Stage 2 *** trainable={t2:,} frozen={f2:,}")
            # 重建 optimizer 把新解冻的 LLaMA 参数加进去
            grouped_params = [
                {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                 "weight_decay": args.weight_decay},
                {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            stage2_lr = args.learning_rate2 if args.learning_rate2 is not None else args.learning_rate
            if is_main:
                print(f"[rank0] Stage 2 LR = {stage2_lr:.2e} (was Stage 1 {args.learning_rate:.2e})")
            optimizer = torch.optim.AdamW(grouped_params, lr=stage2_lr, betas=(0.9, 0.999), eps=1e-8)
            # scheduler 保持原步数计划(continue from current step)
            scheduler = get_lr_scheduler(optimizer, total_update_steps - global_update_step, max(0, args.warmup_steps - global_update_step), args.lr_scheduler_type)
        train_sampler.set_epoch(epoch)
        model.train()
        progress = tqdm(train_loader, desc=f"Stage{current_stage} Ep {epoch + 1}/{total_epochs}", disable=not is_main, position=0)

        running_loss = torch.zeros(1, device=device)
        running_correct = torch.zeros(1, device=device)
        running_total = torch.zeros(1, device=device)

        accum_step = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None: attention_mask = attention_mask.to(device)
            style_indices = batch["style_idx"].to(device)
            music_features = batch["music_features"].to(device)
            dance_retrieval_cond = batch["dance_retrieval_cond"].to(device)

            # autocast forward
            with autocast(enabled=use_amp, dtype=dtype):
                outputs = model(
                    input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                    music_features=music_features, style_indices=style_indices,
                    dance_retrieval_cond=dance_retrieval_cond, stage=2,
                )
                # 用 model 内部 outputs.loss(已对齐 LLaMA 实际 hidden 长度)
                # early_token_weight 暂不应用,等模型先学起来
                loss = outputs.loss / args.gradient_accumulation_steps

            with torch.no_grad():
                logits = outputs.logits
                flat_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
                flat_labels = labels[:, 1:].contiguous().view(-1)
                vmask = flat_labels != -100
                tot = vmask.sum()
                tk = _topk_correct(flat_logits[vmask].float(), flat_labels[vmask])
                running_loss += (loss.detach() * args.gradient_accumulation_steps)
                running_correct += tk[1]
                running_total += tot

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_step += 1
            if accum_step >= args.gradient_accumulation_steps:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_step = 0
                global_update_step += 1

                if is_main:
                    avg_loss = (running_loss.item() / max(1, batch_idx + 1))
                    train_acc = (running_correct / running_total.clamp(min=1)).item()
                    progress.set_postfix({
                        "loss": f"{loss.item() * args.gradient_accumulation_steps:.3f}",
                        "ema_loss": f"{avg_loss:.3f}",
                        "acc": f"{train_acc:.3f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })

                # ===== 验证 =====
                if global_update_step % args.save_steps == 0:
                    val_loss, val_acc = run_validation(model, val_loader, device, dtype, use_amp, vocab_size, args, world_size, is_main)
                    if is_main:
                        print(f"\n[step {global_update_step}] val_loss={val_loss:.4f}  val_top1={val_acc:.4f}")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            early_stop = 0
                            save_state_dict(os.path.join(args.out_dir, "best_model.pt"), model.module)
                            print(f"  ✓ saved best_model.pt (val_loss {val_loss:.4f})")
                        else:
                            early_stop += 1
                            print(f"  no improvement [{early_stop}/{args.early_stopping_patience}]")
                            if early_stop >= args.early_stopping_patience:
                                print(f"  ★ early stop triggered.")
                                cleanup_ddp(); return
                    model.train()

        # 用 all_reduce 聚合 epoch acc(防只看 rank-0)
        ep_correct = running_correct.clone(); ep_total = running_total.clone()
        if dist.is_initialized():
            dist.all_reduce(ep_correct); dist.all_reduce(ep_total)
        ep_acc_global = (ep_correct / ep_total.clamp(min=1)).item()

        if is_main:
            ep_loss = running_loss.item() / max(1, len(train_loader))
            print(f"\n[Stage{current_stage}] Epoch {epoch + 1}: train_loss={ep_loss:.4f}  train_top1(global)={ep_acc_global:.4f}")
            save_state_dict(os.path.join(args.out_dir, f"epoch_{epoch + 1}_stage{current_stage}.pt"), model.module)

        # === target_acc 仅做"至少达标"信息提示,不停训(继续推 acc 更高直到 epochs 跑完)===
        if current_stage == 2 and ep_acc_global >= args.target_train_acc and not target_acc_reached:
            target_acc_reached = True
            if is_main:
                print(f"\n★ Stage 2 target_train_acc {args.target_train_acc} reached at epoch {epoch + 1} (global={ep_acc_global:.4f}). Continuing training.")

    if is_main:
        print(f"\n=== Training done (target_acc_reached={target_acc_reached}, final stage={current_stage}) ===")
    cleanup_ddp()


def run_validation(model, val_loader, device, dtype, use_amp, vocab_size, args, world_size, is_main):
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_correct = torch.zeros(1, device=device)
    total_count = torch.zeros(1, device=device)
    n = torch.zeros(1, device=device)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="val", leave=False, disable=not is_main, position=1):
            input_ids = batch["input_ids"].to(device); labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None: attention_mask = attention_mask.to(device)
            style_indices = batch["style_idx"].to(device)
            music_features = batch["music_features"].to(device)
            dance_retrieval_cond = batch["dance_retrieval_cond"].to(device)
            with autocast(enabled=use_amp, dtype=dtype):
                out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                            music_features=music_features, style_indices=style_indices,
                            dance_retrieval_cond=dance_retrieval_cond, stage=2, infer=True)
            total_loss += out.loss.detach()
            logits = out.logits
            flat_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
            flat_labels = labels[:, 1:].contiguous().view(-1)
            vmask = flat_labels != -100
            tot = vmask.sum()
            tk = _topk_correct(flat_logits[vmask].float(), flat_labels[vmask])
            total_correct += tk[1]; total_count += tot; n += 1
    dist.all_reduce(total_loss); dist.all_reduce(total_correct); dist.all_reduce(total_count); dist.all_reduce(n)
    avg_loss = (total_loss / n.clamp(min=1)).item()
    acc = (total_correct / total_count.clamp(min=1)).item()
    return avg_loss, acc


def main():
    args = parse_args()
    print("Loading VQVAE codebooks ...")
    _, codebooks, _ = load_vqvae_model(checkpoint_path=args.vqvae_checkpoint_path,
                                       mean_path=args.mean_path, std_path=args.std_path)
    print("Spawning workers ...")
    mp.spawn(train_worker, args=(args.world_size, args, codebooks),
             nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
