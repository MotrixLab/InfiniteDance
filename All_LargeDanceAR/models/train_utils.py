import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import json
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb
from models.dancellama import Music2DanceLlamaModel, MusicDanceDataset, GenreBalancedDistributedSampler, variable_length_collate_fn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoConfig, LlamaForCausalLM, get_scheduler


def _topk_correct(logits_2d, labels_1d, ks=(1, 5, 10)):
    """Top-k correct counts for already-masked flat logits/labels.

    logits_2d: [N, V], labels_1d: [N]. Returns {k: 0-dim long tensor}.
    """
    device = logits_2d.device if logits_2d.numel() > 0 else labels_1d.device
    if labels_1d.numel() == 0:
        return {k: torch.zeros((), device=device, dtype=torch.long) for k in ks}
    max_k = min(max(ks), logits_2d.size(-1))
    _, topk_idx = logits_2d.topk(k=max_k, dim=-1)
    match = (topk_idx == labels_1d.unsqueeze(-1))
    out = {}
    for k in ks:
        k_eff = min(k, max_k)
        out[k] = match[:, :k_eff].any(dim=-1).sum()
    return out


def save_checkpoint(path, model, optimizer, lr_scheduler, epoch, global_step, 
                    best_val_loss, best_train_loss, stage, early_stopping_counter):
    """保存完整的训练状态，支持真正的续训"""
    checkpoint = {
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': float(best_val_loss) if isinstance(best_val_loss, torch.Tensor) else best_val_loss,
        'best_train_loss': float(best_train_loss) if isinstance(best_train_loss, torch.Tensor) else best_train_loss,
        'stage': stage,
        'early_stopping_counter': early_stopping_counter,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, device='cpu'):
    """加载完整的训练状态。支持 DDP 保存的 checkpoint（自动去掉 module. 前缀）。"""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # 训练用 DDP 保存时 key 带 "module."，加载到未包装的 model 需去掉
        if state_dict and next(iter(state_dict.keys())).startswith('module.'):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        return {
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'best_train_loss': checkpoint.get('best_train_loss', 1000),
            'stage': checkpoint.get('stage', 1),
            'early_stopping_counter': checkpoint.get('early_stopping_counter', 0),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
            'lr_scheduler_state_dict': checkpoint.get('lr_scheduler_state_dict'),
        }
    else:
        state_dict = checkpoint
        if state_dict and next(iter(state_dict.keys())).startswith('module.'):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        return {
            'epoch': 0,
            'global_step': 0,
            'best_val_loss': float('inf'),
            'best_train_loss': 1000,
            'stage': 1,
            'early_stopping_counter': 0,
            'optimizer_state_dict': None,
            'lr_scheduler_state_dict': None,
        }


def load_matching_weights(path, model, device='cpu'):
    """只加载 key 和 shape 都匹配的参数，适合结构有改动时做迁移初始化。"""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if state_dict and next(iter(state_dict.keys())).startswith('module.'):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model_state = model.state_dict()
    matched_state = {}
    skipped = []

    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append((key, "missing_in_model"))
            continue
        if model_state[key].shape != value.shape:
            skipped.append((key, f"shape_mismatch ckpt={tuple(value.shape)} model={tuple(model_state[key].shape)}"))
            continue
        matched_state[key] = value

    missing_after_load = sorted(set(model_state.keys()) - set(matched_state.keys()))
    model.load_state_dict(matched_state, strict=False)
    return {
        'matched_count': len(matched_state),
        'skipped': skipped,
        'missing_after_load': missing_after_load,
    }


def setup_ddp(rank, world_size, MASTER_PORT="17767"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Rank {rank} initialized on GPU {rank}")

def cleanup_ddp():
    dist.destroy_process_group()

def train_llama_music2dance(rank, world_size, args, codebooks_arg, MASTER_PORT):
    global codebooks
    codebooks = codebooks_arg

    setup_ddp(rank, world_size, MASTER_PORT=MASTER_PORT)
    device = torch.device(f"cuda:{rank}")
    is_main_process = (rank == 0)

    if is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)
        args_output_path_json = os.path.join(args.out_dir, "args.json")
        with open(args_output_path_json, 'w') as f:
            json.dump(vars(args), f, indent=4)
        print(f"Arguments saved to {args_output_path_json}")
        llama_config_path=args.llama_config_path
        config_output_path = os.path.join(args.out_dir, "config.json")
        try:
            with open(llama_config_path, 'r') as f:
                config_data = json.load(f)
            with open(config_output_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            print(f"Model configuration copied from {llama_config_path} to {config_output_path}")
        except FileNotFoundError:
            print(f"Error: Llama config file not found at {llama_config_path}")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {llama_config_path}")
        except Exception as e:
            print(f"An unexpected error occurred while processing config: {e}")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    tensorboard_writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tensorboard_logs")) if is_main_process else None

    if args.use_wandb and is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or os.path.basename(args.out_dir),
            config=vars(args)
        )

    if is_main_process:
        print("Loading training dataset...")
    train_dataset = MusicDanceDataset(
        music_dir=args.music_dir,
        dance_dir=args.dance_dir,
        data_split_dir=args.data_split_dir,
        style_dir=args.style_dir,
        split="train",
        music_length=args.music_length,
        dance_length=args.dance_length,
        max_samples=args.max_train_samples,
        dance_token_start=args.dance_token_start,
        dance_token_end=args.dance_token_start + args.dance_token_increment - 1,
        window_stride=args.window_stride,
        verbose=is_main_process,
        dancedata=args.dancedata,
        vqvae_ckpt_path=args.vqvae_checkpoint_path,
        cap_popular_windows=args.cap_popular_windows,
        popular_max_windows=args.popular_max_windows,
        retrieval_path=args.retrieval_path,
        p_partial_triplet=getattr(args, 'p_partial_triplet', 0.0),  # ★ NEW
    )

    if is_main_process:
        print("Loading validation dataset...")
    if args.dancedata == "finedance" or args.dancedata == "All":
        val_dataset = MusicDanceDataset(
            music_dir=args.music_dir,
            dance_dir=args.dance_dir,
            data_split_dir=args.data_split_dir,
            style_dir=args.style_dir,
            split="eval",
            music_length=args.music_length,
            dance_length=args.dance_length,
            max_samples=args.max_val_samples,
            dance_token_start=args.dance_token_start,
            dance_token_end=args.dance_token_start + args.dance_token_increment - 1,
            window_stride=args.window_stride,
            verbose=is_main_process,
            dancedata=args.dancedata,
            vqvae_ckpt_path=args.vqvae_checkpoint_path,
            cap_popular_windows=args.cap_popular_windows,
            popular_max_windows=args.popular_max_windows,
            retrieval_path=args.retrieval_path,
        )
    else:
        val_dataset = MusicDanceDataset(
            music_dir=args.music_dir,
            dance_dir=args.dance_dir,
            data_split_dir=args.data_split_dir,
            style_dir=args.style_dir,
            split="test",
            music_length=args.music_length,
            dance_length=args.dance_length,
            max_samples=args.max_val_samples,
            dance_token_start=args.dance_token_start,
            dance_token_end=args.dance_token_start + args.dance_token_increment - 1,
            window_stride=args.window_stride,
            verbose=is_main_process,
            dancedata=args.dancedata,
            vqvae_ckpt_path=args.vqvae_checkpoint_path,
            cap_popular_windows=args.cap_popular_windows,
            popular_max_windows=args.popular_max_windows,
            retrieval_path=args.retrieval_path,
        )

    train_sampler = GenreBalancedDistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        total_samples=36000
    )
    val_sampler = GenreBalancedDistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.seed,
        total_samples=7200
    )

    # 当 p_partial_triplet>0 时样本长度不等，需要 padding collate；否则用默认
    _collate_fn = variable_length_collate_fn if getattr(args, 'p_partial_triplet', 0.0) > 0 else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    vocab_size = args.dance_token_start + args.dance_token_increment + args.special_tokens_count
    label_smoothing = getattr(args, 'label_smoothing', 0.1)
    llama_dropout = getattr(args, 'llama_dropout', 0.1)
    cond_drop_prob = getattr(args, 'cond_drop_prob', 0.15)

    model = Music2DanceLlamaModel(
        pretrained_model_name="./models/Llama3.2-1B",
        vocab_size=vocab_size,
        dance_token_start=args.dance_token_start,
        dance_token_increment=args.dance_token_increment,
        dance_ranges_count=args.dance_ranges_count,
        style_embedding_dim=args.style_embedding_dim,
        music_enc_dim=1024,
        dance_enc_dim=1024,
        llama_hidden_dim=2048,
        bridge_hidden_dim=2048,
        bridge_num_heads=8,
        dance_retrieval_cond_dim=264,
        music_len=320,
        retrieval_dance_len=384,
        codebooks=codebooks,
        llama_config_path=args.llama_config_path,
        n_bins=args.n_bins,
        label_smoothing=label_smoothing,
        llama_dropout=llama_dropout,
        cond_drop_prob=cond_drop_prob,
        use_music_cross_attn=getattr(args, 'use_music_cross_attn', False),  # ★ NEW
    )

    resume_state = None
    if args.resume_from_checkpoint and args.init_from_checkpoint:
        raise ValueError("Use only one of --resume_from_checkpoint or --init_from_checkpoint")

    if args.init_from_checkpoint:
        if is_main_process:
            print(f"Initializing matching weights from {args.init_from_checkpoint}")
        if getattr(args, 'load_matching_only', False):
            init_info = load_matching_weights(args.init_from_checkpoint, model, device=device)
            if is_main_process:
                print(f"  Loaded {init_info['matched_count']} matching tensors")
                print(f"  Skipped {len(init_info['skipped'])} tensors due to missing keys or shape mismatch")
                for key, reason in init_info['skipped'][:12]:
                    print(f"    skip: {key} -> {reason}")
                if len(init_info['skipped']) > 12:
                    print(f"    ... and {len(init_info['skipped']) - 12} more")
        else:
            load_checkpoint(args.init_from_checkpoint, model, device=device)
            if is_main_process:
                print("  Loaded checkpoint weights with strict=False (optimizer/epoch not resumed)")
    elif args.resume_from_checkpoint:
        if is_main_process:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        resume_state = load_checkpoint(args.resume_from_checkpoint, model, device=device)
        if is_main_process:
            print(f"  Resumed from stage: {resume_state['stage']}, epoch: {resume_state['epoch']}, global_step: {resume_state['global_step']}")

    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters (before Stage 1): {trainable_params:,}")

    start_stage = resume_state['stage'] if resume_state else 1
    start_epoch = resume_state['epoch'] if resume_state else 0
    start_global_step = resume_state['global_step'] if resume_state else 0
    best_val_loss = resume_state['best_val_loss'] if resume_state else float('inf')
    best_train_loss = resume_state['best_train_loss'] if resume_state else 1000.0
    early_stopping_counter = resume_state['early_stopping_counter'] if resume_state else 0

    early_stopping_patience = getattr(args, 'early_stopping_patience', 100)
    early_stopping_min_delta = getattr(args, 'early_stopping_min_delta', 0.0)
    num_update_steps_per_epoch = len(train_dataloader)

    # ==================== Stage 1 ====================
    if start_stage <= 1:
        if is_main_process:
            print("Starting Stage 1: Fine-tuning style_embedding, music_bridge, dance_bridge, cond_projection, and style_classifier")

        for param in model.module.parameters():
            param.requires_grad = False

        for param in model.module.style_embedding.parameters():
            param.requires_grad = True
        for param in model.module.music_bridge.parameters():
            param.requires_grad = True
        for param in model.module.dance_bridge.parameters():
            param.requires_grad = True
        for param in model.module.cond_projection.parameters():
            param.requires_grad = True
        for param in model.module.style_classifier.parameters():
            param.requires_grad = True
        for param in model.module.llama.parameters():
            param.requires_grad = False

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if is_main_process:
            print(f"Trainable parameters (Stage 1): {trainable_params:,}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate1,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon
        )

        max_train_steps = args.stage1_epoch * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=max_train_steps
        )

        if resume_state and start_stage == 1:
            if resume_state['optimizer_state_dict']:
                try:
                    optimizer.load_state_dict(resume_state['optimizer_state_dict'])
                    if is_main_process:
                        print("  Optimizer state restored for Stage 1")
                except Exception as e:
                    if is_main_process:
                        print(f"  Warning: Could not restore optimizer state: {e}")
            if resume_state['lr_scheduler_state_dict']:
                try:
                    lr_scheduler.load_state_dict(resume_state['lr_scheduler_state_dict'])
                    if is_main_process:
                        print("  LR scheduler state restored for Stage 1")
                except Exception as e:
                    if is_main_process:
                        print(f"  Warning: Could not restore lr_scheduler state: {e}")

        if is_main_process:
            print(f"Training parameters (Stage 1):")
            print(f"  Batch size: {args.batch_size}")
            print(f"  Learning rate: {args.learning_rate1}")
            print(f"  Training epochs: {args.stage1_epoch}")
            print(f"  Training steps: {max_train_steps}")
            print(f"  Device: {device}")
            print(f"  World size: {world_size}")
            print(f"  Dance token range: starting at {args.dance_token_start} with {args.dance_token_increment} tokens")
            print(f"  Total vocabulary size: {vocab_size}")

        global_step = start_global_step if start_stage == 1 else 0
        stage1_start_epoch = start_epoch if start_stage == 1 else 0

        for epoch in range(stage1_start_epoch, args.stage1_epoch):
            model.train()
            train_sampler.set_epoch(epoch)
            total_train_loss = torch.tensor(0.0).to(device)
            total_train_correct = torch.tensor(0.0).to(device)
            total_train_predictions = torch.tensor(0.0).to(device)
            num_batches = 0

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Stage 1 Epoch {epoch+1}/{args.stage1_epoch}",
                disable=not is_main_process
            )

            total_train_correct_top5 = torch.tensor(0.0).to(device)
            total_train_correct_top10 = torch.tensor(0.0).to(device)

            for batch in progress_bar:
                step_start = time.perf_counter()
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
                style_indices = batch["style_idx"].to(device)
                music_features = batch["music_features"].to(device)
                dance_retrieval_cond = batch["dance_retrieval_cond"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    music_features=music_features,
                    style_indices=style_indices,
                    dance_retrieval_cond=dance_retrieval_cond,
                    stage=1
                )

                loss = outputs.loss

                with torch.no_grad():
                    logits = outputs.logits
                    flat_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
                    flat_labels = labels[:, 1:].contiguous().view(-1)
                    valid_mask = flat_labels != -100
                    total = valid_mask.sum()
                    topk_correct = _topk_correct(flat_logits[valid_mask], flat_labels[valid_mask])

                total_train_loss += loss.detach()
                total_train_correct += topk_correct[1]
                total_train_correct_top5 += topk_correct[5]
                total_train_correct_top10 += topk_correct[10]
                total_train_predictions += total
                num_batches += 1

                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if is_main_process:
                    step_time = time.perf_counter() - step_start
                    samples_per_sec = (args.batch_size * world_size) / step_time if step_time > 0 else 0.0
                    train_accuracy = (total_train_correct / total_train_predictions).item() if total_train_predictions > 0 else 0.0
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{train_accuracy:.4f}",
                        "step_s": f"{step_time:.3f}",
                        "samples_s": f"{samples_per_sec:.1f}"
                    })
                    if loss.item() < best_train_loss:
                        best_train_loss = loss.item()
                        save_checkpoint(
                            os.path.join(args.out_dir, "best_model_stage1.pt"),
                            model, optimizer, lr_scheduler, epoch, global_step,
                            best_val_loss, best_train_loss, 1, early_stopping_counter
                        )

                if global_step % args.save_steps == 0:
                    model.eval()
                    total_val_loss = torch.tensor(0.0).to(device)
                    total_val_correct = torch.tensor(0.0).to(device)
                    total_val_correct_top5 = torch.tensor(0.0).to(device)
                    total_val_correct_top10 = torch.tensor(0.0).to(device)
                    total_val_predictions = torch.tensor(0).to(device)

                    with torch.no_grad():
                        for val_batch in tqdm(
                            val_dataloader,
                            desc="Validating",
                            disable=not is_main_process
                        ):
                            val_input_ids = val_batch["input_ids"].to(device)
                            val_labels = val_batch["labels"].to(device)
                            val_attention_mask = val_batch["attention_mask"].to(device) if "attention_mask" in val_batch else None
                            val_style_indices = val_batch["style_idx"].to(device)
                            val_music_features = val_batch["music_features"].to(device)
                            val_dance_retrieval_cond = val_batch["dance_retrieval_cond"].to(device)

                            val_outputs = model(
                                input_ids=val_input_ids,
                                labels=val_labels,
                                attention_mask=val_attention_mask,
                                music_features=val_music_features,
                                style_indices=val_style_indices,
                                dance_retrieval_cond=val_dance_retrieval_cond,
                                stage=1
                            )

                            val_loss = val_outputs.loss

                            val_flat_logits = val_outputs.logits[:, :-1, :].contiguous().view(-1, vocab_size)
                            val_flat_labels = val_labels[:, 1:].contiguous().view(-1)
                            val_valid_mask = val_flat_labels != -100
                            val_total = val_valid_mask.sum()
                            val_topk = _topk_correct(val_flat_logits[val_valid_mask], val_flat_labels[val_valid_mask])

                            total_val_loss += val_loss
                            total_val_correct += val_topk[1]
                            total_val_correct_top5 += val_topk[5]
                            total_val_correct_top10 += val_topk[10]
                            total_val_predictions += val_total

                    agg_train_loss = total_train_loss.clone()
                    agg_train_correct = total_train_correct.clone()
                    agg_train_correct_top5 = total_train_correct_top5.clone()
                    agg_train_correct_top10 = total_train_correct_top10.clone()
                    agg_train_predictions = total_train_predictions.clone()
                    dist.all_reduce(agg_train_loss)
                    dist.all_reduce(agg_train_correct)
                    dist.all_reduce(agg_train_correct_top5)
                    dist.all_reduce(agg_train_correct_top10)
                    dist.all_reduce(agg_train_predictions)
                    dist.all_reduce(total_val_loss)
                    dist.all_reduce(total_val_correct)
                    dist.all_reduce(total_val_correct_top5)
                    dist.all_reduce(total_val_correct_top10)
                    dist.all_reduce(total_val_predictions)

                    if is_main_process:
                        avg_train_loss = agg_train_loss / (num_batches * world_size)
                        train_accuracy = agg_train_correct / agg_train_predictions if agg_train_predictions > 0 else 0
                        train_top5 = agg_train_correct_top5 / agg_train_predictions if agg_train_predictions > 0 else 0
                        train_top10 = agg_train_correct_top10 / agg_train_predictions if agg_train_predictions > 0 else 0
                        avg_val_loss = total_val_loss / (len(val_dataloader) * world_size)
                        val_accuracy = total_val_correct / total_val_predictions if total_val_predictions > 0 else 0
                        val_top5 = total_val_correct_top5 / total_val_predictions if total_val_predictions > 0 else 0
                        val_top10 = total_val_correct_top10 / total_val_predictions if total_val_predictions > 0 else 0
                        val_perplexity = torch.exp(avg_val_loss.clamp(max=20))

                        tensorboard_writer.add_scalar("train/loss", avg_train_loss.item(), global_step)
                        tensorboard_writer.add_scalar("train/accuracy", train_accuracy.item(), global_step)
                        tensorboard_writer.add_scalar("train/top5_acc", train_top5.item(), global_step)
                        tensorboard_writer.add_scalar("train/top10_acc", train_top10.item(), global_step)
                        tensorboard_writer.add_scalar("val/loss", avg_val_loss.item(), global_step)
                        tensorboard_writer.add_scalar("val/accuracy", val_accuracy.item(), global_step)
                        tensorboard_writer.add_scalar("val/top5_acc", val_top5.item(), global_step)
                        tensorboard_writer.add_scalar("val/top10_acc", val_top10.item(), global_step)
                        tensorboard_writer.add_scalar("val/perplexity", val_perplexity.item(), global_step)
                        tensorboard_writer.add_scalar("train/learning_rate1", lr_scheduler.get_last_lr()[0], global_step)

                        if args.use_wandb:
                            wandb.log({
                                "train/loss": avg_train_loss.item(),
                                "train/accuracy": train_accuracy.item(),
                                "train/top5_acc": train_top5.item(),
                                "train/top10_acc": train_top10.item(),
                                "val/loss": avg_val_loss.item(),
                                "val/accuracy": val_accuracy.item(),
                                "val/top5_acc": val_top5.item(),
                                "val/top10_acc": val_top10.item(),
                                "val/perplexity": val_perplexity.item(),
                                "train/learning_rate1": lr_scheduler.get_last_lr()[0],
                                "train/global_step": global_step,
                            })

                        print(f"Stage 1 Step {global_step}: val_loss={avg_val_loss.item():.4f}, "
                              f"top1={val_accuracy.item():.4f}, top5={val_top5.item():.4f}, "
                              f"top10={val_top10.item():.4f}, ppl={val_perplexity.item():.2f}")

                        if avg_val_loss < best_val_loss - early_stopping_min_delta:
                            best_val_loss = avg_val_loss.item()
                            early_stopping_counter = 0
                            save_checkpoint(
                                os.path.join(args.out_dir, "best_model_stage1.pt"),
                                model, optimizer, lr_scheduler, epoch, global_step,
                                best_val_loss, best_train_loss, 1, early_stopping_counter
                            )
                            print(f"Saved best model (Stage 1)")
                        else:
                            early_stopping_counter += 1
                            print(f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

                early_stop_flag = torch.tensor([0], dtype=torch.int32, device=device)
                if is_main_process and early_stopping_counter >= early_stopping_patience:
                    early_stop_flag[0] = 1
                dist.broadcast(early_stop_flag, src=0)

                if early_stop_flag.item():
                    if is_main_process:
                        print(f"Early stopping triggered after {early_stopping_counter} validations without improvement.")
                        save_checkpoint(
                            os.path.join(args.out_dir, "early_stopped_model_stage1.pt"),
                            model, optimizer, lr_scheduler, epoch, global_step,
                            best_val_loss, best_train_loss, 1, early_stopping_counter
                        )
                        print(f"Early stopped model (Stage 1) saved")
                        tensorboard_writer.close()
                        if args.use_wandb:
                            wandb.finish()
                    cleanup_ddp()
                    return

                model.train()
                dist.barrier()

            early_stop_flag = torch.tensor([early_stopping_counter >= early_stopping_patience], dtype=torch.int32, device=device)
            dist.broadcast(early_stop_flag, src=0)
            if early_stop_flag.item():
                break

            agg_train_loss = total_train_loss.clone()
            agg_train_correct = total_train_correct.clone()
            agg_train_predictions = total_train_predictions.clone()
            dist.all_reduce(agg_train_loss)
            dist.all_reduce(agg_train_correct)
            dist.all_reduce(agg_train_predictions)

            if is_main_process:
                avg_train_loss = agg_train_loss / (num_batches * world_size)
                train_accuracy = agg_train_correct / agg_train_predictions if agg_train_predictions > 0 else 0
                print(f"Stage 1 Epoch {epoch+1}: Average Training Loss: {avg_train_loss.item():.4f}, Training Accuracy: {train_accuracy.item():.4f}")

                tensorboard_writer.add_scalar("epoch/train_loss", avg_train_loss.item(), epoch)
                tensorboard_writer.add_scalar("epoch/train_accuracy", train_accuracy.item(), epoch)

                save_checkpoint(
                    os.path.join(args.out_dir, f"model_epoch_{epoch+1}_stage1.pt"),
                    model, optimizer, lr_scheduler, epoch + 1, global_step,
                    best_val_loss, best_train_loss, 1, early_stopping_counter
                )
    else:
        if is_main_process:
            print("Skipping Stage 1 (already completed based on checkpoint)")

    # ==================== Stage 2 ====================
    if is_main_process:
        print("Starting Stage 2: Partial LLaMA fine-tuning (freeze bottom layers)")

    for param in model.module.parameters():
        param.requires_grad = True

    # 冻结 LLaMA 底层：只微调顶部 N 层 + lm_head，大幅减少可训练参数
    freeze_llama_layers = getattr(args, 'freeze_llama_layers', 12)
    num_llama_layers = len(model.module.llama.model.layers)
    num_freeze = min(freeze_llama_layers, num_llama_layers)

    if num_freeze > 0:
        model.module.llama.model.embed_tokens.weight.requires_grad = False
        for i in range(num_freeze):
            for param in model.module.llama.model.layers[i].parameters():
                param.requires_grad = False

    if is_main_process:
        num_trainable_llama = num_llama_layers - num_freeze
        print(f"  LLaMA layers: {num_llama_layers} total, frozen bottom {num_freeze}, trainable top {num_trainable_llama}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters()) - trainable_params
    if is_main_process:
        print(f"Trainable parameters (Stage 2): {trainable_params:,}")
        print(f"Frozen parameters (Stage 2): {frozen_params:,}")

    trainable_llama_params = [p for p in model.module.llama.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": trainable_llama_params, "lr": args.learning_rate2},
        {"params": model.module.music_bridge.parameters(), "lr": args.learning_rate2 * 0.8},
        {"params": model.module.dance_bridge.parameters(), "lr": args.learning_rate2 * 0.8},
        {"params": model.module.style_embedding.parameters(), "lr": args.learning_rate2 * 1.2},
        {"params": model.module.cond_projection.parameters(), "lr": args.learning_rate2 * 0.8},
        {"params": model.module.style_classifier.parameters(), "lr": args.learning_rate2 * 1.2},
    ], weight_decay=args.weight_decay, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon)

    max_train_steps = args.stage2_epoch * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps
    )

    if resume_state and start_stage == 2:
        if resume_state['optimizer_state_dict']:
            try:
                optimizer.load_state_dict(resume_state['optimizer_state_dict'])
                if is_main_process:
                    print("  Optimizer state restored for Stage 2")
            except Exception as e:
                if is_main_process:
                    print(f"  Warning: Could not restore optimizer state: {e}")
        if resume_state['lr_scheduler_state_dict']:
            try:
                lr_scheduler.load_state_dict(resume_state['lr_scheduler_state_dict'])
                if is_main_process:
                    print("  LR scheduler state restored for Stage 2")
            except Exception as e:
                if is_main_process:
                    print(f"  Warning: Could not restore lr_scheduler state: {e}")

    if is_main_process:
        print(f"Training parameters (Stage 2):")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate (llama): {args.learning_rate2}")
        print(f"  Learning rate (music_bridge, dance_bridge, cond_projection): {args.learning_rate2 * 0.8}")
        print(f"  Learning rate (style_embedding, style_classifier): {args.learning_rate2 * 1.2}")
        print(f"  Training epochs: {args.stage2_epoch}")
        print(f"  Training steps: {max_train_steps}")
        print(f"  Device: {device}")
        print(f"  World size: {world_size}")
        print(f"  Dance token range: starting at {args.dance_token_start} with {args.dance_token_increment} tokens")
        print(f"  Total vocabulary size: {vocab_size}")

    if start_stage == 2:
        global_step = start_global_step
        stage2_start_epoch = start_epoch
    else:
        global_step = 0
        stage2_start_epoch = 0
        best_val_loss = float('inf')
        best_train_loss = 1000.0
        early_stopping_counter = 0

    for epoch in range(stage2_start_epoch, args.stage2_epoch):
        model.train()
        train_sampler.set_epoch(epoch)
        total_train_loss = torch.tensor(0.0).to(device)
        total_train_correct = torch.tensor(0.0).to(device)
        total_train_predictions = torch.tensor(0.0).to(device)
        num_batches = 0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Stage 2 Epoch {epoch+1}/{args.stage2_epoch}",
            disable=not is_main_process
        )

        total_train_correct_top5 = torch.tensor(0.0).to(device)
        total_train_correct_top10 = torch.tensor(0.0).to(device)

        for batch in progress_bar:
            step_start = time.perf_counter()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
            style_indices = batch["style_idx"].to(device)
            music_features = batch["music_features"].to(device)
            dance_retrieval_cond = batch["dance_retrieval_cond"].to(device)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                music_features=music_features,
                style_indices=style_indices,
                dance_retrieval_cond=dance_retrieval_cond,
                stage=2
            )

            loss = outputs.loss

            with torch.no_grad():
                logits = outputs.logits
                flat_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
                flat_labels = labels[:, 1:].contiguous().view(-1)
                valid_mask = flat_labels != -100
                total = valid_mask.sum()
                topk_correct = _topk_correct(flat_logits[valid_mask], flat_labels[valid_mask])

            total_train_loss += loss.detach()
            total_train_correct += topk_correct[1]
            total_train_correct_top5 += topk_correct[5]
            total_train_correct_top10 += topk_correct[10]
            total_train_predictions += total
            num_batches += 1

            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    args.max_grad_norm
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if is_main_process:
                step_time = time.perf_counter() - step_start
                samples_per_sec = (args.batch_size * world_size) / step_time if step_time > 0 else 0.0
                local_acc = (total_train_correct / total_train_predictions).item() if total_train_predictions > 0 else 0.0
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{local_acc:.4f}",
                    "step_s": f"{step_time:.3f}",
                    "samples_s": f"{samples_per_sec:.1f}"
                })
                if loss.item() < best_train_loss:
                    best_train_loss = loss.item()
                    save_checkpoint(
                        os.path.join(args.out_dir, "best_model_stage2.pt"),
                        model, optimizer, lr_scheduler, epoch, global_step,
                        best_val_loss, best_train_loss, 2, early_stopping_counter
                    )

            if global_step % args.save_steps == 0:
                model.eval()
                total_val_loss = torch.tensor(0.0).to(device)
                total_val_correct = torch.tensor(0.0).to(device)
                total_val_correct_top5 = torch.tensor(0.0).to(device)
                total_val_correct_top10 = torch.tensor(0.0).to(device)
                total_val_predictions = torch.tensor(0).to(device)

                with torch.no_grad():
                    for val_batch in tqdm(
                        val_dataloader,
                        desc="Validating",
                        disable=not is_main_process
                    ):
                        val_input_ids = val_batch["input_ids"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        val_attention_mask = val_batch["attention_mask"].to(device) if "attention_mask" in val_batch else None
                        val_style_indices = val_batch["style_idx"].to(device)
                        val_music_features = val_batch["music_features"].to(device)
                        val_dance_retrieval_cond = val_batch["dance_retrieval_cond"].to(device)

                        val_outputs = model(
                            input_ids=val_input_ids,
                            labels=val_labels,
                            attention_mask=val_attention_mask,
                            music_features=val_music_features,
                            style_indices=val_style_indices,
                            dance_retrieval_cond=val_dance_retrieval_cond,
                            stage=2
                        )

                        val_loss = val_outputs.loss

                        val_flat_logits = val_outputs.logits[:, :-1, :].contiguous().view(-1, vocab_size)
                        val_flat_labels = val_labels[:, 1:].contiguous().view(-1)
                        val_valid_mask = val_flat_labels != -100
                        val_total = val_valid_mask.sum()
                        val_topk = _topk_correct(val_flat_logits[val_valid_mask], val_flat_labels[val_valid_mask])

                        total_val_loss += val_loss
                        total_val_correct += val_topk[1]
                        total_val_correct_top5 += val_topk[5]
                        total_val_correct_top10 += val_topk[10]
                        total_val_predictions += val_total

                agg_train_loss = total_train_loss.clone()
                agg_train_correct = total_train_correct.clone()
                agg_train_correct_top5 = total_train_correct_top5.clone()
                agg_train_correct_top10 = total_train_correct_top10.clone()
                agg_train_predictions = total_train_predictions.clone()
                dist.all_reduce(agg_train_loss)
                dist.all_reduce(agg_train_correct)
                dist.all_reduce(agg_train_correct_top5)
                dist.all_reduce(agg_train_correct_top10)
                dist.all_reduce(agg_train_predictions)
                dist.all_reduce(total_val_loss)
                dist.all_reduce(total_val_correct)
                dist.all_reduce(total_val_correct_top5)
                dist.all_reduce(total_val_correct_top10)
                dist.all_reduce(total_val_predictions)

                if is_main_process:
                    avg_train_loss = agg_train_loss / (num_batches * world_size)
                    agg_train_predictions_value = agg_train_predictions.item()
                    train_accuracy = agg_train_correct / agg_train_predictions if agg_train_predictions_value > 0 else 0
                    train_top5 = agg_train_correct_top5 / agg_train_predictions if agg_train_predictions_value > 0 else 0
                    train_top10 = agg_train_correct_top10 / agg_train_predictions if agg_train_predictions_value > 0 else 0
                    avg_val_loss = total_val_loss / (len(val_dataloader) * world_size)
                    total_val_predictions_value = total_val_predictions.item()
                    val_accuracy = total_val_correct / total_val_predictions if total_val_predictions_value > 0 else 0
                    val_top5 = total_val_correct_top5 / total_val_predictions if total_val_predictions_value > 0 else 0
                    val_top10 = total_val_correct_top10 / total_val_predictions if total_val_predictions_value > 0 else 0
                    val_perplexity = torch.exp(avg_val_loss.clamp(max=20))

                    tensorboard_writer.add_scalar("train/loss", avg_train_loss.item(), global_step)
                    tensorboard_writer.add_scalar("train/accuracy", train_accuracy.item(), global_step)
                    tensorboard_writer.add_scalar("train/top5_acc", train_top5.item(), global_step)
                    tensorboard_writer.add_scalar("train/top10_acc", train_top10.item(), global_step)
                    tensorboard_writer.add_scalar("val/loss", avg_val_loss.item(), global_step)
                    tensorboard_writer.add_scalar("val/accuracy", val_accuracy.item(), global_step)
                    tensorboard_writer.add_scalar("val/top5_acc", val_top5.item(), global_step)
                    tensorboard_writer.add_scalar("val/top10_acc", val_top10.item(), global_step)
                    tensorboard_writer.add_scalar("val/perplexity", val_perplexity.item(), global_step)
                    tensorboard_writer.add_scalar("train/learning_rate2", lr_scheduler.get_last_lr()[0], global_step)

                    if args.use_wandb:
                        wandb.log({
                            "train/loss": avg_train_loss.item(),
                            "train/accuracy": train_accuracy.item(),
                            "train/top5_acc": train_top5.item(),
                            "train/top10_acc": train_top10.item(),
                            "val/loss": avg_val_loss.item(),
                            "val/accuracy": val_accuracy.item(),
                            "val/top5_acc": val_top5.item(),
                            "val/top10_acc": val_top10.item(),
                            "val/perplexity": val_perplexity.item(),
                            "train/learning_rate2": lr_scheduler.get_last_lr()[0],
                            "train/global_step": global_step,
                        })

                    print(f"Stage 2 Step {global_step}: val_loss={avg_val_loss.item():.4f}, "
                          f"top1={val_accuracy.item():.4f}, top5={val_top5.item():.4f}, "
                          f"top10={val_top10.item():.4f}, ppl={val_perplexity.item():.2f}")

                    if avg_val_loss < best_val_loss - early_stopping_min_delta:
                        best_val_loss = avg_val_loss.item()
                        early_stopping_counter = 0
                        save_checkpoint(
                            os.path.join(args.out_dir, "best_model_stage2.pt"),
                            model, optimizer, lr_scheduler, epoch, global_step,
                            best_val_loss, best_train_loss, 2, early_stopping_counter
                        )
                        print(f"Saved best model (Stage 2)")
                    else:
                        early_stopping_counter += 1
                        print(f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

                early_stop_flag = torch.tensor([0], dtype=torch.int32, device=device)
                if is_main_process and early_stopping_counter >= early_stopping_patience:
                    early_stop_flag[0] = 1
                dist.broadcast(early_stop_flag, src=0)

                if early_stop_flag.item():
                    if is_main_process:
                        print(f"Early stopping triggered after {early_stopping_counter} validations without improvement.")
                        save_checkpoint(
                            os.path.join(args.out_dir, "early_stopped_model_stage2.pt"),
                            model, optimizer, lr_scheduler, epoch, global_step,
                            best_val_loss, best_train_loss, 2, early_stopping_counter
                        )
                        print(f"Early stopped model (Stage 2) saved")
                        tensorboard_writer.close()
                        if args.use_wandb:
                            wandb.finish()
                    cleanup_ddp()
                    return

                model.train()
                dist.barrier()

        early_stop_flag = torch.tensor([early_stopping_counter >= early_stopping_patience], dtype=torch.int32, device=device)
        dist.broadcast(early_stop_flag, src=0)
        if early_stop_flag.item():
            break

        agg_train_loss = total_train_loss.clone()
        agg_train_correct = total_train_correct.clone()
        agg_train_predictions = total_train_predictions.clone()
        dist.all_reduce(agg_train_loss)
        dist.all_reduce(agg_train_correct)
        dist.all_reduce(agg_train_predictions)

        if is_main_process:
            agg_train_predictions_value = agg_train_predictions.item()
            avg_train_loss = agg_train_loss / (num_batches * world_size)
            train_accuracy = agg_train_correct / agg_train_predictions if agg_train_predictions_value > 0 else 0
            print(f"Stage 2 Epoch {epoch+1}: Average Training Loss: {avg_train_loss.item():.4f}, Training Accuracy: {train_accuracy.item():.4f}")
            
            tensorboard_writer.add_scalar("epoch/train_loss", avg_train_loss.item(), epoch)
            tensorboard_writer.add_scalar("epoch/train_accuracy", train_accuracy.item(), epoch)
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    os.path.join(args.out_dir, f"model_epoch_{epoch+1}_stage2.pt"),
                    model, optimizer, lr_scheduler, epoch + 1, global_step,
                    best_val_loss, best_train_loss, 2, early_stopping_counter
                )

    if is_main_process:
        save_checkpoint(
            os.path.join(args.out_dir, "final_model.pt"),
            model, optimizer, lr_scheduler, args.stage2_epoch, global_step,
            best_val_loss, best_train_loss, 2, early_stopping_counter
        )
        print(f"Training completed. Final model saved to {os.path.join(args.out_dir, 'final_model.pt')}")

        tensorboard_writer.close()
        if args.use_wandb:
            wandb.finish()

    cleanup_ddp()
