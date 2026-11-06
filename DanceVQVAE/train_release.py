"""Portable RVQ retraining using the released architecture and motion files."""
import argparse
import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from DanceVQVAE.models.vqvae import RVQVAE
from DanceVQVAE.release_data import ReleaseMotionDataset
from DanceVQVAE.utils.losses import ReConsLoss

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'InfiniteDanceData')
    parser.add_argument('--model-config', type=Path,
                        default=ROOT / 'All_LargeDanceAR/models/checkpoints/args.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=300000)
    parser.add_argument('--validate-every', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=123)
    options = parser.parse_args()
    config = Namespace(**json.loads(options.model_config.read_text()))
    if (config.nb_code, config.code_dim, config.num_quantizers) != (512, 1024, 3):
        raise ValueError('This entry trains the released 512-code, 1024-d, three-layer codec')
    if config.vel_decoder or config.rot_decoder:
        raise ValueError('This entry trains the base codec; auxiliary decoder tuning is separate')
    if min(options.batch_size, options.max_steps, options.validate_every) <= 0:
        raise ValueError('Batch size, steps, and validation interval must be positive')
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    root = options.data_root.resolve()
    train_file = root / 'partition/vq_release_train.txt'
    val_file = root / 'partition/vq_release_val.txt'
    train_ids = set(train_file.read_text().splitlines())
    val_ids = set(val_file.read_text().splitlines())
    test_ids = set((root / 'partition/test_eval861.txt').read_text().splitlines())
    if train_ids & val_ids or (train_ids | val_ids) & test_ids:
        raise ValueError('RVQ train/validation/canonical-test identities must be disjoint')
    motion_dir = root / 'dance/alldata_new_joint_vecs264_ft_balanced'
    meta_dir = root / 'dance/vqvae_train_20260522/meta'
    config.data_root, config.meta_dir = str(motion_dir), str(meta_dir)
    config.split_file_train, config.split_file_eval = str(train_file), str(val_file)
    config.out_dir, config.seed = str(options.output.resolve()), options.seed
    config.batch_size, config.total_iter = options.batch_size, options.max_steps
    train = ReleaseMotionDataset(motion_dir, train_file, meta_dir, config.window_size, True)
    validation = ReleaseMotionDataset(motion_dir, val_file, meta_dir, config.window_size, False)
    if len(train) < options.batch_size:
        raise ValueError('Training split contains fewer clips than one full batch')
    loader = DataLoader(train, batch_size=options.batch_size, shuffle=True, drop_last=True,
                        num_workers=options.workers)
    val_loader = DataLoader(validation, batch_size=options.batch_size, shuffle=False,
                            num_workers=options.workers)
    model = RVQVAE(config, 264, config.nb_code, config.code_dim, config.output_emb_width,
        config.down_t, config.stride_t, config.width, config.depth,
        config.dilation_growth_rate, config.vq_act, config.vq_norm).to(options.device)
    criterion = ReConsLoss(config, config.recons_loss, config.nb_joints, config.w_motion,
        config.w_joints, config.w_vel, config.w_acc, config.w_foot)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=config.lr, betas=(0.9, 0.99), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_scheduler, gamma=config.gamma)
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / 'args.json').write_text(json.dumps(vars(config), indent=2))
    (options.output / 'run.json').write_text(json.dumps(vars(options), default=str, indent=2))
    step, best = 0, float('inf')
    while step < options.max_steps:
        for motion in loader:
            model.train()
            motion = motion.to(options.device)
            prediction, _, _, commitment, _, _ = model(motion)
            reconstruction = criterion.forward_all(prediction, motion)[0]
            if step < config.warm_up_iter:
                for group in optimizer.param_groups:
                    group['lr'] = config.lr * (step + 1) / max(config.warm_up_iter, 1)
                loss = criterion(prediction, motion) + config.commit * commitment
            else:
                loss = reconstruction + 0.05 * commitment
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite RVQ loss at step %d' % step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step >= config.warm_up_iter:
                scheduler.step()
            step += 1
            if step == 1 or step % 100 == 0:
                print(json.dumps({'step': step, 'loss': loss.item()}), flush=True)
            if step % options.validate_every == 0 or step == options.max_steps:
                model.eval()
                total, count = 0.0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(options.device)
                        predicted, _, _, commitment, _, _ = model(batch)
                        value = criterion.forward_all(predicted, batch)[0] + 0.05 * commitment
                        if not torch.isfinite(value):
                            raise FloatingPointError('Nonfinite RVQ validation loss')
                        total += value.item() * len(batch)
                        count += len(batch)
                value = total / count
                print(json.dumps({'step': step, 'validation_loss': value, 'clips': count}), flush=True)
                checkpoint = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict(), 'step': step}
                torch.save(checkpoint, options.output / 'last.pth')
                if value < best:
                    best = value
                    torch.save({'net': model.state_dict(), 'step': step}, options.output / 'best.pth')
            if step >= options.max_steps:
                break


if __name__ == '__main__':
    main()
