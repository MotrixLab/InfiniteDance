"""Train the released RetrievalNet architecture on its published paired windows.

This is a portable retraining entry, not a claim of bitwise checkpoint replay.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from RetrievalNet.configs import get_config
from RetrievalNet.interclip_rp import InterCLIP_AudioJoints
from RetrievalNet.paired_data import AudioMotion_mb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "InfiniteDanceData")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.environ["INFINITEDANCE_DATA_ROOT"] = str(args.data_root.resolve())
    config_dir = ROOT / "RetrievalNet/configs/largedance/musicbody"
    config = get_config(str(config_dir / "InterCLIP.yaml"))
    data_config = get_config(str(config_dir / "datasets45.yaml")).largedance
    dataset = AudioMotion_mb(**data_config)
    if len(dataset) < args.batch_size:
        raise ValueError(f"Need at least one complete batch: {len(dataset)} windows, batch={args.batch_size}")
    lengths = {row["end_idx"] - row["start_idx"] for row in dataset.data_list}
    if lengths != {config.TransEncoder.max_seq_len}:
        raise ValueError(f"Metadata lengths {lengths} do not match the released encoder")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.workers)
    model = InterCLIP_AudioJoints(config).to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.2)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    step = 0
    for epoch in range(args.epochs):
        for names, audio, motion in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model({"seqname": names, "audio": audio.to(args.device),
                             "joints": motion.to(args.device)})
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            step += 1
            if step == 1 or step % 40 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "loss": loss.item()}), flush=True)
            if args.max_steps is not None and step >= args.max_steps:
                break
        scheduler.step()
        done = args.max_steps is not None and step >= args.max_steps
        if done or (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            torch.save({"state_dict": {"model." + k: v for k, v in model.state_dict().items()},
                        "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                        "step": step}, args.output / "last.ckpt")
        if done:
            break


if __name__ == "__main__":
    main()
