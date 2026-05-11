"""
Extract per-frame MuQ embeddings (T, 1024) from audio files (wav / mp3 / ...).
The resulting .npy is the music_features expected by infer_llama_infinitedance.py.

Audio is resampled to 24kHz and MuQ produces ~30 frames / second.

Usage:
    python utils/extract_muq.py --in_dir path/to/audio --out_dir path/to/muq_features
    python utils/extract_muq.py --in_file song.wav    --out_dir path/to/muq_features
"""
import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from muq import MuQ
from tqdm import tqdm


AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def load_muq(device: str) -> MuQ:
    model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def extract_one(model: MuQ, audio_path: str, device: str) -> np.ndarray:
    wav, _ = librosa.load(audio_path, sr=24000)
    wavs = torch.tensor(wav, device=device).unsqueeze(0)
    feats = model(wavs, output_hidden_states=True).last_hidden_state
    return feats.squeeze(0).cpu().numpy()


def collect_inputs(in_file: str, in_dir: str) -> list:
    if in_file:
        return [in_file]
    if not in_dir:
        raise SystemExit("provide --in_file or --in_dir")
    items = []
    for root, _, names in os.walk(in_dir):
        for n in sorted(names):
            if n.lower().endswith(AUDIO_EXTS):
                items.append(os.path.join(root, n))
    if not items:
        raise SystemExit(f"no audio files found under {in_dir}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default=None, help="single audio file")
    ap.add_argument("--in_dir", default=None, help="directory of audio files (recursive)")
    ap.add_argument("--out_dir", required=True, help="output directory for .npy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true", help="re-extract even if .npy exists")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inputs = collect_inputs(args.in_file, args.in_dir)
    print(f"Found {len(inputs)} audio file(s); loading MuQ on {args.device} ...")
    model = load_muq(args.device)

    for path in tqdm(inputs, desc="extract"):
        base = Path(path).stem
        out_path = os.path.join(args.out_dir, f"{base}.npy")
        if os.path.exists(out_path) and not args.overwrite:
            continue
        try:
            feats = extract_one(model, path, args.device)
        except Exception as e:
            tqdm.write(f"FAIL {path}: {e}")
            continue
        np.save(out_path, feats)


if __name__ == "__main__":
    main()
