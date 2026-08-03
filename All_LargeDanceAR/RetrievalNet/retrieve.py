#!/usr/bin/env python3
"""Build an InfiniteDance retrieval JSON for new audio or music55 features."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ALL_LARGEDANCEAR = Path(__file__).resolve().parents[1]
if str(ALL_LARGEDANCEAR) not in sys.path:
    sys.path.insert(0, str(ALL_LARGEDANCEAR))

from RetrievalNet.configs import get_config
from RetrievalNet.extract_music55 import extract_music55
from RetrievalNet.interclip_rp import InterCLIP_AudioJoints


STYLES = ("Ballet", "Popular", "Latin", "Modern", "Folk", "Classic")


def _load_model(config_path, checkpoint_path, device):
    model = InterCLIP_AudioJoints(get_config(str(config_path))).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _windows(features, length=384, stride=192):
    if features.ndim != 2 or features.shape[1] != 55:
        raise ValueError(f"Expected music features shaped (T, 55), got {features.shape}")
    if not len(features):
        raise ValueError("Music feature file is empty")
    windows = []
    for start in range(0, len(features), stride):
        window = features[start : start + length]
        if len(window) < length:
            repeats = (length + len(window) - 1) // len(window)
            window = np.tile(window, (repeats, 1))[:length]
        windows.append(window)
        if start + length >= len(features):
            break
    return np.stack(windows).astype(np.float32)


def _style_for(name, style_map):
    base_name = name.split("@", 1)[0]
    style = style_map.get(base_name)
    if style in STYLES:
        return style
    for candidate in STYLES:
        if base_name.lower().startswith(candidate.lower()):
            return candidate
    return None


def _load_motion_embeddings(source, style_map):
    names = []
    styles = []
    embeddings = []
    source = Path(source)
    if source.is_file():
        with np.load(source) as archive:
            archived_names = [str(name) for name in archive["names"]]
            archived_embeddings = archive["embeddings"].astype(np.float32)
        entries = zip(archived_names, archived_embeddings)
    else:
        entries = (
            (path.stem, np.load(path).astype(np.float32).reshape(-1))
            for path in sorted(source.glob("*.npy"))
        )
    for name, embedding in entries:
        style = _style_for(name, style_map)
        if style is None:
            continue
        if embedding.shape != (256,):
            raise ValueError(f"Unexpected embedding shape {embedding.shape}: {name}")
        names.append(name)
        styles.append(style)
        embeddings.append(embedding)
    if not embeddings:
        raise FileNotFoundError(f"No usable embeddings found in {source}")
    matrix = np.stack(embeddings)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
    return names, styles, matrix


def retrieve(features, model, names, styles, motion_embeddings, device, top_k):
    query = torch.from_numpy(_windows(features)).to(device)
    with torch.no_grad():
        encoded = model.encode_cond({"audio": query})["cond_emb"].cpu().numpy()
    scores = encoded @ motion_embeddings.T
    output = {}
    for query_index, row in enumerate(scores):
        per_style = {style: [] for style in STYLES}
        for candidate_index in np.argsort(row)[::-1]:
            style = styles[candidate_index]
            if len(per_style[style]) >= top_k:
                continue
            per_style[style].append(
                {
                    "muidx": query_index,
                    "name": names[candidate_index],
                    "similarity": float(row[candidate_index]),
                }
            )
            if all(len(items) >= top_k for items in per_style.values()):
                break
        output[f"idx_{query_index}"] = per_style
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="audio file or precomputed (T,55) .npy")
    parser.add_argument("output", help="output retrieval .json")
    parser.add_argument(
        "--config",
        default="RetrievalNet/configs/largedance/musicbody/InterCLIP.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/retrievalnet/retrievalnet_audio55_motion264.ckpt",
    )
    parser.add_argument(
        "--motion-embeddings",
        default="../InfiniteDanceData/dance/retrievalnet_motion_embeddings.npz",
    )
    parser.add_argument(
        "--style-map", default="../InfiniteDanceData/styles/all_style_map.json"
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_path = Path(args.input)
    features = (
        np.load(input_path).astype(np.float32)
        if input_path.suffix.lower() == ".npy"
        else extract_music55(input_path)
    )
    with open(args.style_map, "r", encoding="utf-8") as handle:
        style_map = json.load(handle)
    names, styles, motion_embeddings = _load_motion_embeddings(
        args.motion_embeddings, style_map
    )
    model = _load_model(args.config, args.checkpoint, args.device)
    result = retrieve(
        features,
        model,
        names,
        styles,
        motion_embeddings,
        args.device,
        args.top_k,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"saved {len(result)} retrieval windows to {output_path}")


if __name__ == "__main__":
    main()
