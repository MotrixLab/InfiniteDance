#!/usr/bin/env python3
"""Extract the 30-fps, 55-d music features used by RetrievalNet.

The feature order follows Bailando's released preprocessing: 20 MFCC, 20 MFCC
deltas, 12 chroma-CQT, onset strength, beat one-hot, and the first tempogram
bin. Bailando is Copyright 2022 S-Lab and distributed for non-commercial use.
"""

import argparse
from pathlib import Path

import librosa
import numpy as np


SAMPLE_RATE = 15360
HOP_LENGTH = 512


def extract_music55(audio_path):
    audio, sample_rate = librosa.load(
        str(audio_path), sr=SAMPLE_RATE, mono=True
    )
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc, width=3)

    harmonic, percussive = librosa.effects.hpss(audio)
    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        n_octaves=5,
    )
    onset = librosa.onset.onset_strength(
        y=percussive,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        aggregate=np.median,
    )
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset, sr=sample_rate, hop_length=HOP_LENGTH
    )
    beat = np.zeros((1, onset.shape[0]), dtype=np.float32)
    beat[0, beat_frames[beat_frames < onset.shape[0]]] = 1.0
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset, sr=sample_rate, hop_length=HOP_LENGTH
    )

    frame_count = min(
        mfcc.shape[1],
        mfcc_delta.shape[1],
        chroma.shape[1],
        onset.shape[0],
        beat.shape[1],
        tempogram.shape[1],
    )
    features = np.concatenate(
        [
            mfcc[:, :frame_count],
            mfcc_delta[:, :frame_count],
            chroma[:, :frame_count],
            onset[None, :frame_count],
            beat[:, :frame_count],
            tempogram[:, :frame_count],
        ],
        axis=0,
    )[:55]
    return features.T.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="input wav/mp3/flac file")
    parser.add_argument("output", help="output .npy path")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features = extract_music55(args.audio)
    np.save(output, features)
    print(f"saved {features.shape} float32 features to {output}")


if __name__ == "__main__":
    main()
