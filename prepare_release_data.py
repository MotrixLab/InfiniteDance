"""Extract the downloaded ECCV release archives into the documented layout."""
import argparse
import json
import tarfile
from pathlib import Path


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        for member in handle:
            target = (destination / member.name).resolve()
            is_root_directory = member.isdir() and target == destination.resolve()
            if (not is_root_directory and destination.resolve() not in target.parents) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member: {member.name}")
            handle.extract(member, path=destination)
    print(f"Extracted {archive.name} -> {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "InfiniteDanceData")
    parser.add_argument("--training", action="store_true", help="Also extract all training MuQ shards and per-clip styles")
    args = parser.parse_args()
    root = args.data_root.resolve()
    archives = [
        ("Infinite_MotionTokens_512x1024_3layer_cleandata.tar.gz", "dance"),
        ("retrieval_s192_l384_style.tar.gz", "dance"),
        ("alldata_new_joint_vecs264_ft_balanced.tar.gz", "dance"),
        ("test_eval861_joint_vecs264.tar", "dance"),
        ("infinitedance_smplx_smooth.tar", "dance"),
        ("musicfeature_55_allmusic_pure.tar.gz", "music"),
        ("muq_features_test_infinitedance.tar.gz", "music/muq_features"),
    ]
    if args.training:
        manifest = json.loads((root / "release_training_manifest.json").read_text())
        archives += [(name, "music") for name in manifest["music_shards"]]
        archives += [("styles_Alldata_eccv.tar.gz", "styles")]
    for name, directory in archives:
        extract(root / name, root / directory)


if __name__ == "__main__":
    main()
