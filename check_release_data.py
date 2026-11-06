"""Check every listed training sample; never silently reduce a split."""
import argparse
import json
from pathlib import Path


def inspect(root, split):
    names = [s.strip() for s in (root / "partition" / (split + ".txt")).read_text().splitlines() if s.strip()]
    required = {
        "music": ("music/muq_features", ".npy"),
        "tokens": ("dance/Infinite_MotionTokens_512x1024_3layer_cleandata", ".npy"),
        "style": ("styles/Alldata", ".json"),
        "retrieval": ("dance/retrieval_s192_l384_style", ".json"),
    }
    missing = {key: [name for name in names if not (root / folder / (name + suffix)).is_file()]
               for key, (folder, suffix) in required.items()}
    unavailable = set().union(*map(set, missing.values()))
    return {"listed": len(names), "complete_file_sets": len(names) - len(unavailable), "missing": missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "InfiniteDanceData")
    parser.add_argument("--splits", nargs="+", default=["All_train", "All_val"])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {split: inspect(args.data_root, split) for split in args.splits}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    for split, data in report.items():
        print(split, f"{data['complete_file_sets']}/{data['listed']} complete file sets",
              {k: len(v) for k, v in data['missing'].items()})
    raise SystemExit(1 if any(any(v["missing"].values()) for v in report.values()) else 0)


if __name__ == "__main__":
    main()
