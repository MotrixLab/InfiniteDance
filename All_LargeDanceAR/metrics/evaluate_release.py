"""Evaluate one prediction per declared clip without reading stale feature caches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy import linalg
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from scipy.spatial.distance import pdist
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features

PREFIX = re.compile(r'^dance_sample_(\d+)_(\d+)_(.+)$')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_names(path):
    names = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError('Empty or duplicate identities in ' + str(path))
    return names


def select_predictions(root, names, sample_index=0, allow_partial=False):
    selected, ignored, allowed = {}, [], set(names)
    for path in sorted(Path(root).glob('*.npy')):
        name = path.stem
        match = PREFIX.fullmatch(name)
        if match:
            if int(match.group(1)) != sample_index:
                ignored.append(path.name)
                continue
            name = match.group(3)
        if name not in allowed:
            ignored.append(path.name)
            continue
        if name in selected:
            raise ValueError('Multiple predictions for canonical identity ' + name)
        selected[name] = path
    missing = [name for name in names if name not in selected]
    if missing and not allow_partial:
        raise ValueError('%d canonical predictions missing; first: %s. '
                         'Use --allow-partial only for partial evaluation.' % (len(missing), missing[:5]))
    if len(selected) < 2:
        raise ValueError('FID and diversity require at least two predictions')
    return [(name, selected[name]) for name in names if name in selected], missing, ignored


def load_joints(path, max_frames):
    joints = np.load(path, allow_pickle=False)
    if joints.ndim == 2 and joints.shape[1] == 66:
        joints = joints.reshape(-1, 22, 3)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError('%s: expected world joints (T,22,3) or (T,66), got %s' % (path, joints.shape))
    if max_frames > 0:
        joints = joints[:max_frames]
    if len(joints) < 3 or not np.isfinite(joints).all():
        raise ValueError('%s: fewer than three frames or nonfinite joints' % path)
    return joints


def motion_features(joints):
    # Preserve the published scorer's preprocessing and feature definitions.
    relative = joints - joints[:1, :1]
    relative[:, 1:] -= relative[:, :1]
    kinetic = extract_kinetic_features(relative).astype(np.float32)
    manual = extract_manual_features(relative).astype(np.float32)
    if not np.isfinite(kinetic).all() or not np.isfinite(manual).all():
        raise ValueError('Nonfinite extracted motion features')
    return kinetic, manual


def beat_alignment(joints, music):
    length = len(joints) - 1
    if music.ndim != 2 or music.shape[1] != 55 or len(music) < length:
        raise ValueError('Beat features must have shape (T,55) and cover the evaluated motion')
    if not np.isfinite(music[:length, 53]).all():
        raise ValueError('Nonfinite music beats')
    velocity = np.mean(np.linalg.norm(np.diff(joints, axis=0), axis=2), axis=1)
    motion_beats = argrelextrema(gaussian_filter(velocity, 5), np.less)[0]
    music_beats = np.flatnonzero(music[:length, 53].astype(bool))
    if not len(music_beats) or not len(motion_beats):
        raise ValueError('Beat alignment undefined: music or motion has no detected beats')
    distance = np.min((music_beats[:, None] - motion_beats[None, :]) ** 2, axis=1)
    return float(np.exp(-distance / 18.0).mean())


def distribution_metrics(predicted, ground_truth):
    predicted, ground_truth = np.asarray(predicted), np.asarray(ground_truth)
    if min(len(predicted), len(ground_truth)) < 2:
        raise ValueError('At least two samples required in each feature distribution')
    if not np.isfinite(predicted).all() or not np.isfinite(ground_truth).all():
        raise ValueError('Nonfinite distribution features')
    mean, std = ground_truth.mean(axis=0), ground_truth.std(axis=0)
    pred = (predicted - mean) / (std + 1e-10)
    gt = (ground_truth - mean) / (std + 1e-10)
    cov_pred, cov_gt = np.cov(pred, rowvar=False), np.cov(gt, rowvar=False)
    covmean = linalg.sqrtm(cov_pred.dot(cov_gt))
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_pred.shape[0]) * 1e-5
        covmean = linalg.sqrtm((cov_pred + offset).dot(cov_gt + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError('FID covariance square root has a significant imaginary component')
        covmean = covmean.real
    delta = pred.mean(axis=0) - gt.mean(axis=0)
    fid = float(delta.dot(delta) + np.trace(cov_pred) + np.trace(cov_gt) - 2 * np.trace(covmean))
    return fid, float(pdist(pred).mean())


def evaluate_sample(job):
    name, path, music_root, max_frames = job
    joints = load_joints(path, max_frames)
    kinetic, manual = motion_features(joints)
    music_path = Path(music_root) / (name + '.npy')
    music = np.load(music_path, allow_pickle=False)
    try:
        ba, ba_error = beat_alignment(joints, music), None
    except ValueError as error:
        ba, ba_error = None, str(error)
    return {'name': name, 'prediction': str(path), 'frames': len(joints),
            'prediction_sha256': sha256(path), 'music_sha256': sha256(music_path),
            'ba': ba, 'ba_error': ba_error, 'kinetic': kinetic, 'manual': manual}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-root', type=Path, required=True)
    parser.add_argument('--gt-features', type=Path, required=True)
    parser.add_argument('--gt-list', type=Path, required=True)
    parser.add_argument('--eval-list', type=Path, required=True)
    parser.add_argument('--music-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=1024,
                        help='Same frame cap for FID/Div and BA; <=0 uses the full clip')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    names = read_names(args.eval_list)
    predictions, missing, ignored = select_predictions(args.pred_root, names, args.sample_index, args.allow_partial)
    gt_names = set(read_names(args.gt_list))
    with np.load(args.gt_features, allow_pickle=False) as archive:
        archive_names = list(map(str, archive['names']))
        if len(archive_names) != len(set(archive_names)):
            raise ValueError('Duplicate identities in GT feature archive')
        indices = [i for i, name in enumerate(archive_names) if name in gt_names]
        gt_kinetic, gt_manual = archive['kinetic'][indices], archive['manual'][indices]
    if gt_kinetic.shape[1] == 72:
        gt_kinetic = gt_kinetic[:, :66]
    jobs = [(name, path, args.music_root, args.max_frames) for name, path in predictions]
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    if args.workers == 1:
        rows = list(map(evaluate_sample, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(evaluate_sample, jobs))
    fid_k, div_k = distribution_metrics(np.stack([r['kinetic'] for r in rows]), gt_kinetic)
    fid_m, div_m = distribution_metrics(np.stack([r['manual'] for r in rows]), gt_manual)
    ba_errors = {r['name']: r['ba_error'] for r in rows if r['ba_error']}
    report = {
        'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'partial' if missing else 'complete_declared_list',
        'canonical_count': len(names), 'predictions': len(rows), 'gt_features': len(indices),
        'missing_predictions': missing, 'ignored_predictions': ignored,
        'gt_list_without_features': sorted(gt_names - set(archive_names)),
        'protocol': {'feature_implementation': 'released kinetic/manual_new, unchanged',
                     'kinetic_frame_time': 1 / 60, 'max_frames_all_metrics': args.max_frames,
                     'root_preprocessing': 'initial root subtraction then per-frame local non-root joints',
                     'normalization': 'GT mean/std + 1e-10', 'sample_index': args.sample_index,
                     'beat_smoothing_sigma': 5, 'beat_match_sigma': 3},
        'inputs': {key: {'path': str(value.resolve()), 'sha256': sha256(value)}
                   for key, value in [('gt_features', args.gt_features), ('gt_list', args.gt_list),
                                      ('eval_list', args.eval_list)]},
        'code_sha256': {str(p.relative_to(Path(__file__).parent)): sha256(p)
                        for p in [Path(__file__), Path(__file__).parent / 'features/kinetic.py',
                                  Path(__file__).parent / 'features/manual_new.py',
                                  Path(__file__).parent / 'features/futils.py']},
        'metrics': {'fid_k': fid_k, 'fid_m': fid_m, 'div_k': div_k, 'div_m': div_m,
                    'ba': None if ba_errors else float(np.mean([r['ba'] for r in rows]))},
        'ba_errors': ba_errors,
        'samples': [{k: v for k, v in row.items() if k not in ('kinetic', 'manual')} for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: report[key] for key in ('scope', 'predictions', 'gt_features', 'metrics', 'ba_errors')}, indent=2))
    return 1 if ba_errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
