"""Reproducible Advanced 1 dataset preparation, training and evaluation.

Run directly with Python; torch/ultralytics are imported only when needed.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    from .advanced_cave_explorer import degrade_image
except ImportError:
    from advanced_cave_explorer import degrade_image

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent / "artifact_detection"
CONDITIONS = {"clean": ("none", "mild")}
CONDITIONS.update({f"{effect}_{level}": (effect, level)
                   for effect in ("darkness", "motion_blur")
                   for level in ("mild", "moderate", "severe")})


def write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def audit(source):
    """Validate YOLO detection labels and reject exact cross-split leakage."""
    spec = yaml.safe_load((source / "data.yaml").read_text())
    names = spec["names"]
    root = Path(spec.get("path", source))
    if not root.is_absolute():
        root = source / root
    records, seen = {}, {}
    for split in ("train", "val", "test"):
        image_dir = root / spec[split]
        if image_dir.name != "images" or not image_dir.is_dir():
            raise ValueError(f"Expected a split images directory: {image_dir}")
        records[split] = []
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Unreadable image: {image_path}")
            digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                raise ValueError(f"Cross-split duplicate: {image_path} and {seen[digest][1]}")
            seen[digest] = (split, str(image_path))
            label = image_dir.parent / "labels" / (image_path.stem + ".txt")
            # Explicit empty labels distinguish negatives from missing annotations.
            if not label.is_file():
                raise ValueError(f"Missing label (use an empty file for negatives): {label}")
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                values = np.asarray([float(x) for x in line.split()])
                if (len(values) != 5 or not np.isfinite(values).all()
                        or values[0] != int(values[0]) or not 0 <= values[0] < len(names)
                        or np.any(values[1:] < 0) or np.any(values[1:] > 1)
                        or np.any(values[3:] <= 0)):
                    raise ValueError(f"Invalid YOLO label: {label}: {line}")
            records[split].append((image_path, label, digest))
        if not records[split]:
            raise ValueError(f"Empty split: {split}")
    return names, records


def prepare(args):
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    names, records = audit(source)
    rng = np.random.default_rng(args.seed)
    reusable = {}
    if getattr(args, 'reuse_dataset', None):
        old = json.loads((args.reuse_dataset / 'manifest.json').read_text())
        reusable = {(entry['split'], entry['sha256']): entry['variants'] for entry in old['images']}
    conditions = dict(CONDITIONS)
    include_combined = getattr(args, 'include_combined', False)
    if include_combined:
        conditions.update({f'combined_{level}': ('combined', level)
                           for level in ('mild', 'moderate', 'severe')})
    manifest = {"source": str(source), "seed": args.seed,
                "conditions": conditions, "images": [],
                "combined_order": "darkness_then_horizontal_blur",
                "mixture": "40% clean, 20% darkness, 20% blur, 20% combined" if include_combined
                           else "50% clean, 25% darkness, 25% blur",
                "note": "Exact duplicates checked; review capture sequences for near-duplicate leakage."}
    # Two clean entries, one darkness, one blur, and optionally one combined.
    # Validation uses the same mixture; tests stay separate.
    mixtures = {split: [] for split in ("train", "val")}
    for split, entries in records.items():
        for index, (image_path, label, digest) in enumerate(entries):
            original = cv2.imread(str(image_path))
            paths = {}
            for condition, (effect, severity) in conditions.items():
                target = output / condition / split / "images" / f"{index:06d}_{image_path.stem}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                pixels = degrade_image(original, effect, severity)
                reuse = reusable.get((split, digest), {}).get(condition)
                if reuse and np.array_equal(cv2.imread(reuse), pixels):
                    target.hardlink_to(reuse)
                elif not cv2.imwrite(str(target), pixels):
                    raise OSError(f"Cannot write {target}")
                target_label = target.parent.parent / "labels" / (target.stem + ".txt")
                target_label.parent.mkdir(parents=True, exist_ok=True)
                target_label.write_text(label.read_text())
                paths[condition] = str(target)
            if split in mixtures:
                mixtures[split].extend([paths["clean"], paths["clean"],
                    paths[f"darkness_{rng.choice(['mild', 'moderate', 'severe'])}"],
                    paths[f"motion_blur_{rng.choice(['mild', 'moderate', 'severe'])}"]])
                if include_combined:
                    mixtures[split].append(paths[f"combined_{rng.choice(['mild', 'moderate', 'severe'])}"])
            manifest["images"].append({"split": split, "source": str(image_path),
                                       "sha256": digest, "variants": paths})
    for split, paths in mixtures.items():
        (output / f"mixed_{split}.txt").write_text("\n".join(paths) + "\n")
    for condition in conditions:
        write_yaml(output / f"{condition}.yaml", {
            "path": str(output / condition), "train": "train/images",
            "val": "val/images", "test": "test/images", "names": names})
    write_yaml(output / "mixed.yaml", {"path": str(output),
        "train": str(output / "mixed_train.txt"), "val": str(output / "mixed_val.txt"),
        "test": str(output / "clean/test/images"), "names": names})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(output), "source_counts": {
        split: len(entries) for split, entries in records.items()}}, indent=2))


def train(args):
    from ultralytics import YOLO
    if not args.weights.is_file():
        raise ValueError(f"Weights do not exist: {args.weights}")
    model = YOLO(str(args.weights.resolve()))
    model.train(data=str(args.dataset.resolve() / "mixed.yaml"),
                project=str(args.output.resolve()), name="train", exist_ok=False,
                epochs=args.epochs, patience=args.patience, batch=args.batch,
                imgsz=args.imgsz, device=args.device, workers=0, amp=False,
                optimizer="AdamW", lr0=args.lr, weight_decay=0.0,
                seed=args.seed, resume=False, plots=True)
    print(f"Best robust checkpoint: {model.trainer.best}")


def evaluate(args):
    from ultralytics import YOLO
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Choose a new evaluation output: {output}")
    output.mkdir(parents=True)
    rows, thresholds = [], {}
    conditions = json.loads((args.dataset / 'manifest.json').read_text())['conditions']
    models = [("baseline", args.baseline)]
    if args.previous_robust is not None:
        models.append(("previous_robust", args.previous_robust))
    if args.robust is not None:
        models.append(("robust", args.robust))
    for model_name, weights in models:
        if not weights.is_file():
            raise ValueError(f"Weights do not exist: {weights}")
        model = YOLO(str(weights.resolve()))
        common = dict(imgsz=args.imgsz, batch=args.batch, device=args.device,
                      workers=0, conf=0.001, plots=False, project=str(output))
        validation = model.val(data=str(args.dataset.resolve() / "mixed.yaml"),
                               split="val", name=f"{model_name}_threshold", **common)
        box = validation.box
        threshold = float(box.px[np.argmax(box.f1_curve.mean(axis=0))])
        thresholds[model_name] = threshold
        (output / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
        for condition in conditions:
            metrics = model.val(data=str(args.dataset.resolve() / f"{condition}.yaml"),
                                split="test", name=f"{model_name}_{condition}", **common)
            box = metrics.box
            # AP uses the full low-cutoff prediction set; operating P/R are
            # interpolated at the threshold selected ONLY on validation data.
            p = np.asarray([np.interp(threshold, box.px, curve) for curve in box.p_curve])
            r = np.asarray([np.interp(threshold, box.px, curve) for curve in box.r_curve])
            row = dict(model=model_name, condition=condition, threshold=threshold,
                       map50=float(box.map50), map50_95=float(box.map),
                       precision=float(p.mean()), recall=float(r.mean()))
            rows.append(row)
            per_class = []
            for i, class_id in enumerate(box.ap_class_index):
                per_class.append(dict(class_id=int(class_id), name=model.names[int(class_id)],
                                      map50=float(box.ap50[i]), map50_95=float(box.ap[i]),
                                      precision=float(p[i]), recall=float(r[i])))
            (output / f"{model_name}_{condition}_classes.json").write_text(json.dumps(per_class, indent=2))
            (output / "results.json").write_text(json.dumps(rows, indent=2))
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "experiment.json").write_text(json.dumps({
        "baseline": str(args.baseline.resolve()),
        "robust": str(args.robust.resolve()) if args.robust else None,
        "previous_robust": str(args.previous_robust.resolve()) if args.previous_robust else None,
        "dataset": str(args.dataset.resolve()), "thresholds": thresholds,
        "imgsz": args.imgsz, "confidence_floor": 0.001}, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    for model_name, _ in models:
        selected = [row for row in rows if row["model"] == model_name]
        ax.plot([row["condition"] for row in selected],
                [row["map50_95"] for row in selected], "o-", label=model_name)
    ax.set_ylabel("mAP50–95")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(f"Comparison saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, default=ARTIFACTS / "datasets/rgb_full_dataset_1")
    prepare_parser.add_argument("--output", type=Path, default=HERE / "runs/rgb_robust_v1")
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--include-combined", action="store_true",
                                help="Add simultaneous darkness and blur at three matched severities")
    prepare_parser.add_argument("--reuse-dataset", type=Path,
                                help="Hard-link pixel-verified existing variants to save disk space (same filesystem)")
    prepare_parser.set_defaults(run=prepare)
    for command, function in (("train", train), ("evaluate", evaluate)):
        sub = commands.add_parser(command)
        sub.add_argument("--dataset", type=Path, default=HERE / "runs/rgb_robust_v1")
        sub.add_argument("--output", type=Path, default=HERE / "runs" / command)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--imgsz", type=int, default=640)
        sub.add_argument("--batch", type=int, default=4)
        sub.set_defaults(run=function)
        if command == "train":
            sub.add_argument("--weights", type=Path, default=ARTIFACTS / "models/trained_models/Artefact_detection_model_v2/train/weights/best.pt")
            sub.add_argument("--epochs", type=int, default=100)
            sub.add_argument("--patience", type=int, default=20)
            sub.add_argument("--lr", type=float, default=0.0002)
            sub.add_argument("--seed", type=int, default=42)
        else:
            sub.add_argument("--baseline", type=Path, default=ARTIFACTS / "models/trained_models/Artefact_detection_model_v2/train/weights/best.pt")
            sub.add_argument("--robust", type=Path, help="Omit to establish the baseline before training")
            sub.add_argument("--previous-robust", type=Path,
                             help="Also compare the earlier separate-effects robust model")
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
