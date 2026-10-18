# Advanced 1: Robust perception

This experiment uses the existing labelled RGB dataset. It adds darkness and
horizontal motion blur, fine-tunes an RGB detector, and compares it with the
unchanged original detector. Raw depth remains unchanged for localisation.
The helpers model brightness loss and directional blur, not a complete physical
camera model. Improvement must be measured; it is not guaranteed by augmentation.

Run the following from the repository root. Dataset preparation needs Python,
NumPy, OpenCV and PyYAML. Training/evaluation additionally need the same compatible
PyTorch/Ultralytics environment used for the existing YOLO26 model, plus matplotlib.
Use that environment's Python for all training/evaluation commands. No model or
dependency is automatically downloaded. CPU is the default; use `--device 0` only
where a working CUDA installation is available.

## 1. Prepare data

```bash
python3 src/cave_explorer/advanced_functions/robust_perception.py prepare
```

Default source: `artifact_detection/datasets/rgb_full_dataset_1`.
Default output: `advanced_functions/runs/rgb_robust_v1`.
The output directory must not already exist; use `--output NEW_DIRECTORY` for a
new version. Generated files are ignored by Git. The original dataset is untouched.

Preparation validates labels, requires explicit empty label files for negative
images, and rejects decoded pixel-identical images across splits. It preserves
all source split memberships. This cannot detect near-identical views from the
same capture sequence: review those manually before claiming independent test
performance. If splits change, retrain the baseline too so its training images
cannot leak into the new test set.

Seven PNG variants are written per source image, with unchanged bounding boxes:

| Condition | Settings |
|---|---|
| Clean | No degradation |
| Darkness | Brightness factors 0.7, 0.4, 0.2 |
| Motion blur | Horizontal lengths 5, 11, 21 source-image pixels |

All variants stay in their source split. `manifest.json` records source pixel
hashes, paths, condition definitions and the random seed. `mixed.yaml` uses image
lists containing two clean entries, one randomly selected darkness severity and
one randomly selected blur severity per source image: 50%/25%/25%. Validation uses
the same family proportions. Selection is reproducible with `--seed` (default 42).
Each condition YAML exposes separate validation and test data for controlled
comparisons. Test images never enter training or checkpoint selection.

Effects run before resizing, just as in the ROS node. Use source captures at the
same resolution as the live camera. Training retains Ultralytics' usual additional
augmentations; the resolved settings are saved in the training run. This initial
experiment uses fixed synthetic severities; it does not establish robustness to
every real camera failure or combinations of effects.

## 2. Establish the baseline

```bash
python3 src/cave_explorer/advanced_functions/robust_perception.py evaluate \
  --output src/cave_explorer/advanced_functions/runs/baseline
```

The default baseline is `Artefact_detection_model_v2/train/weights/best.pt`.
Use validation only for development choices. Reserve the final test comparisons
for reporting rather than repeatedly tuning against test results.

## 3. Fine-tune

```bash
python3 src/cave_explorer/advanced_functions/robust_perception.py train --device 0
```

This starts a new run from the existing RGB checkpoint; it does not resume or
overwrite its original training. Defaults: 100 maximum epochs, patience 20,
AdamW, learning rate 0.0002, batch 4, image size 640, seed 42, no AMP. These are
initial settings to assess on validation data. See `train --help` for overrides.
Use `--weights PATH` to initialise from other local weights, including the same
base model used for the original experiment. Best weights normally appear under
`advanced_functions/runs/train/train/weights/best.pt`; Ultralytics increments the
run name if it exists and the command prints the actual best checkpoint path.

An original-versus-fine-tuned comparison measures the combined change. For a
stronger attribution to the degraded images, add a clean-only fine-tuning control
with a comparable optimisation budget. No improvement or final score is assumed.

## 4. Compare

```bash
python3 src/cave_explorer/advanced_functions/robust_perception.py evaluate \
  --robust src/cave_explorer/advanced_functions/runs/train/train/weights/best.pt \
  --output src/cave_explorer/advanced_functions/runs/comparison
```

For each model, the command first selects one operating confidence threshold on
mixed validation images (maximum mean class F1), then freezes it across all seven
test conditions. AP is computed using predictions down to confidence 0.001;
precision/recall are interpolated from the curves at the frozen threshold. They
are macro averages over classes with ground-truth support, not micro counts.
Each model's threshold can differ and is explicitly recorded.

Outputs: `comparison.csv`, `results.json`, per-class JSON files, `thresholds.json`,
`experiment.json`, and `comparison.png`. Compare clean performance as well as
every degradation severity, including failures. Detection metrics do not establish
localisation accuracy; if reporting localisation robustness, measure position
errors separately against known artefact positions.

## 5. Live demonstration

Build/source the `cave_explorer` package using your usual ROS workspace workflow.
The `advanced_functions` Python package is now included by `find_packages()`.
Start the normal Gazebo startup launch. SLAM/TF is needed for map-frame artefact
localisation, but image detection can run without Nav2 or an exploration planner.
Do not also run the usual autonomy/detection launches: they would create competing
nodes. Start the dedicated launch instead:

```bash
ros2 launch cave_explorer cave_explorer_robust_perception.launch.py \
  effect:=darkness severity:=moderate
```

Arguments: `effect` (`none`, `darkness`, `motion_blur`), `severity` (`mild`,
`moderate`, `severe`), `model` (absolute checkpoint path), `threshold`, `device`.
Defaults use the original baseline checkpoint. For the robust run, supply its
checkpoint and validation-selected threshold from `thresholds.json`:

```bash
ros2 launch cave_explorer cave_explorer_robust_perception.launch.py \
  effect:=motion_blur severity:=severe \
  model:=/absolute/path/to/robust/best.pt threshold:=0.6
```

Here 0.6 is only an example; replace it with the measured threshold. Compare both
models at the same robot pose and effect settings. Add three RViz Image displays:

| Display | Topic |
|---|---|
| Original RGB | `/advanced_perception/clean` |
| Degraded detector input | `/advanced_perception/degraded` |
| Detections on the actual input | `/detections_image` |

Change the effect during a run:

```bash
ros2 param set /cave_explorer_node degradation_effect motion_blur
ros2 param set /cave_explorer_node degradation_severity severe
```

The node preserves headers for timestamp matching and caches the actual detector
input alongside unmodified raw depth. Empty detection results publish an image
without boxes. RGB degradation with depth-mode detection is rejected. The dedicated
launch defaults to disabling planning, automatic capture, and waiting for the Nav2
action server. Ordinary autonomy defaults remain planning-enabled with no degradation.

To restore autonomous exploration, keep Gazebo/startup running, start
`ros2 launch cave_explorer cave_explorer_navigation.launch.py` in a second terminal,
and restart the robust-perception launch in a third terminal with
`enable_planning:=true` alongside your model, effect, severity and threshold
arguments. This enables the existing exploration loop and waits for Nav2. Do not
also start the normal autonomy or detection launches. Planning is selected at
startup, so changing it requires restarting the robust-perception launch.

## Verification and report

```bash
python3 -m unittest discover -s src/cave_explorer/advanced_functions -p 'test_*.py' -v
```

Tests cover split/label preservation, exact leakage rejection, missing labels,
effect semantics, and the empty-detection display regression. A simulator run
and real Ultralytics training/evaluation are separate integration checks.

For the assignment, save screenshots of both effects at several severities,
the quantitative comparison and per-class results, and matched before/after
examples. Explain failure cases, clean-performance tradeoffs, dataset splitting,
and the simplified degradation model. Disclose AI assistance as required by the
assignment.

## Combined darkness and blur experiment

`effect:=combined` applies darkness **then** horizontal motion blur to each image.
The matched severities are (brightness factor, blur pixels): mild (0.7, 5),
moderate (0.4, 11), severe (0.2, 21). These are three combined conditions,
not all nine independent severity pairings. Labels and raw depth stay unchanged.

Create a new dataset without modifying the previous experiment:

```bash
python src/cave_explorer/advanced_functions/robust_perception.py prepare \
  --include-combined \
  --reuse-dataset src/cave_explorer/advanced_functions/runs/rgb_robust_v1 \
  --output src/cave_explorer/advanced_functions/runs/rgb_robust_combined_v1
```

`--reuse-dataset` is optional. It hard-links existing images only after verifying
source split/hash and exact generated pixels, saving disk space. Treat generated
images as immutable because linked files share storage. Labels are copied.
The combined dataset uses 40% clean, 20% darkness-only, 20% blur-only and 20%
combined entries in training and validation. Ten fixed evaluation conditions
cover clean, six individual effects, and three combined effects. Earlier datasets
remain supported: evaluation reads conditions from each dataset's manifest.

Train a separate model from the same original RGB checkpoint as the previous
robust experiment (the default `--weights`), so the old robust model is preserved:

```bash
python src/cave_explorer/advanced_functions/robust_perception.py train \
  --dataset src/cave_explorer/advanced_functions/runs/rgb_robust_combined_v1 \
  --output src/cave_explorer/advanced_functions/runs/train_combined_v1 \
  --device 0
```

Compare all three models on identical conditions:

```bash
python src/cave_explorer/advanced_functions/robust_perception.py evaluate \
  --dataset src/cave_explorer/advanced_functions/runs/rgb_robust_combined_v1 \
  --previous-robust src/cave_explorer/advanced_functions/runs/train/train/weights/best.pt \
  --robust src/cave_explorer/advanced_functions/runs/train_combined_v1/train/weights/best.pt \
  --output src/cave_explorer/advanced_functions/runs/comparison_combined_v1 \
  --device 0
```

The comparison plot includes all three models. Thresholds are reselected for each
model using the new mixed validation distribution and frozen for evaluation.
These may differ from the original experiment. More samples per epoch also mean
this is not a strictly equal-optimisation-budget ablation. The existing test scenes
have already been inspected; treat this extension as exploratory and use a fresh
held-out capture sequence for stronger confirmation of generalisation.

For the live launch, use the new model path and `effect:=combined`, with your
chosen `severity` and validation-selected `threshold`. Use
`enable_planning:=true` alongside the navigation launch for exploration.
