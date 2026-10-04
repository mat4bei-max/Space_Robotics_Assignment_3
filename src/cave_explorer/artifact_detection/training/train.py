import json
from pathlib import Path

from omegaconf import OmegaConf
from ultralytics import YOLO
from ultralytics.utils.metrics import smooth


def main():

    # define project directories
    training_dir = Path(__file__).resolve().parent
    dataset_dir = training_dir.parent / "datasets"
    base_model_dir = training_dir.parent / "models" / "base_models"  
    output_dir = training_dir.parent / "models" / "trained_models"

    # load configuration
    config = OmegaConf.load(training_dir / "config.yaml")
    setup = config.setup
    params = config.hyperparameters
    runtime = config.runtime_settings
    model_output_dir = output_dir / setup.model_name

    # define training and evaluation arguments
    train_args = dict(
        data = dataset_dir / setup.dataset,
        name = "train",
        project = model_output_dir,
        plots = True,
        epochs = params.epochs,
        patience = params.patience,
        lr0 = params.learning_rate,
        weight_decay = params.weight_decay,
        optimizer = params.optimizer,
        batch = params.batch_size,
        imgsz = params.img_size,
        device = runtime.device,
        workers = runtime.workers,
        amp = runtime.amp,
    )

    evaluation_args = dict(
        data = dataset_dir / setup.dataset,
        batch = params.batch_size,
        imgsz = params.img_size,
        project = model_output_dir,
        device = runtime.device,
        workers = runtime.workers,
    )

    # train the model
    model = YOLO(base_model_dir / setup.yolo_base_model)
    model.train(**train_args)

    # determine confidence at the best F1 point on the validation set
    best_model = YOLO(str(model.trainer.best))
    validation = best_model.val(**evaluation_args, split="val", conf=0.001, plots=False, name="threshold_selection")
    best_index = smooth(validation.box.f1_curve.mean(0), 0.1).argmax()
    best_conf = float(validation.box.px[best_index])
    print(f"Confidence at best validation F1: {best_conf:.4f}")

    # apply the selected confidence to test evaluation
    metrics = best_model.val(**evaluation_args, split="test", conf=best_conf, plots=True, name="evaluation")
    results = {"confidence": best_conf, **metrics.results_dict}
    with (Path(metrics.save_dir) / "metrics.json").open("w") as file:
        json.dump(results, file, indent=2)
    print(results)

if __name__ == "__main__":
    main()
