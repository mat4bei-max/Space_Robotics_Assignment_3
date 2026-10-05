from pathlib import Path

import cv2
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    SCRIPT_DIR.parent
    / "models"
    / "trained_models"
    / "Artefact_detection_model_v2"
    / "train"
    / "weights"
    / "best.pt"
)
INPUT_IMAGE_PATH = Path(r"/home/mat4b/space_robotics/assignment_3_ws/src/cave_explorer/artifact_detection/data/raw_rgb_images/bdeed8f2-f0b7-48e9-aa35-4e493dea2182.jpg")
OUTPUT_IMAGE_PATH = Path(r"/home/mat4b/space_robotics/assignment_3_ws/report/images/rgb_detection.png")
CONFIDENCE_THRESHOLD = 0.6


COLORS = [
    (144, 238, 144),  # Alien
    (0, 100, 0),      # Gem
    (0, 0, 255),      # Sign
    (230, 216, 173),  # Sphere
    (255, 0, 0),      # Mushroom
    (255, 200, 0),    # Minerals
]


def annotate_image(
    model_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    confidence: float = 0.6,
) -> None:
    model_path, image_path, output_path = Path(model_path), Path(image_path), Path(output_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model file not found: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0.")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read input image: {image_path}")

    model = YOLO(str(model_path))
    result = model.predict(source=image, conf=confidence, verbose=False)[0]
    names = result.names
    line_width = max(1, round(min(image.shape[:2]) / 500))
    font_scale = max(0.4, line_width * 0.5)
    padding = line_width * 2

    for box in result.boxes:
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        class_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        class_name = names[class_id] if isinstance(names, list) else names[class_id]
        color = COLORS[class_id % len(COLORS)]
        label = f"{class_name} {score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
        )
        label_top = max(0, y1 - text_height - baseline - padding * 2)
        cv2.rectangle(
            image,
            (x1, label_top),
            (x1 + text_width + padding * 2, label_top + text_height + baseline + padding * 2),
            color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            image,
            label,
            (x1 + padding, label_top + text_height + padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_width,
            lineType=cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not save annotated image: {output_path}")
    print(f"Annotated image saved to: {output_path}")


def main() -> None:
    annotate_image(
        MODEL_PATH,
        INPUT_IMAGE_PATH,
        OUTPUT_IMAGE_PATH,
        CONFIDENCE_THRESHOLD,
    )


if __name__ == "__main__":
    main()
