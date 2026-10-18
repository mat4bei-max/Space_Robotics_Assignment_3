"""Reusable image degradations for Advanced 1: Robust Perception.

These functions accept non-empty uint8 colour images with shape (height,
width, 3), using either RGB or BGR channel order. They return a new image
with the same shape, dtype, and channel order, without changing the input.

Apply them to RGB camera images (not raw depth measurements). Since neither
effect changes image geometry, existing bounding-box labels remain valid.
This module does not read files, run a detector, or create a ROS node.
"""

import cv2
import numpy as np


# Initial experimental settings. Tune using training/validation images,
# then keep them fixed when comparing models on the test set.
BRIGHTNESS_FACTORS = {"mild": 0.7, "moderate": 0.4, "severe": 0.2}
MOTION_BLUR_LENGTHS = {"mild": 5, "moderate": 11, "severe": 21}


def _validate_image(image):
    """Reject unsupported inputs before applying a degradation."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.uint8:
        raise TypeError("image must use uint8 intensities in the range 0 to 255")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must have non-zero height and width")


def darken_image(image, brightness_factor):
    """Scale all channels equally; 1.0 is unchanged and 0.0 is black.

    This models reduced brightness, not a complete low-light sensor model.
    Floating-point arithmetic avoids uint8 overflow or premature rounding.
    """
    _validate_image(image)
    if isinstance(brightness_factor, (bool, np.bool_)) or not isinstance(
        brightness_factor, (int, float, np.integer, np.floating)
    ):
        raise TypeError("brightness_factor must be a real number")
    if not np.isfinite(brightness_factor) or not 0.0 <= brightness_factor <= 1.0:
        raise ValueError("brightness_factor must be finite and between 0 and 1")

    darkened = image.astype(np.float32) * brightness_factor
    return np.clip(np.rint(darkened), 0, 255).astype(np.uint8)


def motion_blur_image(image, blur_length):
    """Simulate horizontal motion blur by averaging a line of pixels.

    blur_length is a positive odd integer measured in input-image pixels.
    A length of 1 returns an unchanged copy. Larger lengths smear details
    farther horizontally. Reflecting the image at its edges avoids adding
    artificial black borders.
    """
    _validate_image(image)
    if isinstance(blur_length, (bool, np.bool_)) or not isinstance(
        blur_length, (int, np.integer)
    ):
        raise TypeError("blur_length must be an integer")
    if blur_length < 1 or blur_length % 2 == 0:
        raise ValueError("blur_length must be a positive odd integer")

    kernel = np.ones((1, int(blur_length)), dtype=np.float32) / blur_length
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def degrade_image(image, effect="none", severity="mild"):
    """Apply one named degradation using the experimental presets.

    effect: 'none', 'darkness', 'motion_blur', or 'combined'.
    severity: 'mild', 'moderate', or 'severe'; ignored for 'none'.

    'combined' applies darkness followed by blur at the same severity.
    The clean baseline also returns a copy so callers can safely annotate it.
    """
    _validate_image(image)
    if effect == "none":
        return image.copy()
    if effect not in ("darkness", "motion_blur", "combined"):
        raise ValueError("effect must be 'none', 'darkness', 'motion_blur', or 'combined'")
    if severity not in ("mild", "moderate", "severe"):
        raise ValueError("severity must be 'mild', 'moderate', or 'severe'")

    if effect == "darkness":
        return darken_image(image, BRIGHTNESS_FACTORS[severity])
    if effect == "combined":
        return motion_blur_image(darken_image(image, BRIGHTNESS_FACTORS[severity]),
                                 MOTION_BLUR_LENGTHS[severity])
    return motion_blur_image(image, MOTION_BLUR_LENGTHS[severity])
