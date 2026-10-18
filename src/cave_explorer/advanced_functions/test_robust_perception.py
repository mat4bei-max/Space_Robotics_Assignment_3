"""Offline integrity and ROS callback regression tests; no simulator needed."""
import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from advanced_cave_explorer import degrade_image
from robust_perception import audit, prepare


class RobustPerceptionTests(unittest.TestCase):
    def make_source(self, root):
        source = root / 'source'
        source.mkdir()
        (source / 'data.yaml').write_text(yaml.safe_dump({
            'names': ['artefact'], **{s: f'{s}/images' for s in ('train', 'val', 'test')}}))
        for i, split in enumerate(('train', 'val', 'test')):
            images = source / split / 'images'
            labels = source / split / 'labels'
            images.mkdir(parents=True)
            labels.mkdir()
            image = np.full((20, 30, 3), 30 + i * 60, dtype=np.uint8)
            cv2.imwrite(str(images / 'object.png'), image)
            (labels / 'object.txt').write_text('0 0.5 0.5 0.4 0.4\n' if i != 2 else '')
        return source

    def test_generation_preserves_labels_splits_and_mixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source(root)
            output = root / 'result'
            prepare(SimpleNamespace(source=source, output=output, seed=42))
            self.assertEqual(len(list(output.glob('*/test/images/*.png'))), 7)
            self.assertTrue(all(p.read_text() == '' for p in output.glob('*/test/labels/*.txt')))
            paths = (output / 'mixed_train.txt').read_text().splitlines()
            self.assertEqual(len(paths), 4)
            self.assertEqual(sum('/clean/' in p for p in paths), 2)
            self.assertTrue(all('/train/' in p for p in paths))
            with self.assertRaises(ValueError):
                prepare(SimpleNamespace(source=source, output=output, seed=42))

    def test_reject_cross_split_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_source(Path(tmp))
            (source / 'test/images/object.png').write_bytes((source / 'train/images/object.png').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Cross-split'):
                audit(source)

    def test_combined_dataset_includes_both_effects_and_original_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source(root)
            output = root / 'combined'
            prepare(SimpleNamespace(source=source, output=output, seed=42, include_combined=True))
            self.assertEqual(len(list(output.glob('*/test/images/*.png'))), 10)
            paths = (output / 'mixed_train.txt').read_text().splitlines()
            self.assertEqual(len(paths), 5)
            self.assertEqual(sum('/clean/' in p for p in paths), 2)
            self.assertEqual(sum('/combined_' in p for p in paths), 1)
            self.assertTrue(all('/train/' in p for p in paths))
            image = np.zeros((25, 25, 3), dtype=np.uint8)
            image[:, 12] = 200
            combined = degrade_image(image, 'combined', 'mild')
            self.assertEqual(int(combined[0, 12, 0]), 28)
            self.assertEqual(int(combined[0, 11, 0]), 28)
            self.assertEqual(int(image[0, 12, 0]), 200)

    def test_reject_missing_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_source(Path(tmp))
            (source / 'train/labels/object.txt').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing label'):
                audit(source)

    def test_effects_preserve_input_and_geometry(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[:, 15] = 200
        before = image.copy()
        dark = degrade_image(image, 'darkness', 'moderate')
        blur = degrade_image(image, 'motion_blur', 'mild')
        np.testing.assert_array_equal(image, before)
        self.assertEqual(int(dark[0, 15, 0]), 80)
        self.assertEqual(int(blur[0, 15, 0]), 40)
        self.assertEqual(blur.shape, image.shape)

    def test_empty_detection_publishes_cached_frame(self):
        # Execute the actual callback body without requiring a ROS installation.
        import asyncio
        node_file = Path(__file__).resolve().parents[1] / 'cave_explorer/cave_explorer.py'
        tree = ast.parse(node_file.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CaveExplorer')
        callback = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef)
                        and n.name == 'artifact_detections_callback')
        scope = {}
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(node_file), 'exec'), scope)
        pixels = np.full((2, 2, 3), 42, dtype=np.uint8)
        header = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=2))
        frame = SimpleNamespace(pixels=pixels, header=header)
        published = []
        node = SimpleNamespace(
            detection_images_={(1, 2): (frame, frame)},
            cv_bridge_=SimpleNamespace(imgmsg_to_cv2=lambda msg, **kw: msg.pixels,
                                       cv2_to_imgmsg=lambda image, **kw: SimpleNamespace(pixels=image)),
            annotate_image=lambda image, detections: image,
            image_detections_pub_=SimpleNamespace(publish=published.append))
        asyncio.run(scope['artifact_detections_callback'](node, SimpleNamespace(header=header, detections=[])))
        self.assertEqual(len(published), 1)
        self.assertIs(published[0].header, header)
        np.testing.assert_array_equal(published[0].pixels, pixels)
        self.assertFalse(node.detection_images_)

    def test_live_degradation_preserves_header_and_raw_depth(self):
        node_file = Path(__file__).resolve().parents[1] / 'cave_explorer/cave_explorer.py'
        tree = ast.parse(node_file.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CaveExplorer')
        callback = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'image_callback')
        scope = {'np': np, 'cv2': cv2, 'degrade_image': degrade_image}
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(node_file), 'exec'), scope)
        header = SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=20))
        rgb = SimpleNamespace(pixels=np.full((4, 4, 3), 200, dtype=np.uint8), header=header)
        depth = SimpleNamespace(pixels=np.full((4, 4), 2.0), header=header)
        sent, clean, degraded = [], [], []
        node = SimpleNamespace(
            cv_bridge_=SimpleNamespace(imgmsg_to_cv2=lambda msg, **kw: msg.pixels,
                                       cv2_to_imgmsg=lambda image, **kw: SimpleNamespace(pixels=image)),
            capture_requested_=False, detection_image_count_=0, detection_every_n_images_=1,
            detection_images_={}, depth_detection_=False,
            get_parameter=lambda name: SimpleNamespace(value={
                'degradation_effect': 'darkness', 'degradation_severity': 'moderate'}[name]),
            clean_image_pub_=SimpleNamespace(publish=clean.append),
            degraded_image_pub_=SimpleNamespace(publish=degraded.append),
            detection_image_pub_=SimpleNamespace(publish=sent.append))
        scope['image_callback'](node, rgb, depth)
        self.assertEqual(int(sent[0].pixels[0, 0, 0]), 80)
        self.assertEqual(int(clean[0].pixels[0, 0, 0]), 200)
        self.assertIs(sent[0].header, header)
        self.assertIs(node.detection_images_[(10, 20)][0], sent[0])
        self.assertIs(node.detection_images_[(10, 20)][1], depth)
        self.assertTrue(np.all(depth.pixels == 2.0))


if __name__ == '__main__':
    unittest.main()
