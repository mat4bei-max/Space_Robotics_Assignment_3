import math
import unittest

import numpy as np

from advanced_functions.cave_geometry import (
    analyse_grid, cell_to_world, world_to_cell, detect_passages,
)


class GeometryTests(unittest.TestCase):
    def test_square_room_clearance(self):
        grid = np.full((11, 11), 100)
        grid[1:-1, 1:-1] = 0
        result = analyse_grid(grid, 1.0, (5, 5))
        self.assertEqual(result.widest_cell, (5, 5))
        self.assertAlmostEqual(result.clearance[5, 5], 5 - 1 / math.sqrt(2))
        self.assertFalse(result.provisional)

    def test_larger_disconnected_room_excluded(self):
        grid = np.full((15, 25), 100)
        grid[2:7, 2:7] = 0
        grid[1:14, 10:24] = 0
        result = analyse_grid(grid, 1.0, (4, 4))
        self.assertEqual(result.widest_cell, (4, 4))
        self.assertFalse(result.reachable[:, 10:].any())

    def test_narrow_connection_excludes_large_room(self):
        grid = np.full((15, 25), 100)
        grid[2:9, 1:8] = 0
        grid[1:14, 12:24] = 0
        grid[5, 8:12] = 0
        result = analyse_grid(grid, 1.0, (5, 4))
        self.assertFalse(result.reachable[:, 12:].any())

    def test_unknown_and_exterior_are_provisional(self):
        for grid in (np.zeros((11, 11), dtype=int),
                     np.pad(np.zeros((9, 9), dtype=int), 1, constant_values=-1)):
            result = analyse_grid(grid, 1.0, (5, 5))
            self.assertTrue(result.provisional)
            self.assertTrue(np.isfinite(result.clearance).all())

    def test_no_accessible_seed(self):
        for cell in ((-1, 0), (20, 20), (0, 0)):
            result = analyse_grid(np.full((5, 5), 100), 0.3, cell)
            self.assertIsNone(result.widest_cell)
            self.assertFalse(result.reachable.any())

    def test_rotated_origin_and_negative_coordinates(self):
        origin = (-12.3, 6.7, math.pi / 3, 0.3)
        for cell in ((0, 0), (4, 7), (-2, -3)):
            xy = cell_to_world(*cell, *origin)
            self.assertEqual(world_to_cell(*xy, *origin), cell)

    def test_resolution_scaling(self):
        grid = np.pad(np.zeros((9, 9)), 1, constant_values=100)
        result = analyse_grid(grid, 0.3, (5, 5), 0.1, 0.0)
        self.assertAlmostEqual(result.clearance[5, 5], (5 - 1 / math.sqrt(2)) * 0.3)

    def test_invalid_inputs(self):
        for resolution in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                analyse_grid(np.zeros((3, 3)), resolution, (1, 1))


class PassageTests(unittest.TestCase):
    def detect(self, grid, seed=(40, 25), **kwargs):
        result = analyse_grid(grid, 0.2, seed)
        return detect_passages(grid, result, 0.2, **kwargs)

    def corridor(self):
        grid = np.full((81, 141), 100)
        grid[35:46, 5:136] = 0  # 2.2 m wide, 26.2 m long
        return grid

    def test_known_width_corridor(self):
        skeleton, passages = self.detect(self.corridor())
        self.assertTrue(skeleton[40, 60])
        self.assertEqual(len(passages), 1)
        passage = passages[0]
        self.assertEqual(passage.kind, 'narrow_passage')
        self.assertEqual(passage.cell[0], 40)
        self.assertLess(abs(passage.width_m - 2.2), 0.2)
        self.assertTrue(passage.accessible)
        self.assertFalse(passage.provisional)

    def test_wide_room_does_not_trigger(self):
        grid = np.full((81, 141), 100)
        grid[10:71, 10:131] = 0
        lines, passages = self.detect(grid)
        self.assertEqual(passages, [])
        self.assertFalse(lines.any())

    def test_open_cavern_with_pillar_has_no_route_loop_overlay(self):
        grid = np.full((151, 151), 100)
        grid[5:146, 5:146] = 0
        grid[65:86, 65:86] = 100
        lines, passages = self.detect(grid, seed=(30, 30))
        self.assertEqual(passages, [])
        self.assertFalse(lines.any())

    def test_relative_bottleneck_above_absolute_threshold(self):
        grid = np.full((101, 151), 100)
        grid[10:91, 5:66] = 0
        grid[10:91, 85:146] = 0
        grid[40:61, 65:86] = 0  # 4.2 m neck between wide rooms
        lines, passages = self.detect(grid, seed=(50, 30))
        self.assertEqual(len(passages), 1)
        passage = passages[0]
        self.assertEqual(passage.kind, 'bottleneck')
        self.assertGreater(passage.width_m, 3.0)
        self.assertLess(abs(passage.width_m - 4.2), 0.2)
        self.assertTrue(65 <= passage.cell[1] <= 85)
        self.assertTrue(lines[passage.cell])
        self.assertFalse(lines[:, :40].any())
        self.assertFalse(lines[:, 110:].any())

    def test_unknown_wall_marks_provisional(self):
        grid = self.corridor()
        grid[:35, :] = -1
        self.assertTrue(self.detect(grid)[1][0].provisional)

    def test_short_section_filtered(self):
        self.assertEqual(self.detect(self.corridor(), min_length=20.0)[1], [])

    def test_unconnected_corridor_excluded(self):
        grid = self.corridor()
        grid[5:21, 5:31] = 0
        self.assertEqual(self.detect(grid, seed=(12, 15))[1], [])

    def test_single_cell_wall_noise(self):
        grid = self.corridor()
        grid[35, 70] = 100
        passages = self.detect(grid)[1]
        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0].kind, 'narrow_passage')
        self.assertLess(abs(passages[0].width_m - 2.2), 0.5)

    def test_too_narrow_neck_reported_as_inaccessible(self):
        grid = np.full((81, 141), 100)
        grid[10:71, 5:51] = 0
        grid[10:71, 90:136] = 0
        grid[38:43, 50:91] = 0
        passages = self.detect(grid)[1]
        self.assertTrue(passages)
        self.assertFalse(passages[0].accessible)

    def test_empty_reachable_space(self):
        grid = np.full((10, 10), -1)
        skeleton, passages = self.detect(grid, seed=(5, 5))
        self.assertFalse(skeleton.any())
        self.assertEqual(passages, [])

    def test_invalid_parameters(self):
        for kwargs in ({'narrow_width': 0}, {'bottleneck_ratio': 1},
                       {'unknown_margin': -1}, {'context_distance': float('nan')}):
            with self.assertRaises(ValueError):
                self.detect(self.corridor(), **kwargs)


if __name__ == '__main__':
    unittest.main()
