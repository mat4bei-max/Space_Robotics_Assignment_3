"""Behavioural checks for online roadmap safety, persistence and sampling."""
import math
import unittest

import numpy as np

from advanced_functions.online_roadmap import OnlineRoadmap, Settings
from advanced_functions.cave_geometry import cell_to_world


class RoadmapTests(unittest.TestCase):
    def graph(self, mode='distance_guided', **kwargs):
        return OnlineRoadmap(Settings(mode=mode, robot_radius=0.1, safety_margin=0.05,
                                      node_spacing=0.6, connection_radius=3.0, **kwargs))

    def test_observed_unvisited_space_and_matched_baseline(self):
        grid = np.zeros((40, 40), dtype=int)
        origin = (0., 0., 0., 0.25)
        baseline, improved = self.graph('baseline'), self.graph()
        b = baseline.update(grid, origin, (2., 2.))
        p = improved.update(grid, origin, (2., 2.))
        self.assertEqual(b['nodes'], 1)
        self.assertGreater(p['nodes'], 5)
        self.assertGreater(p['node_proximity_coverage'], b['node_proximity_coverage'])
        self.assertTrue(any(math.dist(point, (2., 2.)) > 3 for point in improved.nodes.values()))
        self.assertTrue(all(improved.edge_valid(improved.nodes[a], improved.nodes[b])
                            for a, b in improved.edges))

    def test_wall_unknown_and_corner_crossing(self):
        grid = np.zeros((30, 30), dtype=int)
        grid[:, 15] = 100
        graph = self.graph('baseline')
        graph.update(grid, (0., 0., 0., 0.25), (2., 2.))
        self.assertFalse(graph.edge_valid((2., 2.), (5., 2.)))
        grid[:, 15] = -1
        graph.update(grid, graph.origin, (2., 2.))
        self.assertFalse(graph.edge_valid((2., 2.), (5., 2.)))
        grid[:] = 0
        grid[12, 12] = 100
        graph.update(grid, graph.origin, (2., 2.))
        self.assertFalse(graph.edge_valid((2., 4.), (4., 2.)))

    def test_revalidate_obstacle_and_preserve_world_nodes_on_resize(self):
        graph = self.graph('baseline')
        grid = np.zeros((40, 40), dtype=int)
        origin = (0., 0., 0., 0.25)
        graph.update(grid, origin, (2., 2.))
        graph.update(grid, origin, (4., 2.))
        self.assertEqual(len(graph.edges), 1)
        nodes = dict(graph.nodes)
        graph.update(np.zeros((48, 48)), (-1., -1., 0., 0.25), (4., 2.))
        self.assertEqual(graph.nodes, nodes)
        grid[:, 12] = 100
        graph.update(grid, origin, (4., 2.))
        self.assertFalse(graph.edges)
        self.assertEqual(len(set(graph.groups.values())), 2)
        self.assertEqual(graph.route(0, 1), [])
        grid[:, 12] = 0
        graph.update(grid, origin, (4., 2.))
        self.assertEqual(graph.route(0, 1), [0, 1])
        grid[8, 8] = 100
        result = graph.update(grid, origin, (4., 2.))
        self.assertNotIn(0, graph.nodes)
        self.assertEqual(result['removed_nodes'], 1)

    def test_rotated_map_and_bounded_deterministic_sampling(self):
        origin = (10., -5., math.pi/3, 0.25)
        grid = np.zeros((30, 30), dtype=int)
        robot = cell_to_world(10, 10, *origin)
        first, second = self.graph(max_nodes=12), self.graph(max_nodes=12)
        for _ in range(4):
            first.update(grid, origin, robot)
            second.update(grid, origin, robot)
        self.assertEqual(first.nodes, second.nodes)
        self.assertEqual(first.edges, second.edges)
        self.assertLessEqual(len(first.nodes), 12)
        self.assertTrue(first.edge_valid(robot, cell_to_world(12, 12, *origin)))

    def test_blocked_map_and_invalid_settings(self):
        graph = self.graph()
        result = graph.update(np.full((20, 20), -1), (0., 0., 0., 0.25), (2., 2.))
        self.assertEqual(result['nodes'], 0)
        self.assertEqual(result['node_proximity_coverage'], 0.)
        for kwargs in ({'mode': 'bad'}, {'node_spacing': 0}, {'samples_per_update': -1}):
            with self.assertRaises(ValueError):
                Settings(**kwargs)

    def test_astar_selects_shorter_route(self):
        graph = self.graph()
        graph.nodes = {0: (0., 0.), 1: (1., 0.), 2: (1., 1.), 3: (2., 0.)}
        graph.edges = {(0, 1): 1., (1, 3): 1., (0, 2): math.sqrt(2), (2, 3): math.sqrt(2)}
        self.assertEqual(graph.route(0, 3), [0, 1, 3])


if __name__ == '__main__':
    unittest.main()
