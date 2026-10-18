"""Regression tests for connectivity, budgets, exact queries and replay."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from advanced_functions.online_roadmap import OnlineRoadmap, Settings
from advanced_functions.roadmap_evaluation import GridReference, load_pairs, read_inputs


class ImprovementsTests(unittest.TestCase):
    def make(self, **kwargs):
        settings = dict(mode='uniform', robot_radius=0.1, safety_margin=0.05,
                        node_spacing=0.8, connection_radius=2., samples_per_update=4,
                        attempts_per_update=30, repair_attempts=4, coverage_samples=32)
        settings.update(kwargs)
        graph = OnlineRoadmap(Settings(**settings))
        graph.update(np.zeros((40,60)), (0.,0.,0.,0.2), (2.,2.))
        return graph

    def test_spacing_across_wall_does_not_suppress_node(self):
        graph=self.make(mode='baseline',node_spacing=2.)
        grid=np.zeros((40,60)); grid[:,20]=100
        graph.update(grid, graph.origin,(3.5,3.))
        # Two safe positions < spacing apart but separated by a wall.
        self.assertTrue(graph.add((4.7,3.),'sampled'))
        self.assertFalse(graph.edge_valid((3.5,3.),(4.7,3.)))

    def test_eviction_preserves_neighbour_connectivity_and_protects_bridge(self):
        graph=self.make(mode='baseline', max_nodes=4)
        graph.nodes={0:(2.,2.),1:(4.,2.),2:(4.,3.),3:(5.,2.)}
        graph.sources={0:'travelled',1:'sampled',2:'sampled',3:'sampled'}
        graph.edges={(0,1):2.,(1,2):1.,(1,3):1.}
        self.assertFalse(graph.make_room())  # Star centre is an articulation node.
        graph.edges.update({(0,2):math.sqrt(5),(0,3):3.,(2,3):math.sqrt(2)})
        self.assertTrue(graph.make_room())
        self.assertIn(0,graph.nodes)
        self.assertEqual(len(set(graph.components().values())),1)

    def test_budget_replacement_can_represent_later_region(self):
        graph=self.make(mode='baseline',max_nodes=4)
        graph.nodes={0:(2.,2.),1:(4.,2.),2:(4.,3.),3:(5.,2.)}
        graph.sources={0:'travelled',1:'sampled',2:'sampled',3:'sampled'}
        graph.next_id=4
        graph.edges={(a,b):math.dist(graph.nodes[a],graph.nodes[b]) for a in graph.nodes for b in graph.nodes if a<b}
        graph.groups=graph.components()
        self.assertTrue(graph.add((8.,3.),'sampled',priority=True))
        self.assertLessEqual(len(graph.nodes),4)
        self.assertIn((8.,3.),graph.nodes.values())
        self.assertEqual(graph.replacements,1)

    def test_exact_endpoints_and_unreachable_goal_not_snapped(self):
        graph=self.make(mode='baseline',connection_radius=4.)
        graph.update(graph.grid,graph.origin,(4.,2.))
        start,goal=(1.9,2.1),(4.1,2.1)
        path,status=graph.query_route(start,goal)
        self.assertEqual(status,'ok')
        self.assertEqual(path[0],start); self.assertEqual(path[-1],goal)
        grid=graph.grid.copy(); grid[:,30]=100
        graph.update(grid,graph.origin,(2.,2.))
        self.assertFalse(graph.query_route((2.,2.),(7.,2.))[0])
        self.assertEqual(graph.query_route((2.,2.),(6.1,2.))[1],'unsafe_endpoint')

    def test_grid_reference_open_room_and_blocked_goal(self):
        graph=self.make(mode='baseline')
        reference=GridReference(graph)
        distance=reference.query((2.1,2.1),(4.1,2.1))
        self.assertAlmostEqual(distance,2.,places=5)
        self.assertIsNone(reference.query((2.,2.),(-1.,2.)))

    def test_edge_clearance_against_independent_wall_geometry(self):
        graph=self.make(mode='baseline')
        grid=np.zeros((40,60)); grid[:,30:32]=100
        graph.update(grid,graph.origin,(2.,2.))
        rng=np.random.default_rng(4)
        for _ in range(80):
            a,b=rng.uniform([1.,1.],[11.,7.],size=(2,2))
            if graph.edge_valid(a,b):
                # Analytic separation from the vertical occupied slab [6,6.4].
                low,high=sorted((a[0],b[0]))
                distance=max(6.-high,low-6.4,0.)
                self.assertGreaterEqual(distance,0.15)

    def test_isolation_grace_period_and_diagnostics(self):
        graph=self.make(mode='baseline',isolation_patience=2)
        graph.add((10.,6.),'sampled')
        first=graph.update(graph.grid,graph.origin,(2.,2.))
        self.assertEqual(first['isolated_nodes'],2)
        self.assertEqual(first['isolated_diagnostics'][-1]['reason'],'no_nearby_nodes')
        second=graph.update(graph.grid,graph.origin,(2.,2.))
        self.assertEqual(second['retired_isolates'],1)
        self.assertEqual(len(graph.nodes),1)  # Travelled node is retained.

    def test_replay_and_pair_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            pair=path/'pairs.json'
            pair.write_text(json.dumps([{'name':'doorway','start':[1,2],'goal':[3,4]}]))
            self.assertEqual(load_pairs(pair)[0]['name'],'doorway')
            for index in range(2):
                np.savez_compressed(path/f'{index:06d}.npz', grid=np.zeros((5,5)),
                    origin=[0.,0.,0.,0.2],robot=[1.,1.],frame='map',time_ns=index*2_000_000_000,
                    settings=json.dumps({'mode':'baseline'}))
            snapshots=list(read_inputs(path))
            self.assertEqual(len(snapshots),2)
            self.assertEqual(snapshots[1][3],2.)


if __name__=='__main__':
    unittest.main()
