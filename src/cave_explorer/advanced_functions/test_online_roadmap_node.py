"""ROS message serialization and adapter checks, without a running DDS network."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

try:
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.serialization import serialize_message
    from rclpy.time import Time
    from tf2_ros import TransformException
    from visualization_msgs.msg import Marker
    from advanced_functions.online_roadmap import OnlineRoadmap, Settings
    from advanced_functions.online_roadmap_node import OnlineRoadmapNode
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


@unittest.skipUnless(ROS_AVAILABLE, 'Source ROS Humble and use /usr/bin/python3')
class RoadmapNodeTests(unittest.TestCase):
    def make_node(self):
        node = Mock()
        node.settings = Settings(robot_radius=0.1, safety_margin=0.05)
        node.graphs = {'selected': OnlineRoadmap(node.settings),
                       'baseline': OnlineRoadmap(Settings(mode='baseline', robot_radius=0.1,
                                                         safety_margin=0.05))}
        node.publishers_by_graph = {name: Mock() for name in node.graphs}
        node.config = {'robot_frame': 'base_link', 'metrics_file': '', 'record_directory': ''}
        node.evaluation_pairs = []
        node.frame, node.last_time = None, None
        node.get_clock.return_value.now.return_value = Time(seconds=10)
        node.make_markers = lambda graph, stamp: OnlineRoadmapNode.make_markers(node, graph, stamp)
        node.clear = lambda status: OnlineRoadmapNode.clear(node, status)
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.width = msg.info.height = 40
        msg.info.resolution = 0.25
        msg.info.origin.orientation.w = 1.0
        msg.data = np.zeros((40, 40), dtype=int).ravel().tolist()
        node.map_msg = msg
        tf = TransformStamped()
        tf.transform.translation.x = tf.transform.translation.y = 2.0
        node.buffer.lookup_transform.return_value = tf
        return node

    def test_update_serializes_matched_graphs_and_clears_stale_markers(self):
        node = self.make_node()
        OnlineRoadmapNode.update(node)
        summary = json.loads(node.summary_pub.publish.call_args.args[0].data)
        self.assertEqual(summary['graphs']['baseline']['nodes'], 1)
        self.assertGreater(summary['graphs']['selected']['nodes'], 1)
        serialize_message(node.summary_pub.publish.call_args.args[0])
        for publisher in node.publishers_by_graph.values():
            markers = publisher.publish.call_args.args[0]
            serialize_message(markers)
            self.assertEqual(markers.markers[0].action, Marker.DELETEALL)
        node.buffer.lookup_transform.side_effect = TransformException('missing TF')
        OnlineRoadmapNode.update(node)
        summary = json.loads(node.summary_pub.publish.call_args.args[0].data)
        self.assertEqual(summary['status'], 'waiting_for_robot_transform')
        for publisher in node.publishers_by_graph.values():
            self.assertEqual(len(publisher.publish.call_args.args[0].markers), 1)

    def test_records_inputs_and_labels_failed_fixed_query(self):
        node = self.make_node()
        node.evaluation_pairs = [{'name': 'blocked', 'start': [2., 2.], 'goal': [-3., 2.]}]
        with tempfile.TemporaryDirectory() as directory:
            node.config['record_directory'] = directory
            OnlineRoadmapNode.update(node)
            files = list(Path(directory).glob('*/*.npz'))
            self.assertEqual(len(files), 1)
            with np.load(files[0], allow_pickle=False) as data:
                self.assertEqual(data['grid'].shape, (40, 40))
                self.assertEqual(tuple(data['robot']), (2., 2.))
            summary = json.loads(node.summary_pub.publish.call_args.args[0].data)
            self.assertEqual(summary['graphs']['selected']['evaluation_routes'][0]['status'], 'unsafe_endpoint')
            markers = node.publishers_by_graph['selected'].publish.call_args.args[0]
            self.assertTrue(any(m.ns == 'evaluation_routes' and 'unsafe_endpoint' in m.text for m in markers.markers))
            serialize_message(markers)

    def test_frame_change_resets_persistent_nodes(self):
        node = self.make_node()
        OnlineRoadmapNode.update(node)
        original = node.graphs['selected']
        node.map_msg.header.frame_id = 'new_map'
        OnlineRoadmapNode.update(node)
        self.assertIsNot(node.graphs['selected'], original)
        self.assertEqual(node.frame, 'new_map')


if __name__ == '__main__':
    unittest.main()
