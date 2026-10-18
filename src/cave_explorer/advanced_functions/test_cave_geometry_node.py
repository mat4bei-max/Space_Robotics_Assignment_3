"""Message-level integration checks; source ROS Humble before running."""

import json
import unittest
from unittest.mock import Mock

import numpy as np

try:
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.serialization import serialize_message
    from rclpy.time import Time
    from visualization_msgs.msg import Marker
    from advanced_functions.cave_geometry_node import CaveGeometryNode, PASSAGE_DEFAULTS
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


@unittest.skipUnless(ROS_AVAILABLE, 'ROS Python environment required')
class GeometryNodeTests(unittest.TestCase):
    def test_passages_publish_and_stale_markers_are_deleted(self):
        node = Mock()
        node.settings = {'robot_frame': 'base_link', 'robot_radius': 0.5,
                         'safety_margin': 0.1, 'free_threshold': 25,
                         'display_max_clearance': 10.0}
        node.passage_settings = PASSAGE_DEFAULTS.copy()
        node.get_clock.return_value.now.return_value = Time()
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.width, msg.info.height = 141, 81
        msg.info.resolution = 0.2
        msg.info.origin.orientation.w = 1.0
        grid = np.full((81, 141), 100)
        grid[35:46, 5:136] = 0
        msg.data = grid.ravel().tolist()
        node.map_msg = msg
        transform = TransformStamped()
        transform.transform.translation.x = 5.1
        transform.transform.translation.y = 8.1
        node.tf_buffer.lookup_transform.return_value = transform
        CaveGeometryNode.update(node)
        display = node.clearance_pub.publish.call_args.args[0]
        self.assertAlmostEqual(display.info.origin.position.z, 0.05)
        self.assertEqual(msg.info.origin.position.z, 0.0)
        self.assertEqual(display.info.origin.position.x, msg.info.origin.position.x)
        self.assertEqual(display.info.origin.position.y, msg.info.origin.position.y)
        summary = json.loads(node.summary_pub.publish.call_args.args[0].data)
        self.assertEqual(len(summary['passages']), 1)
        for pub in (node.clearance_pub, node.marker_pub, node.summary_pub):
            serialize_message(pub.publish.call_args.args[0])
        markers = node.marker_pub.publish.call_args.args[0].markers
        self.assertEqual(markers[0].action, Marker.DELETEALL)
        self.assertTrue(any(m.ns == 'narrow_passages' for m in markers))
        grid[10:71, 5:136] = 0
        msg.data = grid.ravel().tolist()
        CaveGeometryNode.update(node)
        self.assertAlmostEqual(
            node.clearance_pub.publish.call_args.args[0].info.origin.position.z, 0.05)
        self.assertEqual(msg.info.origin.position.z, 0.0)
        summary = json.loads(node.summary_pub.publish.call_args.args[0].data)
        self.assertEqual(summary['passages'], [])
        markers = node.marker_pub.publish.call_args.args[0].markers
        self.assertEqual(markers[0].action, Marker.DELETEALL)
        self.assertFalse(any(m.ns == 'narrow_passages' for m in markers))
        CaveGeometryNode.clear_features(node, 'no_accessible_space_at_robot')
        summary = json.loads(node.summary_pub.publish.call_args.args[0].data)
        self.assertEqual(summary['passages'], [])
        self.assertIsNone(summary['widest'])


if __name__ == '__main__':
    unittest.main()
