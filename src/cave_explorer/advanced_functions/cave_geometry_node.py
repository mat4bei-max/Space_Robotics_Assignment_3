"""Online clearance and widest-accessible-disc visualisation for Advanced 2."""

import json
import math
from copy import deepcopy

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray

from advanced_functions.cave_geometry import (
    analyse_grid, cell_to_world, world_to_cell, detect_passages,
)


PASSAGE_DEFAULTS = {'narrow_width': 3.0, 'min_length': 1.2, 'context_distance': 6.0,
                    'bottleneck_ratio': 0.75, 'min_contrast': 0.6,
                    'merge_distance': 2.0, 'unknown_margin': 0.6}


class CaveGeometryNode(Node):
    def __init__(self):
        super().__init__('cave_geometry')
        defaults = {'robot_frame': 'base_link', 'robot_radius': 0.5,
                    'safety_margin': 0.1, 'free_threshold': 25,
                    'update_period': 2.0, 'display_max_clearance': 10.0}
        defaults.update(PASSAGE_DEFAULTS)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.settings = {name: self.get_parameter(name).value for name in defaults}
        for name in ('update_period', 'display_max_clearance'):
            if not math.isfinite(self.settings[name]) or self.settings[name] <= 0:
                raise ValueError(f'{name} must be positive and finite')
        # Validate geometry parameters before waiting for a map.
        validation = analyse_grid(np.zeros((1, 1)), 1.0, (0, 0),
                     self.settings['robot_radius'], self.settings['safety_margin'],
                     self.settings['free_threshold'])
        self.passage_settings = {key: self.settings[key] for key in PASSAGE_DEFAULTS}
        detect_passages(np.zeros((1, 1)), validation, 1.0, **self.passage_settings)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.clearance_pub = self.create_publisher(OccupancyGrid, 'cave_geometry/clearance', qos)
        self.marker_pub = self.create_publisher(MarkerArray, 'cave_geometry/features', qos)
        self.summary_pub = self.create_publisher(String, 'cave_geometry/summary', qos)
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.on_map, qos)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_msg = None
        self.create_timer(self.settings['update_period'], self.update)

    def on_map(self, msg):
        self.map_msg = msg

    def clear_features(self, status):
        marker = Marker()
        marker.action = Marker.DELETEALL
        self.marker_pub.publish(MarkerArray(markers=[marker]))
        self.summary_pub.publish(String(data=json.dumps(
            {'status': status, 'widest': None, 'passages': []})))

    def update(self):
        msg = self.map_msg
        if msg is None:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                msg.header.frame_id, self.settings['robot_frame'], rclpy.time.Time())
        except TransformException as exc:
            self.clear_features('waiting_for_robot_transform')
            self.get_logger().warning(str(exc), throttle_duration_sec=10.0)
            return
        info = msg.info
        q = info.origin.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        origin = (info.origin.position.x, info.origin.position.y, yaw, info.resolution)
        try:
            grid = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
            if info.resolution <= 0:
                raise ValueError('invalid map resolution')
            position = transform.transform.translation
            cell = world_to_cell(position.x, position.y, *origin)
            result = analyse_grid(grid, info.resolution, cell,
                                  self.settings['robot_radius'], self.settings['safety_margin'],
                                  self.settings['free_threshold'])
            skeleton, passages = detect_passages(
                grid, result, info.resolution, self.settings['free_threshold'],
                **self.passage_settings)
        except (ValueError, OverflowError) as exc:
            self.clear_features('invalid_map')
            self.get_logger().error(str(exc), throttle_duration_sec=10.0)
            return
        display = OccupancyGrid()
        display.header = msg.header
        # Separate the visual overlay from the SLAM map plane to avoid z-fighting.
        # Copy metadata so repeated updates cannot move the source map itself.
        display.info = deepcopy(info)
        display.info.origin.position.z += 0.05
        values = np.rint(np.clip(result.clearance / self.settings['display_max_clearance'],
                                  0, 1) * 100).astype(np.int8)
        values[grid < 0] = -1
        display.data = values.ravel().tolist()
        self.clearance_pub.publish(display)
        if result.widest_cell is None:
            self.clear_features('no_accessible_space_at_robot')
            return
        x, y = cell_to_world(*result.widest_cell, *origin)
        radius = float(result.clearance[result.widest_cell])
        # Each latched update is a complete snapshot, including removal of stale IDs.
        delete = Marker()
        delete.action = Marker.DELETEALL
        markers = [delete]
        for ident, kind in ((0, Marker.LINE_STRIP), (1, Marker.TEXT_VIEW_FACING)):
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'widest_accessible_area'
            marker.id, marker.type = ident, kind
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.2, 1.0, 0.3, 1.0
            if ident == 0:
                marker.scale.x = 0.08
                marker.points = [Point(x=x + radius * math.cos(t),
                                       y=y + radius * math.sin(t), z=0.12)
                                 for t in np.linspace(0, 2 * math.pi, 97)]
            else:
                marker.pose.position = Point(x=x, y=y, z=0.6)
                marker.scale.z = 0.5
                suffix = ' (provisional)' if result.provisional else ''
                marker.text = f'Widest accessible disc: {2 * radius:.2f} m{suffix}'
            markers.append(marker)
        centreline = Marker()
        centreline.header = markers[-1].header
        centreline.ns = 'cave_centreline'
        centreline.id = 0
        centreline.type = Marker.POINTS
        centreline.action = Marker.ADD
        centreline.pose.orientation.w = 1.0
        centreline.scale.x = centreline.scale.y = info.resolution * 0.5
        centreline.color.r, centreline.color.g, centreline.color.b, centreline.color.a = 0.2, 0.9, 1.0, 0.8
        centreline.points = [Point(x=px, y=py, z=0.16)
                             for px, py in (cell_to_world(int(r), int(c), *origin)
                                            for r, c in np.argwhere(skeleton))]
        markers.append(centreline)
        passage_summaries = []
        for index, passage in enumerate(passages):
            px, py = cell_to_world(*passage.cell, *origin)
            for offset, kind in ((0, Marker.SPHERE), (1, Marker.TEXT_VIEW_FACING)):
                marker = Marker()
                marker.header = centreline.header
                marker.ns = 'narrow_passages'
                marker.id = index * 2 + offset
                marker.type, marker.action = kind, Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.pose.position = Point(x=px, y=py, z=0.25 if offset == 0 else 0.9)
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.45, 0.05, 1.0
                if passage.provisional:
                    marker.color.g = 0.9
                if offset == 0:
                    marker.scale.x = marker.scale.y = marker.scale.z = 0.35
                else:
                    marker.scale.z = 0.4
                    label = 'Bottleneck' if passage.kind == 'bottleneck' else 'Narrow passage'
                    marker.text = f'{label}: ~{passage.width_m:.2f} m'
                    if passage.provisional:
                        marker.text += ' (provisional)'
                    if not passage.accessible:
                        marker.text += ' (outside reachable space)'
                markers.append(marker)
            passage_summaries.append({
                'x': px, 'y': py, 'width_m': passage.width_m, 'kind': passage.kind,
                'provisional': passage.provisional, 'accessible': passage.accessible,
                'section_length_m': passage.length_m})
        self.marker_pub.publish(MarkerArray(markers=markers))
        summary = {'status': 'ok', 'frame': msg.header.frame_id,
                   'map_stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
                   'widest': {'x': x, 'y': y, 'clearance_m': radius,
                              'diameter_m': 2 * radius, 'provisional': result.provisional},
                   'accessible_area_m2': float(result.reachable.sum() * info.resolution ** 2),
                   'passages': passage_summaries}
        self.summary_pub.publish(String(data=json.dumps(summary)))
        self.get_logger().info(
            f'Widest accessible disc at ({x:.2f}, {y:.2f}): diameter {2 * radius:.2f} m'
            + (' (limited by unexplored space)' if result.provisional else '')
            + f'; {len(passages)} narrow passage candidates',
            throttle_duration_sec=10.0)


def main(args=None):
    rclpy.init(args=args)
    node = CaveGeometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
