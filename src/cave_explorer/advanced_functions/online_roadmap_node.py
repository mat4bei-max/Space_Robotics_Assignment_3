"""ROS adapter for online roadmap construction and matched baseline comparison."""

import colorsys
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray

from advanced_functions.online_roadmap import OnlineRoadmap, Settings


class OnlineRoadmapNode(Node):
    def __init__(self):
        super().__init__('online_roadmap')
        defaults = dict(asdict(Settings()), robot_frame='base_link', update_period=2.0,
                        metrics_file='', record_directory='', evaluation_pairs_file='', compare_uniform=True)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.config = {name: self.get_parameter(name).value for name in defaults}
        settings = Settings(**{key: self.config[key] for key in asdict(Settings())})
        if not math.isfinite(self.config['update_period']) or self.config['update_period'] <= 0:
            raise ValueError('update_period must be positive and finite')
        self.graphs = {'selected': OnlineRoadmap(settings)}
        if settings.mode != 'baseline':
            self.graphs['baseline'] = OnlineRoadmap(replace(settings, mode='baseline'))
        if self.config['compare_uniform'] and settings.mode != 'uniform':
            self.graphs['uniform'] = OnlineRoadmap(replace(settings, mode='uniform'))
        self.evaluation_pairs = []
        if self.config['evaluation_pairs_file']:
            from advanced_functions.roadmap_evaluation import load_pairs
            self.evaluation_pairs = load_pairs(self.config['evaluation_pairs_file'])
        self.record_session = uuid4().hex[:12]
        self.record_index = 0
        self.settings = settings
        self.map_msg = None
        self.frame = None
        self.last_time = None
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.publishers_by_graph = {name: self.create_publisher(
            MarkerArray, f'online_roadmap/{name}', qos) for name in self.graphs}
        self.summary_pub = self.create_publisher(String, 'online_roadmap/summary', qos)
        self.subscription = self.create_subscription(OccupancyGrid, 'map', self.on_map, qos)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.create_timer(self.config['update_period'], self.update)

    def on_map(self, msg):
        self.map_msg = msg

    def clear(self, status):
        for publisher in self.publishers_by_graph.values():
            publisher.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        self.summary_pub.publish(String(data=json.dumps({'status': status})))

    def update(self):
        msg = self.map_msg
        if msg is None:
            return
        now = self.get_clock().now()
        cycle_start = perf_counter()
        if self.frame != msg.header.frame_id or (self.last_time is not None and now < self.last_time):
            self.graphs = {name: OnlineRoadmap(graph.settings) for name, graph in self.graphs.items()}
            self.record_session = uuid4().hex[:12]
            self.record_index = 0
        self.frame, self.last_time = msg.header.frame_id, now
        try:
            tf = self.buffer.lookup_transform(self.frame, self.config['robot_frame'], rclpy.time.Time())
        except TransformException as exc:
            self.clear('waiting_for_robot_transform')
            self.get_logger().warning(str(exc), throttle_duration_sec=10.0)
            return
        q = msg.info.origin.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        origin = (msg.info.origin.position.x, msg.info.origin.position.y, yaw, msg.info.resolution)
        robot = (tf.transform.translation.x, tf.transform.translation.y)
        summary = {'status': 'ok', 'frame': self.frame, 'time_ns': now.nanoseconds,
                   'map_stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
                   'settings': asdict(self.settings), 'graphs': {}}
        try:
            grid = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
            for name, graph in self.graphs.items():
                start = perf_counter()
                metrics = graph.update(grid, origin, robot)
                metrics['update_ms'] = (perf_counter()-start)*1000
                metrics['evaluation_routes'] = []
                for pair in self.evaluation_pairs:
                    path, status = graph.query_route(pair['start'], pair['goal'])
                    metrics['evaluation_routes'].append({'name': pair['name'], 'status': status,
                        'length_m': sum(math.dist(a,b) for a,b in zip(path,path[1:])) if path else None})
                markers, route_length = self.make_markers(graph, now.to_msg())
                metrics['example_route_length_m'] = route_length
                summary['graphs'][name] = metrics
                self.publishers_by_graph[name].publish(markers)
        except (ValueError, OverflowError) as exc:
            self.clear('invalid_map')
            self.get_logger().error(str(exc), throttle_duration_sec=10.0)
            return
        if self.config['record_directory']:
            try:
                directory = Path(self.config['record_directory']).expanduser()/self.record_session
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory/f'{self.record_index:06d}.npz', grid=grid,
                    origin=np.asarray(origin), robot=np.asarray(robot), time_ns=now.nanoseconds,
                    frame=self.frame, settings=json.dumps(asdict(self.settings)))
                self.record_index += 1
            except OSError as exc:
                self.get_logger().error(f'Cannot record roadmap input: {exc}', throttle_duration_sec=10.0)
        summary['processing_ms'] = (perf_counter()-cycle_start)*1000
        encoded = json.dumps(summary)
        self.summary_pub.publish(String(data=encoded))
        if self.config['metrics_file']:
            try:
                with Path(self.config['metrics_file']).expanduser().open('a') as stream:
                    stream.write(encoded + '\n')
            except OSError as exc:
                self.get_logger().error(f'Cannot save roadmap metrics: {exc}', throttle_duration_sec=10.0)

    def make_markers(self, graph, stamp):
        def marker(namespace, kind):
            m = Marker()
            m.header.frame_id, m.header.stamp = self.frame, stamp
            m.ns, m.type, m.action = namespace, kind, Marker.ADD
            m.pose.orientation.w = 1.0
            m.color.a = 1.0
            return m

        def point(node):
            x, y = graph.nodes[node]
            return Point(x=x, y=y, z=0.2)

        isolated = {i for i, neighbours in graph.adjacency().items() if not neighbours}

        def colour(node):
            if node in isolated:
                return ColorRGBA(r=1.0, g=0.05, b=0.05, a=1.0)
            rgb = colorsys.hsv_to_rgb((graph.groups[node]*0.61803398875) % 1.0, 0.65, 1.0)
            return ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=1.0)

        markers = [Marker(action=Marker.DELETEALL)]
        for source, size in (('travelled', 0.28), ('sampled', 0.16), ('repair', 0.2)):
            m = marker(source, Marker.SPHERE_LIST)
            m.scale.x = m.scale.y = m.scale.z = size
            for node in graph.nodes:
                if graph.sources[node] == source:
                    m.points.append(point(node))
                    m.colors.append(colour(node))
            markers.append(m)
        edges = marker('connections', Marker.LINE_LIST)
        edges.scale.x = 0.035
        for a, b in graph.edges:
            edges.points.extend([point(a), point(b)])
            edges.colors.extend([colour(a), colour(b)])
        markers.append(edges)
        # Automatic diagnostic route: does not issue navigation goals.
        anchor = graph.robot_anchor()
        length = None
        if anchor is not None:
            candidates = [i for i in graph.nodes if graph.groups[i] == graph.groups[anchor]]
            goal = max(candidates, key=lambda i: math.dist(graph.nodes[anchor], graph.nodes[i]))
            path = graph.route(anchor, goal)
            if len(path) > 1:
                route = marker('example_astar_route', Marker.LINE_STRIP)
                route.color.r = route.color.g = route.color.b = 1.0
                route.scale.x = 0.09
                route.points = [Point(x=graph.nodes[i][0], y=graph.nodes[i][1], z=0.3) for i in path]
                markers.append(route)
                length = sum(math.dist(graph.nodes[a], graph.nodes[b]) for a, b in zip(path, path[1:]))
        label = marker('summary', Marker.TEXT_VIEW_FACING)
        label.pose.position = Point(x=graph.robot[0], y=graph.robot[1], z=1.2)
        label.scale.z = 0.35
        label.color.r = label.color.g = label.color.b = 1.0
        label.text = f'{graph.settings.mode}: {len(graph.nodes)} nodes, {len(isolated)} isolated'
        markers.append(label)
        for index, pair in enumerate(self.evaluation_pairs):
            path, status = graph.query_route(pair['start'], pair['goal'])
            m = marker('evaluation_routes', Marker.LINE_STRIP if path else Marker.TEXT_VIEW_FACING)
            m.id = index
            m.color.r, m.color.g = (0.1, 1.0) if path else (1.0, 0.1)
            if path:
                m.scale.x = 0.11
                m.points = [Point(x=float(p[0]), y=float(p[1]), z=0.4) for p in path]
            else:
                m.scale.z = 0.4
                m.pose.position = Point(x=float(pair['goal'][0]), y=float(pair['goal'][1]), z=0.7)
                m.text = pair['name']+': '+status
            markers.append(m)
        return MarkerArray(markers=markers), length


def main(args=None):
    rclpy.init(args=args)
    node = OnlineRoadmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
