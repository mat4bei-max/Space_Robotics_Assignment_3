#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from enum import Enum
from pathlib import Path

import uuid
import cv2  # OpenCV2
import rclpy
import message_filters
import numpy as np
from sklearn.cluster import DBSCAN
from cv_bridge import CvBridge
from std_msgs.msg import Empty
from yolo_msgs.msg import DetectionArray
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


def wrap_angle(angle):
    """Function to wrap an angle between 0 and 2*Pi"""
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def pose2d_to_pose(pose_2d):
    """Convert a Pose2D to a full 3D Pose"""
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose


class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    # Add more!


class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer_node')

        # Variables/Flags for mapping
        self.xlim_ = [0.0, 0.0]
        self.ylim_ = [0.0, 0.0]

        # Variables/Flags for perception
        self.artifact_found_ = False

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False

        # Marker for artifact locations
        # See https://wiki.ros.org/rviz/DisplayTypes/Marker
        self.marker_artifacts_ = Marker()
        self.marker_artifacts_.header.frame_id = "map"
        self.marker_artifacts_.ns = "artifacts"
        self.marker_artifacts_.id = 0
        self.marker_artifacts_.type = Marker.SPHERE_LIST
        self.marker_artifacts_.action = Marker.ADD
        self.marker_artifacts_.pose.position.x = 0.0
        self.marker_artifacts_.pose.position.y = 0.0
        self.marker_artifacts_.pose.position.z = 0.0
        self.marker_artifacts_.pose.orientation.x = 0.0
        self.marker_artifacts_.pose.orientation.y = 0.0
        self.marker_artifacts_.pose.orientation.z = 0.0
        self.marker_artifacts_.pose.orientation.w = 1.0
        self.marker_artifacts_.scale.x = 1.5
        self.marker_artifacts_.scale.y = 1.5
        self.marker_artifacts_.scale.z = 1.5
        self.marker_artifacts_.color.a = 1.0
        self.marker_artifacts_.color.r = 0.0
        self.marker_artifacts_.color.g = 1.0
        self.marker_artifacts_.color.b = 0.2
        self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_artifacts', 10)

        # Remember the artifact locations
        # Array of type geometry_msgs.Point
        self.artifact_locations_ = []

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Prepare transformation to get robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Action client for nav2
        self.nav2_action_client_ = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().warn('Waiting for navigate_to_pose action...')
        self.nav2_action_client_.wait_for_server()
        self.get_logger().warn('navigate_to_pose connected')
        self.ready_for_next_goal_ = True
        self.declare_parameter('print_feedback', rclpy.Parameter.Type.BOOL)

        # Publisher for the goal pose visualisation
        self.goal_pose_vis_ = self.create_publisher(PoseStamped, 'goal_pose', 1)

        # Subscribe to the map topic to get current bounds
        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map',  self.map_callback, 1)

        # Prepate image processing
        self.rgb_image_sub = message_filters.Subscriber(self, Image, 'camera/image', 1)
        self.depth_image_sub = message_filters.Subscriber(self, Image, '/camera/depth/image', 1)
        self.synchronizer = message_filters.ApproximateTimeSynchronizer([self.rgb_image_sub, self.depth_image_sub], queue_size=10, slop=0.05)
        self.synchronizer.registerCallback(self.image_callback)

        # Prepare artifact detection
        self.detection_images_ = {}
        self.detection_every_n_images_ = self.declare_parameter('detection_every_n_images', 1).value
        self.depth_detection_ = self.declare_parameter('use_depth_detection', False).value
        self.detection_image_count_ = 0
        self.latest_artifact_detections_ = None
        self.detection_image_pub_ = self.create_publisher(Image, '/artifact_detection/image', 1)
        self.image_detections_pub_ = self.create_publisher(Image, '/detections_image', 1)
        self.artifact_detections_sub_ = self.create_subscription(DetectionArray, '/yolo/detections', self.artifact_detections_callback, 1,)

        # Prepare image capture
        data_directory = self.declare_parameter('data_directory', '').value
        self.activate_rgb_capture_ = self.declare_parameter('activate_rgb_capture', False).value
        self.activate_depth_capture_ = self.declare_parameter('activate_depth_capture', False).value
        self.activate_processed_depth_capture_ = self.declare_parameter('activate_processed_depth_capture', False).value
        self.rgb_capture_directory_ =   Path(data_directory) / Path(self.declare_parameter('rgb_image_capture_directory', '/tmp/rgb').value)
        self.depth_capture_directory_ = Path(data_directory) / Path(self.declare_parameter('depth_image_capture_directory', '/tmp/depth').value)
        self.processed_depth_capture_directory_ = Path(data_directory) / Path(self.declare_parameter('processed_depth_image_capture_directory', '/tmp/processed_depth').value)
        self.capture_requested_ = False
        self.capture_sub_ = self.create_subscription(Empty, '/capture_image', self.capture_callback, 10)

        # Timer for main loop
        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)
    
    def get_pose_2d(self):
        """
        Get the 2d pose of the robot.
        """

        # Lookup the transform
        try:
            t = self.tf_buffer.lookup_transform_async(
                'map',
                'base_link',
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().error(f'Could not transform: {ex}')
            return

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = t.transform.translation.x
        pose.y = t.transform.translation.y

        qw = t.transform.rotation.w
        qz = t.transform.rotation.z

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw))

        self.get_logger().warn(f'Pose: {pose}')

        return pose

    async def get_pose_2d_async(self, stamp=None):
        """
        Get the 2d pose of the robot.
        Optionally, a specific timestamp can be provided to get the robot's pose at that time.
        """
        lookup_time = (
            rclpy.time.Time.from_msg(stamp)
            if stamp is not None
            else rclpy.time.Time()
        )

        # Lookup the transform
        try:
            t = await self.tf_buffer.lookup_transform_async(
                'map',
                'base_link',
                lookup_time
            )
        except TransformException as ex:
            self.get_logger().error(f'Could not transform: {ex}')
            return

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = t.transform.translation.x
        pose.y = t.transform.translation.y

        qw = t.transform.rotation.w
        qz = t.transform.rotation.z

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw))

        self.get_logger().warn(f'Pose: {pose}')

        return pose

    def map_callback(self, map_msg: OccupancyGrid):
        """New map received, so update x and y limits"""

        # Extract data from message
        map_origin = [map_msg.info.origin.position.x, 
                      map_msg.info.origin.position.y]
        map_resolution = map_msg.info.resolution
        map_height = map_msg.info.height
        map_width = map_msg.info.width

        # Set current limits
        self.xlim_ = [map_origin[0], map_origin[0]+map_width*map_resolution]
        self.ylim_ = [map_origin[1], map_origin[1]+map_height*map_resolution]

        # self.get_logger().warn('Map received:')
        # self.get_logger().warn(f'  xlim = [{self.xlim_[0]:.2f}, {self.xlim_[1]:.2f}]')
        # self.get_logger().warn(f'  ylim = [{self.ylim_[0]:.2f}, {self.ylim_[1]:.2f}]')
    
    def image_callback(self, rgb_msg, depth_msg):
        """
        Recieve an RGB image.
        Orchestrate image capture for the artefact detection pipeline.
        Publishes the captured images to the appropriate topics for further processing.
        """

        # Copy the image messages to a cv image
        rgb_image = self.cv_bridge_.imgmsg_to_cv2(rgb_msg, desired_encoding='passthrough')
        depth_image = self.cv_bridge_.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        
        # Process depth image to human readable format
        processed_depth_image = (np.clip((depth_image.astype(np.float64) / 15.0), 0.0, 1.0) * 255.0).astype(np.uint8)

        # Capture images if an image capture has been requested
        if self.capture_requested_:
            image_name = f"{uuid.uuid4()}.jpg"

            if self.activate_rgb_capture_:
                rgb_image_path = self.rgb_capture_directory_ / image_name
                cv2.imwrite(str(rgb_image_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                self.get_logger().info(f'Captured rgb image: {rgb_image_path}')

            if self.activate_depth_capture_:
                depth_image_path = self.depth_capture_directory_ / image_name
                cv2.imwrite(str(depth_image_path), depth_image)
                self.get_logger().info(f'Captured depth image: {depth_image_path}')
            
            if self.activate_processed_depth_capture_:
                processed_depth_image_path = self.processed_depth_capture_directory_ / image_name
                cv2.imwrite(str(processed_depth_image_path), processed_depth_image)
                self.get_logger().info(f'Captured processed depth image: {processed_depth_image_path}')

            self.capture_requested_ = False

        # Forward every N-th synchronized RGB/depth pair to yolo_ros.
        self.detection_image_count_ += 1
        if self.detection_image_count_ >= self.detection_every_n_images_:
            key = (rgb_msg.header.stamp.sec, rgb_msg.header.stamp.nanosec)
            self.detection_images_[key] = (rgb_msg, depth_msg)
            while len(self.detection_images_) > 10:
                del self.detection_images_[next(iter(self.detection_images_))]

            # Publish the appropriate image (depth or RGB) for artifact detection.
            if self.depth_detection_:
                processed_depth_image = cv2.cvtColor(processed_depth_image, cv2.COLOR_GRAY2BGR)
                depth_msg = self.cv_bridge_.cv2_to_imgmsg(processed_depth_image, encoding='bgr8')
                depth_msg.header = rgb_msg.header
                self.detection_image_pub_.publish(depth_msg)
            else:
                self.detection_image_pub_.publish(rgb_msg)

            self.detection_image_count_ = 0


    async def artifact_detections_callback(self, msg):
        """
        Receive artifact detections and process them for localization and visualization.
        """
        self.latest_artifact_detections_ = msg
        self.artifact_found_ = bool(msg.detections)

        # If no artifacts are found, skip further processing
        if not self.artifact_found_:
            return

        # Log artifact finding
        self.get_logger().info('Artifact found!')

        # Retrieve the corresponding image for the detections based on the timestamp key
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        images = self.detection_images_.pop(key, None)
        if images is None:
            return
        
        # Unpack the images
        rgb_msg, depth_msg = images
        rgb_image = self.cv_bridge_.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.cv_bridge_.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        
        # Annotate the image with the detections and publish it
        if self.depth_detection_:
            detection_image = (np.clip((depth_image.astype(np.float64) / 15.0), 0.0, 1.0) * 255.0).astype(np.uint8)
            detection_image = cv2.cvtColor(detection_image, cv2.COLOR_GRAY2BGR)
            annotated = self.annotate_image(detection_image, msg.detections)
        else:
            annotated = self.annotate_image(rgb_image, msg.detections)

        annotated_msg = self.cv_bridge_.cv2_to_imgmsg(annotated, encoding='bgr8')
        annotated_msg.header = msg.header
        self.image_detections_pub_.publish(annotated_msg)

        # Get robot pose at the time of the image capture
        robot_pose = await self.get_pose_2d_async(depth_msg.header.stamp)
        if robot_pose is None:
            self.get_logger().warn(f'localise_artifact: robot_pose is None.')
            return
        
        # Localise artifact
        for detection in msg.detections:
            self.localise_artifact(robot_pose, depth_image, detection)


    def annotate_image(self, image, detections):
        """
        Helper function to annotate an image with bounding boxes for detected artifacts.
        """
        # BGR colors for classes 0–5
        colors = [
            (144, 238, 144),  # Alien
            (0, 100, 0),      # Gem
            (0, 0, 255),      # Sign
            (230, 216, 173),  # Sphere
            (255, 0, 0),      # Mushroom
            (255, 200, 0),    # Minerals
        ]

        annotated = image.copy()

        for detection in detections:
            box = detection.bbox
            x, y = box.center.position.x, box.center.position.y
            w, h = box.size.x, box.size.y

            start = (round(x - w / 2), round(y - h / 2))
            end = (round(x + w / 2), round(y + h / 2))

            class_id = detection.class_id
            color = colors[class_id] if 0 <= class_id < len(colors) else (255, 255, 255)
            name = detection.class_name or f'Klasse {class_id}'

            cv2.rectangle(annotated, start, end, color, 2)
            cv2.putText(
                annotated, name,
                (max(0, start[0]), max(20, start[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )

        return annotated


    def capture_callback(self, msg):
        """
        Recieve an image capture request message.
        Image capture request can be made by:
            ros2 topic pub --once /capture_image std_msgs/msg/Empty '{}'
        """
        self.capture_requested_ = True
        self.get_logger().info('Requesting image capture.')


    def localise_artifact(self, robot_pose, depth_image, detection):
        """
        Compute the location of the artifact
        Save it to a list, publish rviz marker
        """

        # Camera parameters
        HORIZONTAL_AOV = 2/3 * math.pi
        IMAGE_WIDTH = 720
        FOCAL_LENGTH = IMAGE_WIDTH / (2.0 * math.tan(HORIZONTAL_AOV / 2.0))
        
        # Calculate artefact angle from detection coordinates and image AOV (120°)
        box = detection.bbox
        cx, cy = round(box.center.position.x), round(box.center.position.y)
        artifact_bearing = math.atan2((IMAGE_WIDTH/2 - cx), FOCAL_LENGTH)
        absolut_artifact_angle = wrap_angle(robot_pose.theta + artifact_bearing)

        # Calculate the artifact distance from the depth image
        artifact_depth = float(depth_image[cy, cx])
        if not np.isfinite(artifact_depth) or artifact_depth <= 0:
            return
        artifact_distance = artifact_depth / math.cos(artifact_bearing)

        # Calculate the artifact's position in the world frame.
        point = Point()
        point.x = robot_pose.x + artifact_distance * math.cos(absolut_artifact_angle)
        point.y = robot_pose.y + artifact_distance * math.sin(absolut_artifact_angle)
        point.z = 1.0

        # Save the artifact's position to the list
        self.artifact_locations_.append(point)

        # Publish the markers
        self.publish_artifact_markers()


    def publish_artifact_markers(self):
        """ Publish the artifact location markers"""

        # Merge the artifact points
        merged_points = self.merge_points(self.artifact_locations_)

        # Update the locations with the merged points
        self.marker_artifacts_.points = merged_points

        # Create and publish the MarkerArray
        marker_array = MarkerArray()
        marker_array.markers = [self.marker_artifacts_]
        self.marker_pub_.publish(marker_array)


    def merge_points(self, points):
        """
        Merge the given points to identify single artifacts.
        'Density-Based Spatial Clustering of Applications with Noise' is used to group nearby points into clusters.
        """
        points = np.array([[p.x, p.y, p.z] for p in points])

        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=3.0,       
            min_samples=1
        ).fit(points)

        labels = clustering.labels_
        merged_points = []

        # Iterate over each cluster label and merge the points
        for label in np.unique(labels):
            cluster = points[labels == label]
            merged_points.append(cluster.mean(axis=0))

        merged_points = [
            Point(x=float(x), y=float(y), z=float(z))
            for x, y, z in np.array(merged_points)
        ]

        return merged_points


    def planner_go_to_pose2d(self, pose2d):
        """Go to a provided 2d pose"""

        # Send a goal to navigate_to_pose with self.nav2_action_client_
        action_goal = NavigateToPose.Goal()
        action_goal.pose.header.stamp = self.get_clock().now().to_msg()
        action_goal.pose.header.frame_id = 'map'
        action_goal.pose.pose = pose2d_to_pose(pose2d)

        # Publish visualisation
        self.goal_pose_vis_.publish(action_goal.pose)

        # Decide whether to show feedback or not
        if self.get_parameter('print_feedback').value:
            feedback_method = self.feedback_callback
        else:
            feedback_method = None

        # Send goal to action server
        self.get_logger().warn(f'Sending goal [{pose2d.x:.2f}, {pose2d.y:.2f}]...')
        self.send_goal_future_ = self.nav2_action_client_.send_goal_async(
            action_goal,
            feedback_callback=feedback_method)
        self.send_goal_future_.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """The requested goal pose has been sent to the action server"""

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        # Goal accepted: get result when it's completed
        self.get_logger().warn(f'Goal accepted')
        self.get_result_future_ = goal_handle.get_result_async()
        self.get_result_future_.add_done_callback(self.goal_reached_callback)

    def feedback_callback(self, feedback_msg):
        """Monitor the feedback from the action server"""

        feedback = feedback_msg.feedback

        self.get_logger().info(f'{feedback.distance_remaining:.2f} m remaining')

    def goal_reached_callback(self, future):
        """The requested goal has been reached"""

        result = future.result().result
        self.get_logger().info(f'Goal reached!')
        self.ready_for_next_goal_ = True


    def planner_move_forwards(self, distance):
        """Simply move forward by the specified distance"""

        pose_2d = self.get_pose_2d()

        pose_2d.x += distance * math.cos(pose_2d.theta)
        pose_2d.y += distance * math.sin(pose_2d.theta)

        self.planner_go_to_pose2d(pose_2d)

    def planner_go_to_first_artifact(self):
        """Go to a pre-specified artifact location"""

        goal_pose2d = Pose2D(
            x = 18.1,
            y = 6.6,
            theta = math.pi/2
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_return_home(self):
        """Return to the origin"""

        goal_pose2d = Pose2D(
            x = 0.0,
            y = 0.0,
            theta = math.pi
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_walk(self):
        """Go to a random location, which may be invalid"""

        # Select a random location
        goal_pose2d = Pose2D(
            x = random.uniform(self.xlim_[0], self.xlim_[1]),
            y = random.uniform(self.ylim_[0], self.ylim_[1]),
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_goal(self):
        """Go to a random location out of a predefined set"""

        # Hand picked set of goal locations
        random_goals = [[15.2, 2.2],
                        [30.7, 2.2],
                        [43.0, 11.3],
                        [36.6, 21.9],
                        [33.0, 30.4],
                        [40.4, 44.3],
                        [51.5, 37.8],
                        [16.0, 24.1],
                        [3.4, 33.5],
                        [7.9, 13.8],
                        [14.2, 37.7]]

        # Select a random location
        goal_valid = False
        while not goal_valid:
            idx = random.randint(0,len(random_goals)-1)
            goal_x = random_goals[idx][0]
            goal_y = random_goals[idx][1]

            # Only accept this goal if it's within the current costmap bounds
            if goal_x > self.xlim_[0] and goal_x < self.xlim_[1] and \
               goal_y > self.ylim_[0] and goal_y < self.ylim_[1]:
                goal_valid = True
            else:
                self.get_logger().warn(f'Goal [{goal_x}, {goal_y}] out of bounds')

        goal_pose2d = Pose2D(
            x = goal_x,
            y = goal_y,
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def main_loop(self):
        """
        Set the next goal pose and send to the action server
        See https://docs.nav2.org/concepts/index.html
        """
        
        # Don't do anything until SLAM is launched
        if not self.tf_buffer.can_transform(
                'map',
                'base_link',
                rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform... Have you launched a SLAM node?')
            return

        #######################################################
        # Update flags related to the progress of the current planner

        # Check if previous goal still running
        if not self.ready_for_next_goal_:
            # self.get_logger().info(f'Previous goal still running')
            return

        self.ready_for_next_goal_ = False

        if self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.get_logger().info('Successfully reached first artifact!')
            self.reached_first_artifact_ = True
        if self.planner_type_ == PlannerType.RETURN_HOME:
            self.get_logger().info('Successfully returned home!')
            self.returned_home_ = True

        #######################################################
        # Select the next planner to execute
        # Update this logic as you see fit!
        if not self.reached_first_artifact_:
            self.planner_type_ = PlannerType.GO_TO_FIRST_ARTIFACT
        elif not self.returned_home_:
            self.planner_type_ = PlannerType.RETURN_HOME
        else:
            self.planner_type_ = PlannerType.RANDOM_GOAL

        #######################################################
        # Execute the planner by calling the relevant method
        # Add your own planners here!
        self.get_logger().info(f'Calling planner: {self.planner_type_.name}')
        if self.planner_type_ == PlannerType.MOVE_FORWARDS:
            self.planner_move_forwards(10)
        elif self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.planner_go_to_first_artifact()
        elif self.planner_type_ == PlannerType.RETURN_HOME:
            self.planner_return_home()
        elif self.planner_type_ == PlannerType.RANDOM_WALK:
            self.planner_random_walk()
        elif self.planner_type_ == PlannerType.RANDOM_GOAL:
            self.planner_random_goal()
        else:
            self.get_logger().error('No valid planner selected')
            self.destroy_node()


        #######################################################

def main():
    # Initialise
    rclpy.init()

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    while rclpy.ok():
        rclpy.spin(cave_explorer)