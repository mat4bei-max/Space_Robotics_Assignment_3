"""Publish a fixed RGB image and report the normal yolo_ros output."""
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from yolo_msgs.msg import DetectionArray


class YoloTest(Node):
    def __init__(self):
        super().__init__('yolo_test_node')

        # Configuration
        image_path = Path(
            '/home/mat4b/projects/space_robotics/assignment_3_ws/src/cave_explorer/'
            'artifact_detection/data/raw_rgb_images/'
            '2af0ad97-478e-44d9-8f22-e0f8fe11cb92.jpg'
        )
        publish_period = float(self.declare_parameter('publish_period', 2.0).value)
        if publish_period <= 0:
            raise ValueError('publish_period must be positive')
        if not image_path.is_file():
            raise FileNotFoundError(f'Image does not exist: {image_path}')

        # Prepare RGB image, matching the camera stream
        self.bridge_ = CvBridge()
        bgr_image = cv2.imread(str(image_path))
        if bgr_image is None:
            raise ValueError(f'Could not decode image: {image_path}')
        self.rgb_image_ = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Publisher and subscriber, matching CaveExplorer topics and QoS
        self.image_pub_ = self.create_publisher(Image, '/artifact_detection/image', 1)
        self.detections_sub_ = self.create_subscription(
            DetectionArray, '/yolo/detections', self.detections_callback, 1,
        )

        # Periodic image publishing (wall time by default; no Gazebo needed)
        self.timer_ = self.create_timer(publish_period, self.publish_image)
        self.get_logger().info(f'Test image: {image_path.resolve()}')

    def publish_image(self):
        if self.image_pub_.get_subscription_count() == 0:
            self.get_logger().info('Waiting for image subscriber...', throttle_duration_sec=5.0)
            return
        msg = self.bridge_.cv2_to_imgmsg(self.rgb_image_, encoding='rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_optical_frame'
        self.image_pub_.publish(msg)
        self.get_logger().info('Published test image on /artifact_detection/image')

    def detections_callback(self, msg):
        if not msg.detections:
            self.get_logger().info('YOLO result received: no artifacts detected.')
            return
        self.get_logger().info(f'Artifact found! {len(msg.detections)} detection(s)')
        for detection in msg.detections:
            box = detection.bbox
            self.get_logger().info(
                f'class={detection.class_name} (id={detection.class_id}), '
                f'confidence={detection.score:.4f}, '
                f'bbox center=({box.center.position.x:.1f}, {box.center.position.y:.1f}), '
                f'size=({box.size.x:.1f}, {box.size.y:.1f}) px'
            )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = YoloTest()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
