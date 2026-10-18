import rclpy
from rclpy.node import Node
from std_msgs.msg import String

COMMANDS = {
    "1": ("Toggle Depth Detection", "depth_detection"),
    "2": ("Toggle RGB Detection", "rgb_detection"),
    "3": ("Toggle Degraded Detection", "degraded_detection"),
    "4": ("Change Detection Frame Rate (1/2/4)", "change_detection_framrate"),
}

class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")

        self.publisher = self.create_publisher(
            String, "/control/commands", 10
        )

    def run(self):

        while rclpy.ok():
            print("\n=== Robot Control ===")

            for key, (description, _) in COMMANDS.items():
                print(f"{key} - {description}")

            print("q - Quit")
            print("=====================")

            key = input("Select option: ").strip()

            if key == "q":
                break

            if key not in COMMANDS:
                print("Invalid option")
                continue

            description, command = COMMANDS[key]

            self.publisher.publish(String(data=command))
            print(f"Executed: {description}")


def main():
    rclpy.init()
    node = ControllerNode()

    try:
        node.run()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()