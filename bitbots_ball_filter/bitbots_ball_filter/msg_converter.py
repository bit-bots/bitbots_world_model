import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from soccer_vision_3d_msgs.msg import RobotArray
from geometry_msgs.msg import PoseStamped, PoseArray, Pose


class MsgConverter(Node):
    """
    converts RobotArray msgs to PoseArray msgs
    """
    def __init__(self):
        super().__init__('msg_converter')

        # setup robot subscriber:
        self.subscriber = self.create_subscription(
            RobotArray,
            'robots_relative',
            self.robot_callback,
            1
        )

        # setup position publisher:
        self.pub_pos_err_array_viz = self.create_publisher(PoseArray, 'position_err_array', 1)

    def robot_callback(self, msg: RobotArray) -> None:
        """
        Converts RobotArray msg to PoseArray

        :param msg: List of robot-detections
        """
        stamp = msg.header.stamp
        pose_array = []
        if msg.robots:
            for robot in msg.robots:
                pose_array.append(
                    self.gen_pose_cov_stamped_msg(
                        robot.bb.center.position.x,
                        robot.bb.center.position.y
                    ))
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = stamp
        pose_array_msg.header.frame_id = 'odom'
        pose_array_msg.poses = pose_array

        self.pub_pos_err_array_viz.publish(pose_array_msg)

    def gen_pose_cov_stamped_msg(self, x, y):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.orientation.w = 1.0
        return pose



def main(args=None):
    rclpy.init(args=args)
    node = MsgConverter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()