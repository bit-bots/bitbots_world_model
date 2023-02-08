

#! /usr/bin/env python3
import rclpy
import numpy as np
import argparse
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped
from soccer_vision_3d_msgs.msg import Ball, BallArray, Obstacle, ObstacleArray, Robot, RobotArray


class SimObject(Node):
    def __init__(self, object_type: str = 'ball'):
        super().__init__('object_sim_node')
        self.logger = self.get_logger()
        self.object_type = object_type
        self.logger.info('Created object_sim_node with object type: ' + self.object_type)
        #self.num_of_objects = 1

        # Simulation timings
        self.pub_frequency = 20
        self.dt = 1.0 / self.pub_frequency

        # Max object movements
        self.max_velocity = np.array([2, 2])  # m/s
        self.max_acceleration = np.array([0.8, 0.8])  # m/s/s
        self.max_error = np.array([1, 1])

        # Initialize velocity and position
        # self.velocity_array = []
        # self.position_array = []
        # for i in range(self.num_of_objects):
        #     self.velocity_array.append(np.zeros((2)))
        #     self.position_array.append(np.zeros((2))) #todo use np.random.rand() for random starting position
        self.velocity = np.zeros((2))
        self.position = np.zeros((2))

        # Create publishers and timer
        self.pub_pos_viz = self.create_publisher(PoseWithCovarianceStamped, 'position', 1)
        self.pub_pos_err_viz = self.create_publisher(PoseWithCovarianceStamped, 'position_err', 1)

        if self.object_type == 'ball':
            self.pub_pos_err = self.create_publisher(BallArray, 'balls_relative', 1)
        elif self.object_type == 'robot':
            self.pub_pos_err = self.create_publisher(RobotArray, 'robots_relative', 1)
        else:
            self.pub_pos_err = self.create_publisher(ObstacleArray, 'obstacles_relative', 1)
        self.timer = self.create_timer(self.dt, self.step)

    def step(self):
        # Step the object
        # Adjust velocity and position of each object:
        # for i in range(self.num_of_objects):
        #     self.velocity_array[i] = np.clip(self.velocity + self.gen_acceleration() * self.dt, -self.max_velocity,
        #                         self.max_velocity)
        #     self.position_array[i] += self.velocity_array[i] * self.dt
        #
        #     p_err = self.position_array[i] + self.gen_error()

        self.velocity = np.clip(self.velocity + self.gen_acceleration()  * self.dt, -self.max_velocity, self.max_velocity)
        self.position += self.velocity * self.dt
        p_err = self.position + self.gen_error()

            # Publish results
        self.pub_pos_viz.publish(self.gen_pose_cov_stamped_msg(self.position[0], self.position[1])) #todo pub pose arrays
        self.pub_pos_err_viz.publish(self.gen_pose_cov_stamped_msg(p_err[0], p_err[1])) #todo pub pose arrays
            # check with object type should be published
        if self.object_type == 'ball':
            self.pub_pos_err.publish(self.gen_ball_array_msg(p_err[0], p_err[1]))
        elif self.object_type == 'robot':
            self.pub_pos_err.publish(self.gen_robot_array_msg(p_err[0], p_err[1]))
        else:
            pass #todo

    def gen_error(self):
        return np.multiply(np.random.rand(2) * 2 - 1, self.max_error)

    def gen_acceleration(self):
        return np.clip(np.random.randn(2), -self.max_acceleration, self.max_acceleration)

    def gen_pose_cov_stamped_msg(self, x, y):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rclpy.time.Time.to_msg(self.get_clock().now())
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.pose.position.x = x
        pose_msg.pose.pose.position.y = y
        pose_msg.pose.pose.orientation.w = 1.0
        return pose_msg

    def gen_ball_array_msg(self, x, y):
        object_msg = Ball()
        object_msg.center.x = x
        object_msg.center.y = y
        object_msg.center.z = 0.0
        object_msg.confidence.confidence = 0.9

        object_array_msg = BallArray()
        object_array_msg.header.stamp = rclpy.time.Time.to_msg(self.get_clock().now())
        object_array_msg.header.frame_id = 'odom'
        object_array_msg.balls.append(object_msg)
        return object_array_msg

    def gen_robot_array_msg(self, x, y):#todo better values
        object_msg = Robot()
        object_msg.bb.center.position.x = x
        object_msg.bb.center.position.y = y
        object_msg.bb.center.position.z = 0.0
        object_msg.bb.center.orientation.x = 0.0
        object_msg.bb.center.orientation.y = 0.0
        object_msg.bb.center.orientation.z = 0.0
        object_msg.bb.center.orientation.w = 0.0
        object_msg.bb.size.x = 1.0
        object_msg.bb.size.y = 1.0
        object_msg.bb.size.z = 1.0
        object_msg.attributes.player_number = 0
        object_msg.attributes.team = 0
        object_msg.attributes.state = 0
        object_msg.attributes.facing = 0
        object_msg.confidence.confidence = 0.9

        object_array_msg = RobotArray()
        object_array_msg.header.stamp = rclpy.time.Time.to_msg(self.get_clock().now())
        object_array_msg.header.frame_id = 'odom'
        object_array_msg.robots.append(object_msg)
        return object_array_msg


def main(args):
    rclpy.init()
    node = SimObject(object_type=args.object_type)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Publishes messages of objects with simulated movement")
    # adds the object type as an argument with ball as the default
    parser.add_argument(
        nargs='?', const='ball', type=str, default='ball',
        help="Type of the object to be simulated (ball, obstacle, robot)", dest="object_type")
    args = parser.parse_args()
    main(args)
