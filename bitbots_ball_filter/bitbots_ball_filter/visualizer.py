import matplotlib.pyplot as plt
from rclpy.node import Node
import rclpy
from soccer_vision_3d_msgs.msg import RobotArray
from geometry_msgs.msg import PoseWithCovarianceStamped


# x axis values
# x = [1,2,3]
# # corresponding y axis values
# y = [2,4,1]
#
# # plotting the points 
# plt.plot(x, y)
#
# # naming the x axis
# plt.xlabel('x - axis')
# # naming the y axis
# plt.ylabel('y - axis')
#
# # giving a title to my graph
# plt.title('My first graph!')
#
# # function to show the plot
# plt.show()

class Visualizer(Node):
    def __init__(self) -> None:
        super().__init__('visualizer')
        self.array_position_truth_x = []
        self.array_position_truth_y = []
        self.array_position_filtered_x = []
        self.array_position_filtered_y = []
        self.max_length = 1000
        self.unfinished_truth = True
        self.unfinished_filtered = True

        self.subscriber_robots_relative_groundtruth = self.create_subscription(
            RobotArray,
            "robots_relative",  # todo is relative really the ground truth?
            self.robots_relative_groundtruth_callback,
            1
        )

        self.subscriber_robots_relative_filtered = self.create_subscription(
            PoseWithCovarianceStamped,
            "robot_position_relative_filtered",
            self.robots_relative_filtered_callback,
            1
        )

    def robots_relative_groundtruth_callback(self, robots_relative_groundtruth):
        if self.unfinished_truth:
            robot = sorted(robots_relative_groundtruth.robots, key=lambda robot: robot.confidence.confidence)[-1]
            position = robot.bb.center.position
            self.array_position_truth_x.append(position.x)
            self.array_position_truth_y.append(position.y)
            if len(self.array_position_truth_x) >= self.max_length:
                print('finished ground truth')
                plt.plot(self.array_position_truth_x, self.array_position_truth_y, label='ground truth')
                self.unfinished_truth = False

    def robots_relative_filtered_callback(self, robots_relative_filtered):
        if self.unfinished_filtered:
            self.robot_position_filtered = robots_relative_filtered.pose.pose.position
            self.array_position_filtered_x.append(self.robot_position_filtered.x)
            self.array_position_filtered_y.append(self.robot_position_filtered.y)
            if len(self.array_position_filtered_x) >= self.max_length:
                print('finished filtered')
                plt.plot(self.array_position_filtered_x, self.array_position_filtered_y, label='filtered')
                self.unfinished_filtered = False


    def plot(self):
        print('plotting...')
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title('My first graph!')
        plt.legend()
        plt.show()

    def is_truth_unfinished(self):
        return self.unfinished_truth

    def is_filtered_unfinished(self):
        print(self.unfinished_filtered)
        return self.unfinished_filtered



def main(args=None):
    rclpy.init()
    visualizer = Visualizer()
    try:
        while visualizer.is_truth_unfinished() or visualizer.is_filtered_unfinished():
            rclpy.spin_once(visualizer)
    except KeyboardInterrupt:
        visualizer.destroy_node()
        rclpy.shutdown()

    visualizer.plot()
    visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()