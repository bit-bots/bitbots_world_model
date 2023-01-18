import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from rclpy.node import Node
import rclpy
import math
import random
import numpy as np
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
        self.array_position_detected_x = []
        self.array_position_detected_y = []
        self.array_position_filtered_x = []
        self.array_position_filtered_y = []
        self.array_error = []
        self.max_length = 400
        self.unfinished_truth = True
        self.unfinished_filtered = True
        self.unfinished_detected = True

        # setup mathplotlib stuff
        # plt.rcParams["figure.figsize"] = [7.50, 3.50]
        # plt.rcParams["figure.autolayout"] = True
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()


        self.subscriber_groundtruth = self.create_subscription(
            PoseWithCovarianceStamped,
            "position",
            self.robots_groundtruth_callback,
            1
        )

        self.subscriber_robots_relative = self.create_subscription(
            RobotArray,
            "robots_relative",
            self.robots_relative_callback,
            1
        )

        self.subscriber_robots_relative_filtered = self.create_subscription(
            PoseWithCovarianceStamped,
            "robot_position_relative_filtered",
            self.robots_relative_filtered_callback,
            1
        )

    def robots_groundtruth_callback(self, robots_groundtruth):
        if self.unfinished_truth:
            position = robots_groundtruth.pose.pose.position
            self.array_position_truth_x.append(position.x)
            self.array_position_truth_y.append(position.y)
            if len(self.array_position_truth_x) >= self.max_length:
                print('finished ground truth')
                plt.plot(self.array_position_truth_x, self.array_position_truth_y,
                         label='ground truth',
                         lw=1.5,
                         c="black")
                self.unfinished_truth = False

    def robots_relative_callback(self, robots_relative_detected):
        if self.unfinished_detected:
            robot = sorted(robots_relative_detected.robots, key=lambda robot: robot.confidence.confidence)[-1]
            position = robot.bb.center.position
            self.array_position_detected_x.append(position.x)
            self.array_position_detected_y.append(position.y)
            if len(self.array_position_detected_x) >= self.max_length:
                print('finished detected')
                plt.plot(self.array_position_detected_x, self.array_position_detected_y,
                         label='detected',
                         lw=1.5,
                         c="blue")
                self.unfinished_detected = False

    def robots_relative_filtered_callback(self, robots_relative_filtered):
        if self.unfinished_filtered:
            self.robot_position_filtered = robots_relative_filtered.pose.pose.position
            self.array_position_filtered_x.append(self.robot_position_filtered.x)
            self.array_position_filtered_y.append(self.robot_position_filtered.y)

            # calculate error of the filtered msg
            point_1 = (self.array_position_truth_x[len(self.array_position_truth_x) - 1],
                       self.array_position_truth_y[len(self.array_position_truth_y) - 1])
            point_2 = (self.robot_position_filtered.x,
                       self.robot_position_filtered.y)
            distance = math.dist(point_1, point_2)
            self.array_error.append(distance)
            # distances are added up to create average value later
            # self.error_sum += distance #todo do and display this

            if len(self.array_position_filtered_x) >= self.max_length:
                print('finished filtered')
                #plt.plot(self.array_position_filtered_x, self.array_position_filtered_y, label='filtered')
                self.unfinished_filtered = False






    def plot(self):
        print('plotting...')
        viridis = mpl.colormaps['RdYlGn']  #.resampled(8)
        for i in range(0, len(self.array_position_filtered_x) - 12):
            error_sum = 0
            for error in self.array_error[i:i+10]:
                error_sum += error
            #todo make this better by showing the legend of the scale
            # Todo create mean from surrounders and deal with edges
            color = viridis(error_sum / 10)
            self.ax.plot(
                self.array_position_filtered_x[i:i+2],
                self.array_position_filtered_y[i:i+2],
                c=color,
                lw=2
            )

        # plot error
        # plt.plot(self.array_position_filtered_x, self.array_position_filtered_y, label='filtered')

        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title('My first graph!')
        plt.legend()
        plt.show()

    def is_truth_unfinished(self):
        return self.unfinished_truth

    def is_detected_unfinished(self):
        return self.unfinished_detected

    def is_filtered_unfinished(self):
        print(self.unfinished_filtered)
        return self.unfinished_filtered



def main(args=None):
    rclpy.init()
    visualizer = Visualizer()
    try:
        while visualizer.is_truth_unfinished() \
                or visualizer.is_filtered_unfinished() \
                or visualizer.is_detected_unfinished():
            rclpy.spin_once(visualizer)
    except KeyboardInterrupt:
        visualizer.destroy_node()
        rclpy.shutdown()

    visualizer.plot()
    visualizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()