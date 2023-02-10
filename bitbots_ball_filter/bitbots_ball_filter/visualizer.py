import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from rclpy.node import Node
import rclpy
import math
import random
import json
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
        self.groundtruth_position_x_array = []
        self.groundtruth_position_y_array = []
        self.detected_position_x_array = []
        self.detected_position_y_array = []
        self.filtered_position_x_array = []
        self.filtered_position_y_array = []
        self.array_error = []
        self.max_length = 1000
        self.use_noise_data = True
        self.noise_array = []
        self.noise_array_counter = 0
        self.trial_number = -1
        self.unfinished_truth = True
        self.unfinished_filtered = True
        self.unfinished_detected = True
        self.output_data = None

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
            self.robots_detection_callback,
            1
        )

        self.subscriber_robots_relative_filtered = self.create_subscription(
            PoseWithCovarianceStamped,
            "robot_position_relative_filtered",
            self.robots_relative_filtered_callback,
            1
        )

        with open('output_data.json', 'r') as f1:
            self.output_data = json.load(f1)
        f1.close()
        if self.trial_number == -1 or self.trial_number > len(self.output_data['study_outputs']) - 1:
            trial = self.output_data['study_outputs'][
                len(self.output_data['study_outputs']) - 1]
        else:
            trial = self.output_data['study_outputs'][self.trial_number]
        self.noise_array = trial['noise_array']
        if len(self.noise_array) == 0:
            self.use_noise_data = False
            print('ERROR: no noise data available, output will use normal detected messages')
        print("...initialization done")

        # plot average error:
        average_error_array = trial['average_error_array']
        average_error_array_x = []
        average_error_array_y = []
        for i in range(0, len(average_error_array) - 1):
            if average_error_array[i] < 0.5:
                average_error_array_x.append(i)
                average_error_array_y.append(average_error_array[i])
        # for i in range(0, len(average_error_array) - 3):
        #     if average_error_array[i] != 999 and average_error_array[i+1] != 999:
        #         print("hey")
        #         self.ax.plot(
        #             average_error_array_x[i:i + 2],
        #             average_error_array_y[i:i + 2],
        #             c="blue",
        #             lw=1.5
        #         )



        plt.plot(average_error_array_x, average_error_array_y,
                 label='average error',
                 lw=1.5,
                 c="blue")
        #plt.xticks(range(0, len(average_error_array_y) + 1, 1))

        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title('My first graph!')
        plt.legend()
        plt.show()

        # todo remove for normal behaviour
        self.unfinished_filtered = False
        self.unfinished_truth = False
        self.unfinished_detected = False




    def robots_groundtruth_callback(self, robots_groundtruth):
        if self.unfinished_truth:
            position = robots_groundtruth.pose.pose.position
            self.groundtruth_position_x_array.append(position.x)
            self.groundtruth_position_y_array.append(position.y)
            if self.use_noise_data:
                self.detected_position_x_array.append(position.x + self.noise_array[self.noise_array_counter][0])
                self.detected_position_y_array.append(position.y + self.noise_array[self.noise_array_counter][1])
                self.noise_array_counter += 1

            if len(self.groundtruth_position_x_array) >= self.max_length:
                print('finished ground truth')

                self.unfinished_truth = False

                if self.use_noise_data:
                    print('finished detected')

                    self.unfinished_detected = False

    def robots_detection_callback(self, robots_relative_detected):
        if self.unfinished_detected and not self.use_noise_data:
            robot = sorted(robots_relative_detected.robots, key=lambda robot: robot.confidence.confidence)[-1]
            position = robot.bb.center.position
            self.detected_position_x_array.append(position.x)
            self.detected_position_y_array.append(position.y)
            if len(self.detected_position_x_array) >= self.max_length:
                print('finished detected')

                self.unfinished_detected = False

    def robots_relative_filtered_callback(self, robots_relative_filtered):
        if self.unfinished_filtered:
            self.robot_position_filtered = robots_relative_filtered.pose.pose.position
            self.filtered_position_x_array.append(self.robot_position_filtered.x)
            self.filtered_position_y_array.append(self.robot_position_filtered.y)

            # calculate error of the filtered msg
            point_1 = (self.groundtruth_position_x_array[len(self.groundtruth_position_x_array) - 1],
                       self.groundtruth_position_y_array[len(self.groundtruth_position_y_array) - 1])
            point_2 = (self.robot_position_filtered.x,
                       self.robot_position_filtered.y)
            distance = math.dist(point_1, point_2)
            self.array_error.append(distance)
            # distances are added up to create average value later
            # self.error_sum += distance #todo do and display this

            if len(self.filtered_position_x_array) >= self.max_length:
                print('finished filtered')
                #plt.plot(self.array_position_filtered_x, self.array_position_filtered_y, label='filtered')
                self.unfinished_filtered = False






    def plot(self):
        print('plotting...')

        plt.plot(self.detected_position_x_array, self.detected_position_y_array,
                 label='detected',
                 lw=1.0,
                 c="silver")



        viridis = mpl.colormaps['RdYlGn']  #.resampled(8)
        print(self.filtered_position_x_array)
        for i in range(0, len(self.filtered_position_x_array) - 4):
            error_sum = 0
            for error in self.array_error[i:i+2]:
                error_sum += error
            #todo make this better by showing the legend of the scale
            # Todo create mean from surrounders and deal with edges
            color = viridis(error_sum / 2)
            self.ax.plot(
                self.filtered_position_x_array[i:i + 2],
                self.filtered_position_y_array[i:i + 2],
                c=color,
                lw=2
            )

        plt.plot(self.groundtruth_position_x_array, self.groundtruth_position_y_array,
                 label='ground truth',
                 lw=1.5,
                 c="black")

        print('finished filtered')

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