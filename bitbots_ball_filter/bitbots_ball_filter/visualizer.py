import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from rclpy.node import Node
import rclpy
import math
from statistics import mean
import random
import os

import subprocess
import argparse
import json
import signal
import time
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
        self.file_numbers = [1]
        self.array_error = []
        self.max_length = 1000
        self.use_noise_data = False
        self.noise_array = []
        self.noise_array_counter = 0
        self.study_number = study_number
        # mwn comparison
        #self.trial_array = [36, 35, 37] # 1000 2
        #self.trial_array = [39, 38, 40] # 100 2
        #self.trial_array = [42, 41, 43] # 1000 0.5
        #self.trial_array = [45, 44, 46] # 100 0.5
        self.study_array = [study_number]
        self.unfinished_truth = True
        self.unfinished_filtered = True
        self.unfinished_detected = True
        self.output_data = None
        self.pls_plot = False
        self.current_file = 0
        self.current_study = 0

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

        with open('output_data1.json', 'r') as f1:
            self.output_data = json.load(f1)
        f1.close()
        param_str = ''

        study = self.output_data['study_outputs'][self.study_number]
        for parameter in study['parameters']:
            param_str += parameter["name"] + "#" + str(parameter['result']) + "#"
        param_str += "trial_number" + "#" + str(self.study_number) + "#"

        # save params to file so that the filter can access them
        with open('text_params.txt', 'w') as file:
            file.write(param_str)
        file.close()
        print("finished writing params")



        # todo remove for normal behaviour
        # self.unfinished_filtered = False
        # self.unfinished_truth = False
        # self.unfinished_detected = False
        # self.dont_plot = True


    def plot_average_error(self):
        temp = 0
        for study_num in self.study_array:

            with open('output_data2.json', 'r') as f1:
                self.output_data = json.load(f1)
            f1.close()
            if self.study_number == -1 or self.study_number > len(self.output_data['study_outputs']) - 1:
                trial = self.output_data['study_outputs'][study_num]
            else:
                trial = self.output_data['study_outputs'][self.study_number]
            self.noise_array = trial['noise_array']
            if len(self.noise_array) == 0:
                self.use_noise_data = False
                print('ERROR: no noise data available, output will use normal detected messages')
            print("...initialization done")

            # plot average error:
            print(trial['study_number'])
            average_error_array = trial['average_error_array']
            average_error_array_x = []
            average_error_array_y = []
            for i in range(0, len(average_error_array) - 1):
                if average_error_array[i] > 0:
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

            print("best value: {}".format(min(average_error_array_y)))
            print("time: {}".format(trial['time']))

            if temp == 0:
                color = 'red'
                label = 'wide parameter space'
            elif temp == 1:
                color = 'blue'
                label = 'medium parameter space'
            else:
                color = 'green'
                label = 'narrow parameter space'
            plt.plot(average_error_array_x, average_error_array_y,
                     label=label,
                     lw=1.0,
                     c=color)
            # plt.xticks(range(0, len(average_error_array_y) + 1, 1))
            temp += 1
            print("min for {}: {}".format(study_num, min(average_error_array_y)))
            print("convergence for {}: {}".format(study_num, self.estimate_q(average_error_array_y)))

        # plt.yscale('log')
        # #plt.xscale('log')
        # plt.xlabel('Trial')
        # plt.ylabel('Average Error')
        # plt.title('')
        # plt.legend()
        # plt.savefig('100gbu05.pdf')
        # plt.show()

    def estimate_q(self, eps):
        """
        estimate rate of convergence q from sequence esp
        """
        x = np.arange(len(eps) - 1)
        y = np.log(np.abs(np.diff(np.log(eps))))
        line = np.polyfit(x, y, 1)  # fit degree 1 polynomial
        q = np.exp(line[0])  # find q
        return q

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


        norm = mpl.colors.Normalize(vmin=0, vmax=0.5)
        color_map = mpl.colormaps['RdYlGn_r']  #.resampled(8)
        total_error_sum = 0
        for i in range(0, len(self.filtered_position_x_array) - 4):
            error_sum = 0
            total_error_sum += self.array_error[i]
            for error in self.array_error[i:i+2]:
                error_sum += error ** 2
            color = color_map(norm(error_sum / 2))
            # self.ax.plot(
            #     self.filtered_position_x_array[i:i + 2],
            #     self.filtered_position_y_array[i:i + 2],
            #     c=color,
            #     lw=3
            # )
        print(len(self.filtered_position_x_array))
        output_text = "average error: {}".format(math.sqrt(total_error_sum / (len(self.filtered_position_x_array))))
        print(output_text)

        plt.plot(self.groundtruth_position_x_array, self.groundtruth_position_y_array,
                 label='ground truth',
                 lw=1.5,
                 c="black")

        print('finished filtered')

        # plot error
        # plt.plot(self.array_position_filtered_x, self.array_position_filtered_y, label='filtered')
        if self.pls_plot:
            plt.xlabel('x - axis')
            plt.ylabel('y - axis')
            plt.title('Robot Position')
            plt.legend()
            plt.savefig('example.pdf')
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
    parser = argparse.ArgumentParser('Optimizes parameters of a tracking filter')
    parser.add_argument(
        '--study',
        type=int,
        default='1',
        help='number of study'
    )
    args = parser.parse_args()
    study_number = args.study
    main()