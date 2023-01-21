import optuna
import os
import rclpy
import math
import json
import argparse
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from soccer_vision_3d_msgs.msg import RobotArray, Robot
import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt


class TrialOptimizer(Node):
    def __init__(self) -> None:
        super().__init__('robot_optimizer')
        self.logger = self.get_logger()
        self.current_filter_cycle = 1  # todo change this to 0 again
        self.error_sum = 0
        self.stop_trial = False
        self.current_robot_relative_err_msg = None
        self.robot_msg_err_queue = []  # initializing queue
        self.robot_position_groundtruth = None
        self.robot_position_filtered = None
        self.robot_position_true_queue = []

        # setup subscriber for ground truth robot positions: todo not necessary todo really?
        self.subscriber_robots_relative_groundtruth = self.create_subscription(
            RobotArray,
            "robots_relative",  # todo is relative really the ground truth?
            self.robots_relative_groundtruth_callback,
            1
        )

        # setup subscriber for filtered robot positions:
        # todo this might become an array or a different type once I publish multiple filters
        self.subscriber_robots_relative_filtered = self.create_subscription(
            PoseWithCovarianceStamped,
            "robot_position_relative_filtered",
            self.robots_relative_filtered_callback,
            1
        )

        # setup publisher for robot relative with noise from rosbag:
        self.robot_position_err_publisher = self.create_publisher( #todo
            RobotArray,
            'robots_relative',
            1
        )

        # reset filter:
        # todo get reset service name from config
        # robot_filter_reset_service_name = "ball_filter_reset"
        # os.system("ros2 service call /{} std_srvs/Trigger".format(robot_filter_reset_service_name))

    def startup(self, robot_position_true_queue, robot_msg_err_queue):
        self.robot_position_true_queue = robot_position_true_queue
        self.robot_msg_err_queue = robot_msg_err_queue
        self.publish_robot_position_err()

    def publish_robot_position_err(self):
        """
        pops first message from the message queue created out of the rosbag and publishes it
        """
        if len(self.robot_msg_err_queue) > 0:
            self.current_robot_relative_err_msg = self.robot_msg_err_queue.pop()
            self.robot_position_err_publisher.publish(self.current_robot_relative_err_msg)  # the msg is a tuple of time stamp + msg
        else:
            self.logger.warn("Ran out of messages to publish. Stopping trial")
            self.stop_trial = True

    def robots_relative_groundtruth_callback(self, robots_relative_groundtruth) -> None: # todo no necessary anymore
        """
        receives and saves the last value for the ground truth of the robot positions

        param robots_relative_groundtruth: ground truth of robot data
        """
        # todo deal with multiple robots instead of just choosing one
        robot = sorted(robots_relative_groundtruth.robots, key=lambda robot: robot.confidence.confidence)[-1]
        self.robot_position_groundtruth = robot.bb.center.position

    def robots_relative_filtered_callback(self, robots_relative_filtered) -> None:
        """
        receives and saves the last value for the filtered robot data

        param robots_relative_filtered: filtered robot data
        """
        # todo deal with multiple robots
        self.robot_position_filtered = robots_relative_filtered.pose.pose.position

        # calculates the error based on the distance between last ground truth of the robot positions and the current filtered robot positions
        # todo is this correct?

        temp = self.robot_position_true_queue[0]
        self.robot_position_groundtruth = temp[1].pose.pose.position
        point_1 = (self.robot_position_groundtruth.x,
                   self.robot_position_groundtruth.y)
        point_2 = (self.robot_position_filtered.x,
                   self.robot_position_filtered.y)
        distance = math.dist(point_1, point_2)
        # distances are added up to create average value later
        self.error_sum += distance
        self.current_filter_cycle += 1

        self.publish_robot_position_err()


    def get_filter_cycles(self) -> int:
        """
        returns number of filter cycles that have been passed through
        """
        return self.current_filter_cycle

    def get_average_error(self) -> float:
        """
        returns average error based on the distance between ground truth and filtered robot positions for this trial
        """
        return self.error_sum / self.current_filter_cycle

    def get_stop_trial(self):
        """
        returns boolean whether trial should be stopped prematurely
        """
        return self.stop_trial

class BagFileParser():
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()

        ## create a message type map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_type = {name_of:type_of for id_of,name_of,type_of in topics_data}
        self.topic_id = {name_of:id_of for id_of,name_of,type_of in topics_data}
        self.topic_msg_message = {name_of:get_message(type_of) for id_of,name_of,type_of in topics_data}

    def __del__(self):
        self.conn.close()

    # Return [(timestamp0, message0), (timestamp1, message1), ...]
    def get_messages(self, topic_name):

        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        # Deserialise all and timestamp them
        return [ (timestamp,deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp,data in rows]

def objective(self, trial) -> float:
    """
    Optuna's objective function that runs through a trial, tries out parameter values
    and calculates the evaluation value

    param trial: Optuna trial object
    """

    # suggest parameter values
    data_file = open('data.json')
    data = json.load(data_file)
    if args.debug == 'True':
        debug_var = ''
    else:
        debug_var = '> /dev/null 2>&1'
    for parameter in data["parameters"]:
        temp = None
        if parameter["type"] == "int":
            temp = trial.suggest_int(parameter["name"], parameter["min"], parameter["max"])
        elif parameter["type"] == "float":
            temp = trial.suggest_float(parameter["name"], parameter["min"], parameter["max"])
        elif parameter["type"] == "categorical":
            temp = trial.suggest_categorical(parameter["name"], parameter["choices"])
        if args.debug == 'True':
            print("Suggestion for " + parameter["name"] + ": " + str(temp))
        os.system("ros2 param set /bitbots_ball_filter {} {} {}".format(parameter["name"], str(temp), debug_var))
    data_file.close()
    # start filter optimizer
    # todo
    rclpy.init()
    filter_optimizer = TrialOptimizer()
    filter_optimizer.startup(robot_position_true_queue, robot_msg_err_queue)
    try:
        while filter_optimizer.get_filter_cycles() < args.cycles and not filter_optimizer.get_stop_trial():
            rclpy.spin_once(filter_optimizer)
    except KeyboardInterrupt:
        filter_optimizer.destroy_node()
        rclpy.shutdown()

    average_error = filter_optimizer.get_average_error()
    filter_optimizer.destroy_node()
    rclpy.shutdown()

    # return evaluation value
    return average_error

def generate_msgs(use_noise, bag_file):
    # unpack rosbag:
    print("Unpacking bag")
    bag_parser = BagFileParser(bag_file)
    # extract groundtruth:
    robot_position_true_queue = bag_parser.get_messages("/position")     # todo am I doing this the wrong way around? is this actually getting the last msg first?
    # extract noisy position:
    noise_array = []
    if not use_noise:
        # use available noisy position:
        robot_msg_err_queue = bag_parser.get_messages("/robots_relative")
    else:
        # generate own noisy position:
        robot_msg_err_queue = []
        for i in range(0, len(robot_position_true_queue)):
            max_error = np.array([1, 1])
            error = np.multiply(np.random.rand(2) * 2 - 1, max_error)
            noise_array.append(error)  # save error for later visualization
            groundtruth_msg = robot_position_true_queue[i][1]  # takes the msg from the timestamp + msg tuple
            p_err = [groundtruth_msg.pose.pose.position.x,
                     groundtruth_msg.pose.pose.position.y] + error
            robot_msg_err_queue.append(gen_robot_array_msg(p_err[0], p_err[1], groundtruth_msg.header.stamp))
    print("Message extraction finished")
    return robot_position_true_queue, robot_msg_err_queue, noise_array

def gen_robot_array_msg(x, y, timestamp):#todo better values
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
    object_array_msg.header.stamp = timestamp
    object_array_msg.header.frame_id = 'odom'
    object_array_msg.robots.append(object_msg)
    return object_array_msg

if __name__ == '__main__':
    # argument parsing:
    parser = argparse.ArgumentParser("Optimizes parameters of a tracking filter")
    parser.add_argument(
        '--debug',
        type=str,
        default='False',
        help="Whether debug messages should be printed")
    parser.add_argument(
        '--trials',
        type=int,
        default='10',
        help="Number of Optuna trials")
    parser.add_argument(
        '--cycles',
        type=int,
        default='100',
        help="Max amount of filter cycles per trial")
    args = parser.parse_args()
    print(args)

    # configuration:
    if args.debug == 'True':
        print("Debug enabled")
    use_noise = True
    # bag_file = '/homes/18hbrandt/Dokumente/rosbag2_2022_12_13-12_34_57_0/rosbag2_2022_12_13-12_34_57_0.db3'
    bag_file = '/home/hendrik/Documents/rosbag2_2022_12_13-12_34_57_0/rosbag2_2022_12_13-12_34_57_0.db3'    # todo bag name from config or somehting?

    # pre-calculation:
    robot_position_true_queue, robot_msg_err_queue, noise_array = generate_msgs(use_noise, bag_file)
    with open('data.json') as data_file:
        data = json.load(data_file)
    data_file.close()

    # create study:
    study = optuna.create_study()

    # start study with set number of trials:
    try:
        study.optimize(objective, n_trials=args.trials)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Aborting optimization study")

    best_params = study.best_params

    # Save best parameter values to output file:
    with open('output_data.json', 'r') as f2:
        data2 = json.load(f2)
    with open('output_data.json', 'w') as f3:
        parameters = []
        for parameter in data["parameters"]:
            parameter_name = parameter["name"]
            parameter_type = parameter["type"]
            if parameter_type == "categorical":
                parameter_choices = parameter["choices"]
                parameters.append({
                    "name": parameter_name,
                    "type": parameter_type,
                    "choices": parameter_choices,
                    "result": best_params[parameter_name]
                })
            else:
                parameter_min = parameter["min"]
                parameter_max = parameter["max"]
                parameters.append({
                    "name": parameter_name,
                    "type": parameter_type,
                    "min": parameter_min,
                    "max": parameter_max,
                    "result": best_params[parameter_name]
                })
            if args.debug == 'True':
                print("Found {}. Best value is: {}".format(parameter_name, best_params[parameter_name]))
        num_previous_trials = len(data2['trial_outputs'])
        trial_output = {
            "trial_number": num_previous_trials,
            "noise_array": noise_array,
            "parameters": parameters
        }
        data2['trial_outputs'].append(trial_output)
        json.dump(data2, f3, indent=4)




