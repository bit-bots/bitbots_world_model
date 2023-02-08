import time

import optuna
import os
import rclpy
import math
import json
import argparse
import subprocess
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from soccer_vision_3d_msgs.msg import RobotArray, Robot
import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt


class TrialOptimizer(Node):
    def __init__(self, trial) -> None:
        super().__init__('trial_optimizer')
        self.logger = self.get_logger()
        self.current_filter_cycle = 0
        self.current_optimizer_cycle = 0
        self.error_sum = 0
        self.stop_trial = False
        self.use_debug = use_debug
        self.timeout_time_step = 60
        self.timeout_incidents = 0
        self.trial = trial
        self.error_test_array = error_test_array

        self.robot_groundtruth_msg_queue = robot_position_true_queue.copy()
        self.robot_detection_msg_queue = robot_msg_err_queue.copy()
        self.current_robot_groundtruth_position = None
        self.current_robot_groundtruth_msg = self.robot_groundtruth_msg_queue.pop(0)[1]
        self.current_robot_detection_msg = self.robot_detection_msg_queue.pop(0)[1]
        self.current_robot_filtered_position = None
        self.current_robot_filtered_msg = None

        self.optimizer_rate = 60 # todo get from config
        self.optimizer_time_step = 1.0 / self.optimizer_rate
        self.use_timer = False

        # setup subscriber for filtered robot positions:
        self.robot_position_filtered_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            "robot_position_relative_filtered",
            self.robot_position_filtered_callback,
            1
        )

        # setup publisher for robot detection with noise:
        self.robot_position_detection_publisher = self.create_publisher(
            RobotArray,
            'robots_relative',
            1
        )

        if self.use_debug:
            print('...initialization finished')
        self.create_timer(self.timeout_time_step, self.ran_out_of_time())

        # wait until robot filter has been created
        while self.robot_position_detection_publisher.get_subscription_count() == 0:
            self.logger.info("Waiting for robot filter...")
            time.sleep(0.5)

        # either create a timer to step at a fixed rate or only step when a response from the filter is received:
        if self.use_timer:
            self.optimizer_timer = self.create_timer(self.optimizer_time_step, self.optimizer_step)
        else:
            self.optimizer_step()

    def optimizer_step(self):
        """
        Performs an optimizer step in which the groundtruth and noisy detected position are advanced
        with the latter one being published to be filtered.
        """
        #if self.use_debug:
        if True:
            print('stepping optimizer, opti cycle: {}, filter cycle: {}'.format(self.current_optimizer_cycle, self.current_filter_cycle))
        if len(self.robot_detection_msg_queue) > 0 and len(self.robot_groundtruth_msg_queue) > 0:
            # the elements in the que are tuple of time stamp + msg
            self.current_robot_groundtruth_msg = self.robot_groundtruth_msg_queue.pop(0)[1]
            self.current_robot_detection_msg = self.robot_detection_msg_queue.pop(0)[1]
            self.robot_position_detection_publisher.publish(self.current_robot_detection_msg)
            if self.use_debug:
                print('published message')
        else:
            self.logger.warn("Ran out of messages to publish. Stopping trial")
            self.stop_trial = True
        self.current_optimizer_cycle += 1

    def robot_position_filtered_callback(self, robots_relative_filtered) -> None:
        """
        Receives and saves the last value for the filtered robot data.

        param robots_relative_filtered: Filtered robot data.
        """
        if self.use_debug:
            print('handling filtered msg callback')

        self.current_robot_filtered_msg = robots_relative_filtered
        self.current_robot_filtered_position = self.current_robot_filtered_msg.pose.pose.position

        # calculates the error based on the distance between last groundtruth and the current filtered robot position:
        self.current_robot_groundtruth_position = self.current_robot_groundtruth_msg.pose.pose.position

        #groundtruth = self.current_robot_groundtruth_position[1]
        point_1 = (self.current_robot_groundtruth_position.x,
                   self.current_robot_groundtruth_position.y)
        point_2 = (self.current_robot_filtered_position.x,
                   self.current_robot_filtered_position.y)
        distance = math.dist(point_1, point_2)

        # distances are added up to create average value later
        self.error_sum += distance
        self.current_filter_cycle += 1

        # step optimizer manually if it doesn't run on timer:
        if not self.use_timer:
            self.optimizer_step()

    def get_optimizer_cycles(self) -> int:
        """
        Returns number of filter cycles that have been passed through.
        """
        return self.current_optimizer_cycle

    def get_average_error(self) -> float:
        """
        Returns average error based on the distance between ground truth and filtered robot positions for this trial.
        If no filter cycles occured
        """
        if self.current_filter_cycle > 0:
            return self.error_sum / self.current_filter_cycle
        else:
            return 999

    def ran_out_of_time(self):
        if self.timeout_incidents >= 1:
            self.stop_trial = True
            string = 'WARNING: Trial ran out of time'
            print(f'\033[33m{string}\033[0m')
        self.timeout_incidents += 1

    def get_stop_trial(self):
        """
        Returns boolean whether trial should be stopped prematurely.
        """
        return self.stop_trial

class BagFileParser:
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

def objective(trial) -> float:
    """
    Optuna's objective function that runs through a trial, tries out parameter values
    and calculates the evaluation value.

    param trial: Optuna trial object.
    """

    # initialize robot filter:
    if use_debug:
        pass
    print("starting robot filter")
    ball_filter_process = subprocess.Popen(["ros2","launch","bitbots_ball_filter","robot_filter.launch"])

    # suggest parameter values
    if use_debug:
        debug_var = ''
    else:
        debug_var = '> /dev/null 2>&1'
    # create param dictionary:
    params = {}
    param_str = ""
    for input_parameter in data["parameters"]:
        temp = None
        if input_parameter["type"] == "int":
            temp = trial.suggest_int(input_parameter["name"], input_parameter["min"], input_parameter["max"])
        elif input_parameter["type"] == "float":
            temp = trial.suggest_float(input_parameter["name"], input_parameter["min"], input_parameter["max"])
        elif input_parameter["type"] == "categorical":
            temp = trial.suggest_categorical(input_parameter["name"], input_parameter["choices"])
        if use_debug:
            print("Suggestion for " + input_parameter["name"] + ": " + str(temp))
        param_str += input_parameter["name"] + "#" + str(temp) + "#"
        # params["{}".format(input_parameter["name"])] = "{}".format(str(temp))
    # set the param values:
    # os.system("ros2 param set /bitbots_ball_filter {} {} {}".format("adjusted_params", str(params), debug_var))
    param_str += "trial_number" + "#" + str(trial.number) + "#"
    #print("ros2 param set /bitbots_ball_filter adjusted_params " + str(param_str) + str(debug_var))

    os.system("ros2 param set /bitbots_ball_filter adjusted_params " + str(param_str) + str(debug_var))


    # start filter optimizer
    rclpy.init()
    exception = False
    if use_debug:
        print("Initializing trial optimizer...")
    trial_optimizer = TrialOptimizer(trial)
    try:
        while not trial_optimizer.get_stop_trial()\
                and trial_optimizer.get_optimizer_cycles() < args.cycles:
            rclpy.spin_once(trial_optimizer)
    except KeyboardInterrupt:
        trial_optimizer.destroy_node()
        rclpy.shutdown()
        exception = True
    # except TypeError:
    #     if trial_optimizer:
    #         trial_optimizer.destroy_node()
    #     rclpy.shutdown()
    #     string = 'ERROR: Optimizer node got destroyed'
    #     print(f'\033[31m{string}\033[0m')
    #     exception = True

    # cleanup:
    # kill the robot filter:
    # process = subprocess.Popen(["ros2","param", "set", "/bitbots_ball_filter", "selfdestruct", "True"])
    os.system("ros2 param set /bitbots_ball_filter selfdestruct True")
    time.sleep(1)
    ball_filter_process.terminate()
    average_error = trial_optimizer.get_average_error()
    if not exception:
        trial_optimizer.destroy_node()
        rclpy.shutdown()

    # return evaluation value
    average_error_array.append(average_error)

    # check if trial has reached invalid result
    if average_error == 999:
        string = 'WARNING: Trial invalid'
        print(f'\033[33m{string}\033[0m')
        average_error = None
        failed_trial_array.append(trial.params)


    return average_error

def generate_msgs() -> None:
    """
    Extracts messages from rosbag and creates noisy detected positions from noise and the groundtruth if desired.

    param bag_file: Source bag file containing the messages to be extracted.
    param use_noise: Whether to generate noisy detected positions.
    param max_error: The maximum error of the noise. Only necessary when use_noise is True.
    """
    # unpack rosbag:
    if use_debug:
        print("Unpacking bag")
    bag_parser = BagFileParser(bag_file)
    # extract groundtruth:
    for msg in bag_parser.get_messages("/position"):
        robot_position_true_queue.append(msg)
    # extract noisy position:
    if use_noise != 'True':
        # use available noisy position:
        for msg in bag_parser.get_messages("/robots_relative"):
            robot_msg_err_queue.append(msg)
    else:
        # generate own noisy position:
        for i in range(0, len(robot_position_true_queue)):
            error = np.multiply(np.random.rand(2) * 2 - 1, max_error)
            noise_array.append(error.tolist())  # save error for later visualization
            groundtruth_msg = robot_position_true_queue[i][1]  # takes the msg from the timestamp + msg tuple
            p_err = [groundtruth_msg.pose.pose.position.x,
                     groundtruth_msg.pose.pose.position.y] + error
            msg_tuple = [robot_position_true_queue[i][0],
                     gen_robot_array_msg(p_err[0], p_err[1], groundtruth_msg.header.stamp)]
            robot_msg_err_queue.append(msg_tuple)
    if use_debug:
        print("Message extraction finished")

def gen_robot_array_msg(x, y, timestamp) -> RobotArray:
    """
    Generates a RobotArray message that contains one robot.

    param x: Position of the robot on the x-axis.
    param y: Position of the robot on the y-axis.
    param timestamp: Timestamp of the source message.
    """
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
        default='100',
        help="Number of Optuna trials")
    parser.add_argument(
        '--cycles',
        type=int,
        default='100',
        help="Max amount of optimizer cycles and therefore messages send to the filter per trial")
    parser.add_argument(
        '--noise',
        type=str,
        default=False,
        help='Adds noise to the groundtruth position from the rosbag to create detected position '
             'that is used in optimization and logs the noise. '
             'Otherwise the noisy detected position from the rosbag is used')
    args = parser.parse_args()
    print(args)

    # configuration:
    use_noise = args.noise
    if args.debug == 'True':
        use_debug = True
    else:
        use_debug = False
    max_error = np.array([1,1])
    noise_array = []
    failed_trial_array = []
    average_error_array = []
    robot_msg_err_queue = []
    robot_position_true_queue = []
    error_test_array = []
    # bag_file = '/homes/18hbrandt/Dokumente/rosbag2_2022_12_13-12_34_57_0/rosbag2_2022_12_13-12_34_57_0.db3'
    bag_file = '/home/hendrik/Documents/rosbag2_2022_12_13-12_34_57_0/rosbag2_2022_12_13-12_34_57_0.db3'    # todo bag name from config or somehting?
    if use_debug:
        print("Debug enabled")

    # pre-calculation:
    generate_msgs()
    with open('data.json') as f1:
        data = json.load(f1)
    f1.close()

    # create study:
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())

    # start study with set number of trials:
    try:
        study.optimize(objective, n_trials=args.trials)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Aborting optimization study")

    # Save best parameter values to output file:
    best_params = study.best_params
    with open('output_data.json', 'r') as f2:
        output_data = json.load(f2)
    f2.close()

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
            if use_debug:
                print("Found {}. Best value is: {}".format(parameter_name, best_params[parameter_name]))
        num_previous_studies = len(output_data['study_outputs'])
        trial_output = {
            "study_number": num_previous_studies,
            "noise_array": noise_array,
            "failed_trial_params": failed_trial_array,
            "average_error_array": average_error_array,
            "parameters": parameters
        }
        output_data['study_outputs'].append(trial_output)
        json.dump(output_data, f3, indent=4)
    f3.close()

    plt.plot(average_error_array,
             label='average error',
             lw=1.5,
             c="blue")
    plt.xticks(range(0, len(average_error_array) + 1 , 1))

    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('My first graph!')
    plt.legend()
    plt.show()




