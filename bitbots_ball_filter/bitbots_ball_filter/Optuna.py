import time

import optuna
import os
import rclpy
import signal
import math
import random
import json
import argparse
import subprocess
import numpy as np
import timeit
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
        self.error_measure = error_measure
        self.error_test_array = error_test_array

        self.robot_groundtruth_msg_queue = robot_position_true_queue.copy()
        self.robot_groundtruth_msg_queue.extend(robot_position_true_queue.copy()) #todo remove
        self.robot_detection_msg_queue = robot_msg_err_queue.copy()
        self.robot_detection_msg_queue.extend(robot_msg_err_queue.copy()) #todo remove
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
            'robot_position_relative_filtered',
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
            self.logger.info('Waiting for subscriber...')
            time.sleep(0.5)

        # either create a timer to step at a fixed rate or only step when a response from the filter is received:
        if self.use_timer:
            self.optimizer_timer = self.create_timer(self.optimizer_time_step, self.optimizer_step)
        else:
            self.optimizer_step()

    def optimizer_step(self):
        '''
        Performs an optimizer step in which the groundtruth and noisy detected position are advanced
        with the latter one being published to be filtered.
        '''
        #todo remove
        average_error = 0
        if self.current_filter_cycle > 0:
            average_error = self.error_sum/self.current_filter_cycle
        if self.use_debug:
            print('stepping optimizer, opti cycle: {}, filter cycle: {}, avg. error: {}'.format(self.current_optimizer_cycle, self.current_filter_cycle, average_error))
        if len(self.robot_detection_msg_queue) > 0 and len(self.robot_groundtruth_msg_queue) > 0:
            # the elements in the que are tuple of time stamp + msg
            self.current_robot_groundtruth_msg = self.robot_groundtruth_msg_queue.pop(0)[1]
            self.current_robot_detection_msg = self.robot_detection_msg_queue.pop(0)[1]
            self.robot_position_detection_publisher.publish(self.current_robot_detection_msg)
            if self.use_debug:
                print('published message')
        else:
            self.logger.warn('Ran out of messages to publish. Stopping trial')
            self.stop_trial = True
            raise KeyboardInterrupt
        self.current_optimizer_cycle += 1

    def robot_position_filtered_callback(self, robots_relative_filtered) -> None:
        '''
        Receives and saves the last value for the filtered robot data.

        param robots_relative_filtered: Filtered robot data.
        '''
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
        if self.error_measure == 'RMSE':
            self.error_sum += distance ** 2
        elif self.error_measure == 'MAE':
            self.error_sum += distance
        else:
            self.logger.error('this error measure is not implemented')
        self.current_filter_cycle += 1

        # step optimizer manually if it doesn't run on timer:
        if not self.use_timer:
            #print(self.current_robot_filtered_position)
            self.optimizer_step()

    def get_optimizer_cycles(self) -> int:
        '''
        Returns number of filter cycles that have been passed through.
        '''
        return self.current_optimizer_cycle

    def get_average_error(self) -> float:
        '''
        Returns average error based on the distance between ground truth and filtered robot positions for this trial.
        If no filter cycles occured
        '''
        if self.current_filter_cycle > 0:
            if self.error_measure == 'RMSE':
                average_error = math.sqrt( self.error_sum / self.current_filter_cycle )
            elif self.error_measure == 'MAE':
                average_error = self.error_sum / self.current_filter_cycle
            else:
                self.logger.error('this error measure is not implemented')
                average_error = 999
            return average_error
        else:
            return 999

    def ran_out_of_time(self):
        if self.timeout_incidents >= 1:
            self.stop_trial = True
            string = 'WARNING: Trial ran out of time'
            print(f'\033[33m{string}\033[0m')
        self.timeout_incidents += 1

    def get_stop_trial(self):
        '''
        Returns boolean whether trial should be stopped prematurely.
        '''
        return self.stop_trial

class BagFileParser:
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()

        ## create a message type map
        topics_data = self.cursor.execute('SELECT id, name, type FROM topics').fetchall()
        self.topic_type = {name_of:type_of for id_of,name_of,type_of in topics_data}
        self.topic_id = {name_of:id_of for id_of,name_of,type_of in topics_data}
        self.topic_msg_message = {name_of:get_message(type_of) for id_of,name_of,type_of in topics_data}

    def __del__(self):
        self.conn.close()

    # Return [(timestamp0, message0), (timestamp1, message1), ...]
    def get_messages(self, topic_name):

        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute('SELECT timestamp, data FROM messages WHERE topic_id = {}'.format(topic_id)).fetchall()
        # Deserialise all and timestamp them
        return [ (timestamp,deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp,data in rows]

def objective(trial):
    """
    Optuna's objective function that runs through a trial, tries out parameter values
    and calculates the evaluation value.

    param trial: Optuna trial object.
    """
    # signal.signal(signal.SIGALRM, signal_handler)
    # signal.alarm(100)  # raises TimeoutException after specified number of seconds
    trial_optimizer = None
    filter_process = None
    # try:
    # suggest parameter values
    params = {}
    param_str = ''
    for input_parameter in data['parameters']:
        temp = None
        # Consider each case of parameter type:
        if input_parameter['type'] == 'int':
            temp = trial.suggest_int(input_parameter['name'], input_parameter['min'], input_parameter['max'])
        elif input_parameter['type'] == 'float':
            temp = trial.suggest_float(input_parameter['name'], input_parameter['min'], input_parameter['max'])
        elif input_parameter['type'] == 'categorical':
            temp = trial.suggest_categorical(input_parameter['name'], input_parameter['choices'])
        if use_debug:
            print("Suggestion for " + input_parameter["name"] + ": " + str(temp))
        param_str += input_parameter["name"] + "#" + str(temp) + "#"  # todo change this to dict
        # params["{}".format(input_parameter["name"])] = "{}".format(str(temp))
    param_str += "trial_number" + "#" + str(trial.number) + "#"

    # save params to file so that the filter can access them
    with open('text_params.txt', 'w') as file:
        file.write(param_str)
    file.close()

    # initialize robot filter:
    if use_debug:
        print('starting robot filter')
    filter_process = subprocess.Popen(['ros2','launch','bitbots_ball_filter','robot_filter.launch'])

    # start filter optimizer
    rclpy.init()
    exception = False
    pruned = False
    if use_debug:
        print('Initializing trial optimizer...')
    trial_optimizer = TrialOptimizer(trial)
    try:
        while not trial_optimizer.get_stop_trial()\
                and trial_optimizer.get_optimizer_cycles() < args.cycles:
            rclpy.spin_once(trial_optimizer)
            trial.report(trial_optimizer.get_average_error(), trial_optimizer.get_optimizer_cycles())
            if trial.should_prune():
                pruned = True
                string = 'pruning trial after {} cycles'.format(trial_optimizer.get_optimizer_cycles())
                print(f'\033[33m{string}\033[0m')
                break
    except KeyboardInterrupt:
        exception = True
    except TypeError:
        string = 'ERROR: Optimizer node ran into problem'
        print(f'\033[31m{string}\033[0m')
        exception = True
    # except TimeoutException:
    #     string = 'ERROR: Optimizer timed out'
    #     print(f'\033[31m{string}\033[0m')
    #     exception = True

    # cleanup:
    rclpy.shutdown()
    # kill the robot filter:
    #os.system('ros2 param set /bitbots_ball_filter selfdestruct True')
    filter_alive_flag = True
    while filter_alive_flag:  # make sure to only terminate the subprocess once the filter has been killed
        os.system('ros2 param set /bitbots_ball_filter selfdestruct True')
        nodes = str(subprocess.check_output(['ros2', 'node', 'list']), 'utf-8')
        # go through nodes and repeat if filter is still alive:
        for node in nodes.split('\n'):
            if node == '/bitbots_ball_filter':
                print('Waiting for filter to die...')
                filter_process.terminate()
                filter_alive_flag = True
                break
            else:
                filter_alive_flag = False
    if use_debug:
        print('killed filter node')

    # check if trial has reached invalid result
    if trial_optimizer:
        average_error = trial_optimizer.get_average_error()
        trial_optimizer.destroy_node()
    else:
        average_error = -1
    if exception or average_error == -1:  # exception case
        string = 'WARNING: Trial invalid'
        print(f'\033[33m{string}\033[0m')
        average_error_array.append(-1)
        failed_trial_array.append(trial.params)
        return None
    elif pruned:  # pruning case
        average_error_array.append(-2)
        raise optuna.TrialPruned()
    else:  # normal case
        average_error_array.append(average_error)
        return average_error


def signal_handler(signum, frame):
    raise TimeoutException('Timed out!')

class TimeoutException(Exception): pass

def generate_msgs() -> None:
    '''
    Extracts messages from rosbag and creates noisy detected positions from noise and the groundtruth if desired.

    param bag_file: Source bag file containing the messages to be extracted.
    param use_noise: Whether to generate noisy detected positions.
    param max_error: The maximum error of the noise. Only necessary when use_noise is True.
    '''
    # unpack rosbag:
    if use_debug:
        print('Unpacking bag')
    bag_parser = BagFileParser(bag_file)
    # extract groundtruth:
    for msg in bag_parser.get_messages('/position'):
        robot_position_true_queue.append(msg)
    # extract noisy position:
    if use_noise != 'True':
        # use available noisy position:
        for msg in bag_parser.get_messages('/robots_relative'):
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
        print('Message extraction finished')

def gen_robot_array_msg(x, y, timestamp) -> RobotArray:
    '''
    Generates a RobotArray message that contains one robot.

    param x: Position of the robot on the x-axis.
    param y: Position of the robot on the y-axis.
    param timestamp: Timestamp of the source message.
    '''
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
    parser = argparse.ArgumentParser('Optimizes parameters of a tracking filter')
    parser.add_argument(
        '--debug',
        type=str,
        default='False',
        choices=['true', 'True', 'false', 'False'],
        help='Whether debug messages should be printed')
    parser.add_argument(
        '--trials',
        type=int,
        default='100',
        help='Number of Optuna trials')
    parser.add_argument(
        '--cycles',
        type=int,
        default='100',
        help='Max amount of optimizer cycles and therefore messages send to the filter per trial')
    parser.add_argument(
        '--noise',
        type=str,
        default='False',
        choices=['true', 'True', 'false', 'False'],
        help='Adds noise to the groundtruth position from the rosbag to create detected position '
             'that is used in optimization and logs the noise. '
             'Otherwise the noisy detected position from the rosbag is used')
    parser.add_argument(
        '--error',
        type=str,
        default='RMSE',
        choices=['RMSE', 'MAE'],
        help='Decides the type of error calculation used. Either RMSE for root-mean-square error '
             'or MAE for mean absolute error')
    parser.add_argument(
        '--bag',
        type=str,
        default='rosbag2_2022_12_13-12_34_57_0/rosbag2_2022_12_13-12_34_57_0.db3',
        help='Path to the rosbag containing the messages')
    parser.add_argument(
        '--plot',
        type=str,
        default='False',
        choices=['true', 'True', 'false', 'False'],
        help='Decides whether average error of the study should be plotted at the end')
    parser.add_argument(
        '--prune',
        type=str,
        default='False',
        choices=['true', 'True', 'false', 'False'],
        help='Decides whether a pruner should be used for the study'
    )
    parser.add_argument(
        '--noise_size',
        type=float,
        default=1.0,
        help='Determines the level of noise for each dimension'
    )
    parser.add_argument(#todo choices and implementation
        '--sampler',
        type=str,
        default='TPESampler',
        help='Decides the sampler for the study'
    )
    parser.add_argument(#todo choices and implementation
        '--pruner',
        type=str,
        default='MedianPruner',
        help='Decides the pruner for the study'
    )
    args = parser.parse_args()

    # configuration:
    if args.noise == 'True' or args.noise == 'true':
        use_noise = True
    else:
        use_noise = False
    if args.debug == 'True' or args.debug == 'true':
        use_debug = True
    else:
        use_debug = False
    if args.plot == 'True' or args.plot == 'true':
        do_plot = True
    else:
        do_plot = False
    if args.sampler == 'RandomSampler':
        sampler = optuna.samplers.RandomSampler()
    else:
        sampler = optuna.samplers.TPESampler()
    if args.prune == 'True' or args.prune == 'true':
        #pruner = optuna.pruners.ThresholdPruner(upper=10)
        #pruner = optuna.pruners.HyperbandPruner(min_resource=args.cycles * 0.25, max_resource=args.cycles)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(round(args.trials * 0.1)),
            n_warmup_steps=int(round(args.cycles * 0.25)),
            #interval_steps=int(round(args.cycles * 0.1)),
            n_min_trials=int(round(args.trials * 0.1))
            )
    else:
        pruner = optuna.pruners.NopPruner()  # pruner that never prunes
    max_error = np.array([args.noise_size,args.noise_size])
    noise_array = []
    error_measure = args.error
    failed_trial_array = []
    average_error_array = []
    robot_msg_err_queue = []
    robot_position_true_queue = []
    error_test_array = []
    bag_file = args.bag
    if use_debug:
        print('Debug enabled')

    # pre-calculation:
    generate_msgs()
    with open('dataMid.json') as file:
        data = json.load(file)
    file.close()

    # create study:
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(),
                                pruner=pruner
                                )

    # start study with set number of trials:
    retry = True
    while retry:
        retry = False
        start = time.time()
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(1800)  # raises TimeoutException after specified number of seconds
        try:
            study.optimize(objective, n_trials=args.trials)
        except KeyboardInterrupt:
            print('Keyboard interrupt. Aborting optimization study')
        except TimeoutException:
            string = 'ERROR: Study timed out, Retrying...'
            print(f'\033[31m{string}\033[0m')
            retry = True
        end = time.time()

    # save best parameter values to output file:
    best_params = study.best_params
    with open('output_data2.json', 'r') as file:
        output_data = json.load(file)
    file.close()
    with open('output_data2.json', 'w') as file:
        parameters = []
        for parameter in data['parameters']:
            parameter_name = parameter['name']
            parameter_type = parameter['type']
            if parameter_type == 'categorical':
                parameter_choices = parameter['choices']
                parameters.append({
                    'name': parameter_name,
                    'type': parameter_type,
                    'choices': parameter_choices,
                    'result': best_params[parameter_name]
                })
            else:
                parameter_min = parameter['min']
                parameter_max = parameter['max']
                parameters.append({
                    'name': parameter_name,
                    'type': parameter_type,
                    'min': parameter_min,
                    'max': parameter_max,
                    'result': best_params[parameter_name]
                })
            if use_debug:
                print('Found {}. Best value is: {}'.format(parameter_name, best_params[parameter_name]))
        num_previous_studies = len(output_data['study_outputs'])
        trial_output = {
            'study_number': num_previous_studies,
            'time': end - start,
            'noise_size': args.noise_size,
            'trials': args.trials,
            'cycles': args.cycles,
            'sampler':args.sampler,
            'pruner':args.pruner,
            'error':args.error,
            'noise_array': noise_array,
            'failed_trial_params': failed_trial_array,
            'average_error_array': average_error_array,
            'parameters': parameters
        }
        output_data['study_outputs'].append(trial_output)
        json.dump(output_data, file, indent=4)
    file.close()

    # plot results:
    if do_plot:

        average_error_array_x = []
        average_error_array_y = []
        for i in range(0, len(average_error_array) - 1):
            if average_error_array[i] >= 0:
                average_error_array_x.append(i)
                average_error_array_y.append(average_error_array[i])
            elif average_error_array[i] == -2:
                string = 'INFO: Trial {} was pruned and will not be shown'.format(i)
                print(string)
            else:
                string = 'WARNING: Trial {} failed and will not be shown'.format(i)
                print(f'\033[33m{string}\033[0m')

        plt.plot(average_error_array_x, average_error_array_y,
                 label='average error',
                 lw=1.5,
                 c="black")

        plt.xlabel('Trial Number')
        plt.ylabel('Average Error')
        plt.title('Study Results')
        plt.show()




