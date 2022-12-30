import optuna
import os
import rclpy
import math
import json
from rclpy.node import Node
from rosbags.rosbag2 import Reader
from rosbags import interfaces
from rosbags.interfaces import Connection
from rosbags.serde import deserialize_cdr
from geometry_msgs.msg import PoseWithCovarianceStamped
from soccer_vision_3d_msgs.msg import RobotArray
import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt


class FilterOptimizer(Node):
    def __init__(self) -> None:
        super().__init__('robot_optimizer')
        print("RobotOptimizer initialised")
        self.current_filter_cycle = 1  # todo change this to 0 again
        self.error_sum = 0
        self.stop_trial = False
        self.current_robot_position_err_msg = None
        self.robot_position_err_queue = []  # initializing queue
        self.robot_position_groundtruth = None
        self.robot_position_filtered = None

        # setup subscriber for ground truth robot positions: todo not necessary
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

        # setup publisher for ground truth from rosbag:
        self.robot_position_err_publisher = self.create_publisher( #todo
            PoseWithCovarianceStamped,
            'position_err',
            1
        )


        # reset filter:
        # todo get reset service name from config
        # robot_filter_reset_service_name = "ball_filter_reset"
        # os.system("ros2 service call /{} std_srvs/Trigger".format(robot_filter_reset_service_name))

        # unpack rosbag:
        # bag_file = '/homes/18hbrandt/Dokumente/rosbag2_2022_12_13-12_34_57_0/rosbag2_2022_12_13-12_34_57_0.db3'
        print("unpacking bag")
        bag_file = '/home/hendrik/Documents/rosbag2_2022_12_13-12_34_57_0/rosbag2_2022_12_13-12_34_57_0.db3'
        parser = BagFileParser(bag_file)
        # todo am I doing this the wrong way around? is this actually getting the last msg first?
        self.robot_position_err_queue = parser.get_messages("/position_err")
        # todo get the non-err positions out of it for distance calculation


        # create reader instance and open for reading
        # bag_file = '/homes/18hbrandt/Dokumente/rosbag2_2022_12_13-12_34_57_0/' \
        #            'rosbag2_2022_12_13-12_34_57_0.db3'
        # with Reader('/homes/18hbrandt/Dokumente/rosbag2_2022_12_13-12_34_57_0') as reader:
        #     for msg in reader.messages():
        #         pass
                #print(msg)
                #https://gitlab.com/ternaris/rosbags/-/blob/master/src/rosbags/interfaces/__init__.py
            # for connection, timestamp, rawdata in reader.messages(['/position']):
            #     msg = deserialize_cdr(rawdata, connection.msgtype)
            #     print(msg.header.frame_id)
        #self.robot_groundtruth_queue.append()
        # todo bag name from config or somehting?
        # todo unpack bag
        self.publish_robot_position_err()

    def publish_robot_position_err(self):
        """
        pops first message from the message queue created out of the rosbag and publishes it
        """
        if len(self.robot_position_err_queue) > 0:
            self.current_robot_position_err_msg = self.robot_position_err_queue.pop()
            print(self.current_robot_position_err_msg[1])
            self.robot_position_err_publisher.publish(self.current_robot_position_err_msg[1])  # the msg is a tuple of time stamp + msg
        else:
            FilterOptimizer.get_logger(self).warn("Ran out of messages to publish. Stopping trial")
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
        robot = sorted(self.current_robot_position_err_msg.robots, key=lambda robot: robot.confidence.confidence)[-1]
        self.robot_position_groundtruth = robot.bb.center.position
        point_1 = (self.robot_position_groundtruth.x,
                   self.robot_position_groundtruth.y)
        point_2 = (self.robot_position_filtered.x,
                   self.robot_position_filtered.y)
        distance = math.dist(point_1, point_2)
        print("error: " + str(distance))
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


def objective(trial) -> float:
    """
    Optuna's objective function that runs through a trial, tries out parameter values
    and calculates the evaluation value

    param trial: Optuna trial object
    """

    # suggest parameter values
    f = open('data.json')
    data = json.load(f)
    for parameter in data["parameters"]:
        temp = None
        if parameter["type"] == "int":
            temp = trial.suggest_int(parameter["name"], parameter["min"], parameter["max"])
        elif parameter["type"] == "float":
            temp = trial.suggest_float(parameter["name"], parameter["min"], parameter["max"])
        elif parameter["type"] == "categorical":
            temp = trial.suggest_categorical(parameter["name"], parameter["choices"])
        print("Suggestion for " + parameter["name"] + ": " + str(temp))
        os.system("ros2 param set /bitbots_ball_filter " + parameter["name"] + " " + str(temp))
    f.close()
    # start filter optimizer
    # todo
    rclpy.init()
    filter_optimizer = FilterOptimizer()
    try:
        while filter_optimizer.get_filter_cycles() < 10 and not filter_optimizer.get_stop_trial():
            rclpy.spin_once(filter_optimizer)
    except KeyboardInterrupt:
        filter_optimizer.destroy_node()
        rclpy.shutdown()

    average_error = filter_optimizer.get_average_error()
    filter_optimizer.destroy_node()
    rclpy.shutdown()

    # return evaluation value
    return average_error


if __name__ == '__main__':
    #todo

    # p_des_1 = [trajectory.points[i].positions[0] for i in range(len(trajectory.points))]
    # t_des = [trajectory.points[i].time_from_start.sec + trajectory.points[i].time_from_start.nanosec*1e-9 for i in range(len(trajectory.points))]

    # plt.plot(t_des, p_des_1)
    #
    # plt.show()

    # create study:
    study = optuna.create_study()
    # start study with set number of trials
    try:
        study.optimize(objective, n_trials=1)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Aborting optimization study")

    # todo proper debug:
    best_params = study.best_params

    print("_HERE")
    with open('data.json') as f:
        data = json.load(f)
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
            print("Found {}. Best value is: {}".format(parameter_name, best_params[parameter_name]))
        num_previous_trials = len(data2['trial_outputs'])
        print(num_previous_trials)
        trial_output = {
            "trial_number": num_previous_trials,
            "parameters": parameters
        }
        data2['trial_outputs'].append(trial_output)
        json.dump(data2, f3, indent=4)




