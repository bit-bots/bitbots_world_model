import optuna
import os
import rclpy
import math
from rclpy.node import Node
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from soccer_vision_3d_msgs.msg import RobotArray


class FilterOptimizer(Node):
    def __init__(self) -> None:
        super().__init__('robot_optimizer')
        print("RobotOptimizer initialised")
        self.current_filter_cycle = 0
        self.error_sum = 0
        self.stop_trial = False
        self.current_robot_groundtruth_msg = None
        self.robot_groundtruth_queue = []  # initializing queue
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
        self.robot_relative_groundtruth_publisher = self.create_publisher(
            RobotArray,
            "robots_relative", #config['robot_subscribe_topic'],
            1
        )

        # reset filter:
        # todo get reset service name from config
        robot_filter_reset_service_name = "ball_filter_reset"
        os.system("ros2 service call /{} std_srvs/Trigger".format(robot_filter_reset_service_name))

        # unpack rosbag:
        #self.robot_groundtruth_queue.append()
        # todo bag name from config or somehting?
        # todo unpack bag
        self.publish_robots_relative_groundtruth()

    def publish_robots_relative_groundtruth(self):
        """
        pops first message from the message queue created out of the rosbag and publishes it
        """
        if len(self.robot_groundtruth_queue) > 0:
            self.current_robot_groundtruth_msg = self.robot_groundtruth_queue.pop()
            self.robot_relative_groundtruth_publisher.publish(self.current_robot_groundtruth_msg)
        else:
            rospy.logwarn("Ran out of messages to publish. Stopping trial")
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
        robot = sorted(self.current_robot_groundtruth_msg.robots, key=lambda robot: robot.confidence.confidence)[-1]
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

        self.publish_robots_relative_groundtruth()


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

def objective(trial) -> float:
    """
    Optuna's objective function that runs through a trial, tries out parameter values
    and calculates the evaluation value

    param trial: Optuna trial object
    """

    # suggest parameter values
    # todo for more parameters
    x = trial.suggest_int("x", 1, 10)
    print("Suggestion for filter_reset_distance: " + str(x))
    os.system("ros2 param set /bitbots_ball_filter filter_reset_distance " + str(x))

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
    # create study:
    study = optuna.create_study()
    # start study with set number of trials
    try:
        study.optimize(objective, n_trials=100)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Aborting optimization study")

    # todo proper debug:
    best_params = study.best_params
    found_x = best_params["x"]
    print("Found x: {} as best value".format(found_x))
