import optuna
import os
import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from soccer_vision_3d_msgs.msg import RobotArray


class FilterOptimizer(Node):
    def __init__(self) -> None:
        super().__init__('robot_optimizer')
        print("RobotOptimizer initialised")
        self.filter_cycles = 0
        self.distance_sum = 0
        self.robot_position_true = None  # todo better name
        self.robot_position_filtered = None  # todo better name

        # setup subscriber for ground truth robot positions:
        self.subscriber_robots_relative_true = self.create_subscription(
            RobotArray,
            "robots_relative",  # todo is relative really the ground truth?
            self.robots_relative_true_callback,
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

        # reset filter:
        # todo get reset service name form config
        robot_filter_reset_service_name = "ball_filter_reset"
        os.system("ros2 service call /{} std_srvs/Trigger".format(robot_filter_reset_service_name))
        # start rosbag:
        # todo

    def robots_relative_true_callback(self, robots_relative_true) -> None:
        """
        receives and saves the last value for the ground truth of the robot positions

        param robots_relative_true: ground truth of robot data
        """
        # todo deal with multiple robots instead of just choosing one
        robot = sorted(robots_relative_true.robots, key=lambda robot: robot.confidence.confidence)[-1]
        self.robot_position_true = robot.bb.center.position

    def robots_relative_filtered_callback(self, robots_relative_filtered) -> None:
        """
        receives and saves the last value for the filtered robot data

        param robots_relative_filtered: filtered robot data
        """
        # todo deal with multiple robots
        self.robot_position_filtered = robots_relative_filtered.pose.pose.position

        # calculates distance between last ground truth of the robot positions and the current filtered robot positions
        if self.robot_position_true:
            point_1 = (self.robot_position_true.x,
                       self.robot_position_true.y)
            point_2 = (self.robot_position_filtered.x,
                       self.robot_position_filtered.y)
            distance = math.dist(point_1, point_2)
            print("distance: " + str(distance))
            # distances are added up to create average value later
            self.distance_sum += distance
            self.filter_cycles += 1

    def get_filter_cycles(self) -> int:
        """
        returns number of filter cycles that have been passed through
        """
        return self.filter_cycles

    def get_average_distance(self) -> float:
        """
        returns average distance between ground truth and filtered robot positions for this trial
        """
        return self.distance_sum / self.filter_cycles


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
        while filter_optimizer.get_filter_cycles() < 10:
            rclpy.spin_once(filter_optimizer)
    except KeyboardInterrupt:
        filter_optimizer.destroy_node()
        rclpy.shutdown()

    average_distance = filter_optimizer.get_average_distance()
    filter_optimizer.destroy_node()
    rclpy.shutdown()

    # return evaluation value
    return average_distance


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
