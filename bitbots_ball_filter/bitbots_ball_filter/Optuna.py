import optuna
import os
import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from soccer_vision_3d_msgs.msg import RobotArray

class RobotOptimizer(Node):
    def __init__(self) -> None:
        super().__init__('robot_optimizer')
        print("RobotOptimizer initialised")
        self.testcounter = 0
        self.distance_sum = 0
        self.robot_position_true = None
        self.robot_position_filtered = None
        self.subscriber_robots_relative_true = self.create_subscription(
            RobotArray,
            "robots_relative", #todo is relative really the ground truth?
            self.robots_relative_true_callback,
            1
        )
        self.subscriber_robots_relative_filtered = self.create_subscription(
            PoseWithCovarianceStamped,
            "robot_position_relative_filtered",
            self.robots_relative_filtered_callback,
            1
        )

    def robots_relative_true_callback(self, robots_relative_true):
        robot = sorted(robots_relative_true.robots, key=lambda robot: robot.confidence.confidence)[-1]
        self.robot_position_true = robot.bb.center.position

    def robots_relative_filtered_callback(self, robots_relative_filtered):
        self.robot_position_filtered = robots_relative_filtered.pose.pose.position
        if self.robot_position_true:
            point_1 = (self.robot_position_true.x,
                       self.robot_position_true.y)
            point_2 = (self.robot_position_filtered.x,
                       self.robot_position_filtered.y)
            distance = math.dist(point_1, point_2)
            print("distance: " + str(distance))
            self.distance_sum += distance
            self.testcounter += 1

    def get_testcounter(self) -> int:
        return self.testcounter

    def get_average_distance(self) -> float:
        return self.distance_sum / self.testcounter


def objective(trial):
    x = trial.suggest_int("x", 1, 10)
    print("Suggestion for filter_reset_distance: " + str(x))
    os.system("ros2 param set /bitbots_ball_filter filter_reset_distance " + str(x))


    rclpy.init()
    robot_optimizer = RobotOptimizer()
    try:
        while robot_optimizer.get_testcounter() < 10:
            # print(robot_optimizer.get_testcounter())
            rclpy.spin_once(robot_optimizer)
    except KeyboardInterrupt:
        robot_optimizer.destroy_node()
        rclpy.shutdown()

    average_distance = robot_optimizer.get_average_distance()
    robot_optimizer.destroy_node()
    rclpy.shutdown()

    return average_distance



if __name__ == '__main__':
    # Optuna stuff:
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    found_x = best_params["x"]
    print("Found x: {} as best value for filter_reset_distance".format(found_x))













