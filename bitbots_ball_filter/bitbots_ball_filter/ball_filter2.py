#! /usr/bin/env python3
from typing import Union, Tuple

import math
import numpy as np

import rclpy
import tf2_ros as tf2

from copy import deepcopy
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

from rclpy.node import Node
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, TwistWithCovarianceStamped
from tf2_geometry_msgs import PointStamped
from humanoid_league_msgs.msg import PoseWithCertaintyStamped
from soccer_vision_3d_msgs.msg import RobotArray, Robot

class RobotWrapper():
    #todo  what does this do
    def __init__(self, position, header, confidence):
        self.position = position
        self.header = header
        self.confidence = confidence

    def get_header(self):
        return self.header

    def get_position(self):
        return self.position

    def get_confidence(self):
        return self.confidence

class ObjectFilter(Node):
    def __init__(self) -> None:
        #todo what do these do
        super().__init__("robot_filter", automatically_declare_parameters_from_overrides=True)
        self.logger = self.get_logger()
        self.tf_buffer = tf2.Buffer(cache_time=rclpy.duration.Duration(seconds=2))
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self)

        # Setup dynamic reconfigure config
        self.config = {}
        self.add_on_set_parameters_callback(self._dynamic_reconfigure_callback)
        self._dynamic_reconfigure_callback(self.get_parameters_by_prefix("").values())

    def _dynamic_reconfigure_callback(self, config) -> SetParametersResult:
        """
        todo

        paran config: todo
        """
        # construct config from the params:
        tmp_config = deepcopy(self.config)
        for param in config:
            tmp_config[param.name] = param.value
        config = tmp_config

        # create Kalman filter:
        self.kf = KalmanFilter(dim_x=4, dim_z=2, dim_u=0)
        self.filter_initialized = False
        self.robot = None  # type: RobotWrapper
        self.last_robot_stamp = None

        self.filter_rate = config['filter_rate'] #todo replace
        self.measurement_certainty = config['measurement_certainty']
        self.filter_time_step = 1.0 / self.filter_rate
        self.filter_reset_duration = rclpy.duration.Duration(seconds=config['filter_reset_time'])
        self.filter_reset_distance = config['filter_reset_distance']
        self.closest_distance_match = config['closest_distance_match']

        #todo what does this do
        filter_frame = config['filter_frame']
        if filter_frame == "odom":
            self.filter_frame = config['odom_frame']
        elif filter_frame == "map":
            self.filter_frame = config['map_frame']
        self.logger.info(f"Using frame '{self.filter_frame}' for robot filtering")

        # adapt velocity factor to frequency#todo whats this OWW
        self.velocity_factor = (1 - config['velocity_reduction']) ** (1 / self.filter_rate)
        self.process_noise_variance = config['process_noise_variance']

        # setup publishers and subscribers:

        #todo what other attributes for the robot would need to be published:

        # publishes positions of robots:
        self.robot_position_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            config['robot_position_publish_topic'],
            1
        )
        # publishes velocity of robots:
        self.robot_movement_publisher = self.create_publisher(
            TwistWithCovarianceStamped,
            config['robot_movement_publish_topic'],
            1
        )
        # publishes robot:
        self.robot_publisher = self.create_publisher(
            PoseWithCertaintyStamped,
            config['robot_publish_topic'],
            1
        )
        # setup robot subscriber:
        self.subscriber = self.create_subscription(
            RobotArray,
            config['robot_subscribe_topic'],
            self.robot_callback,
            1
        )
        # setup reset service
        self.reset_service = self.create_service(
            Trigger,
            config['robot_filter_reset_service_name'],
            self.reset_filter_callback
        )

        #todo redo:
        self.config = config
        self.filter_timer = self.create_timer(self.filter_time_step, self.filter_step)
        return SetParametersResult(successful=True)

    def reset_filter_callback(self, req, response) -> Tuple[bool, str]:
        """
        resets the filter when the reset trigger was received

        paran req: todo
        param response: todo
        """
        self.logger.info("Resetting bitbots robot filter...")
        self.filter_initialized = False
        response.success = True
        return response


    def robot_callback(self, msg: RobotArray) -> None:
        """
        Assigns each detected robot to existing or new filter
        #todo the original just decides which ball to take since it doesnt need to assign filters
        #todo so do I already assign filters here or later?

        :param robot_msg: List of robot-detections
        """

        if msg.robots:
            # todo this
            if self.closest_distance_match:  # Select robot closest to previous prediction
                robot_msg = self._get_closest_robot_to_previous_prediction(msg)
            else:
                robot_msg = sorted(msg.robots, key=lambda robot: robot.confidence.confidence)[-1]
            position = self._get_transform(msg.header, robot_msg.bb.center.position)
            if position is not None:
                self.robot = RobotWrapper(position, msg.header, robot_msg.confidence.confidence)

    def _get_closest_robot_to_previous_prediction(self, robot_array: RobotArray) -> Union[Robot, None]:
        closest_distance = math.inf
        closest_robot_msg = robot_array.robots[0]
        for robot_msg in robot_array.robots:
            robot_transform = self._get_transform(robot_array.header, robot_msg.bb.center.position)
            if robot_transform and self.robot:
                distance = math.dist(
                    (robot_transform.point.x, robot_transform.point.y),
                    (self.robot.get_position().point.x, self.robot.get_position().point.y))
                if distance < closest_distance:
                    closest_robot_msg = robot_msg
        return closest_robot_msg

    def _get_transform(self,
                       header: Header,
                       point: Point,
                       frame: Union[None, str] = None,
                       timeout: float = 0.3) -> Union[PointStamped, None]:
        # todo
        if frame is None:
            frame = self.filter_frame

        point_stamped = PointStamped()
        point_stamped.header = header
        point_stamped.point = point
        try:
            return self.tf_buffer.transform(point_stamped, frame, timeout=rclpy.duration.Duration(seconds=timeout))
        except (tf2.ConnectivityException, tf2.LookupException, tf2.ExtrapolationException) as e:
            self.logger.warning(str(e))

    def filter_step(self) -> None:
    #todo (because we only do it whne we have a measurement anyway) why is the original filter step done with a timer and not every time you get a new measurement?
    #todo explain that in paper
        if self.robot:  # Robot measurement exists
            # Reset filter, if distance between last prediction and latest measurement is too large
            distance_to_robot = math.dist(
                (self.kf.get_update()[0][0], self.kf.get_update()[0][1]), self.get_robot_measurement())
            if self.filter_initialized and distance_to_robot > self.filter_reset_distance:
                self.filter_initialized = False
                self.logger.info(
                    f"Reset filter! Reason: Distance to robot {distance_to_robot} > {self.filter_reset_distance} (filter_reset_distance)")
            # Initialize filter if not already
            if not self.filter_initialized:
                self.init_filter(*self.get_robot_measurement())
            # Predict and publish
            self.kf.predict()
            self.kf.update(self.get_robot_measurement())
            self.publish_data(*self.kf.get_update())
            self.last_robot_stamp = self.robot.get_header().stamp
            self.robot = None  # Clear handled measurement
        else:  # No new robot measurement to handle
            if self.filter_initialized:
                # Reset filer,if last measurement is too old
                age = self.get_clock().now() - rclpy.time.Time.from_msg(self.last_robot_stamp)
                if not self.last_robot_stamp or age > self.filter_reset_duration:
                    self.filter_initialized = False
                    self.logger.info(
                        f"Reset filter! Reason: Latest robot is too old {age} > {self.filter_reset_duration} (filter_reset_duration)")
                    return
                # Empty update, as no new measurement available (and not too old)
                self.kf.predict()
                self.kf.update(None)
                self.publish_data(*self.kf.get_update())
            else:  # Publish old state with huge covariance
                state_vec, cov_mat = self.kf.get_update()
                huge_cov_mat = np.eye(cov_mat.shape[0]) * 10
                self.publish_data(state_vec, huge_cov_mat)


    def get_robot_measurement(self) -> Tuple[float, float]:
        """extracts filter measurement from robot message"""
        return self.robot.get_position().point.x, self.robot.get_position().point.y

    def init_filter(self, x: float, y: float) -> None:
        #todo look up what these things mean and put them in the text

        # how do we deal with the multiple filters
        self.kf.x = np.array([x, y, 0, 0]) # initial position of robot + velocity in x and y direction???

        # transition matrix?
        self.kf.F = np.array([[1.0, 0.0, 1.0, 0.0],
                              [0.0, 1.0, 0.0, 1.0],
                              [0.0, 0.0, self.velocity_factor, 0.0],
                              [0.0, 0.0, 0.0, self.velocity_factor]
                              ])

        # measurement function
        self.kf.H = np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0]
                              ])

        # multiplying by the initial uncertainty
        self.kf.P = np.eye(4) * 1000

        # assigning measurement noise
        self.kf.R = np.array([[1, 0],
                              [0, 1]]) * self.measurement_certainty

        # assigning process noise todo what does this mean
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=self.filter_time_step, var=self.process_noise_variance,
                                           block_size=2, order_by_dim=False)

        self.filter_initialized = True
        pass

    def publish_data(self, state_vec: np.array, cov_mat: np.array) -> None:
        # todo look at how its done in the objectsim
        header = Header()
        header.frame_id = self.filter_frame
        header.stamp = rclpy.time.Time.to_msg(self.get_clock().now())

        # position
        point_msg = Point()
        point_msg.x = float(state_vec[0])
        point_msg.y = float(state_vec[1])

        pos_covariance = np.eye(6).reshape((36))
        pos_covariance[0] = float(cov_mat[0][0])
        pos_covariance[1] = float(cov_mat[0][1])
        pos_covariance[6] = float(cov_mat[1][0])
        pos_covariance[7] = float(cov_mat[1][1])

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = header
        pose_msg.pose.pose.position = point_msg
        pose_msg.pose.covariance = pos_covariance
        pose_msg.pose.pose.orientation.w = 1.0
        self.robot_position_publisher.publish(pose_msg)

        # velocity todo done except var names
        movement_msg = TwistWithCovarianceStamped()
        movement_msg.header = header
        movement_msg.twist.twist.linear.x = float(state_vec[2] * self.filter_rate)
        movement_msg.twist.twist.linear.y = float(state_vec[3] * self.filter_rate)
        movement_msg.twist.covariance = np.eye(6).reshape((36))
        movement_msg.twist.covariance[0] = float(cov_mat[2][2])
        movement_msg.twist.covariance[1] = float(cov_mat[2][3])
        movement_msg.twist.covariance[6] = float(cov_mat[3][2])
        movement_msg.twist.covariance[7] = float(cov_mat[3][3])
        self.robot_movement_publisher.publish(movement_msg)

        # robot
        robot_msg = PoseWithCertaintyStamped()
        robot_msg.header = header
        robot_msg.pose.pose.pose.position = point_msg
        robot_msg.pose.pose.covariance = pos_covariance
        robot_msg.pose.confidence = self.robot.get_confidence() if self.robot else 0.0
        self.robot_publisher.publish(robot_msg)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()
