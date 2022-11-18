#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from soccer_vision_3d_msgs.msg import RobotArray, Robot
from copy import deepcopy

class ObjectFilter(Node):
    def __init__(self) -> None:
        pass #todo

    def _dynamic_reconfigure_callback(self, params) -> SetParametersResult:
        # construct config from the params:
        config = deepcopy(self._config)
        for param in params:
            config[param.name] = param.value

        # create Kalman filter:
        #todo potentially put this in a different method

        # adapt velocity factor to frequency#todo whats this OWW
        self.velocity_factor = (1 - config['velocity_reduction']) ** (1 / self.filter_rate)
        self.process_noise_variance = config['process_noise_variance']

        # setup publishers and subscribers:

        #todo what other attributes for the robot would need to be published:

        # publishes positions of robots:
        self.robot_pose_publisher = self.create_publisher(
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
        # publishes robots:
        self.ball_publisher = self.create_publisher(
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
        #todo what is this?
        self.reset_service = self.create_service(
            Trigger,
            config['ball_filter_reset_service_name'],
            self.reset_filter_cb
        )

        #todo why:
        self.config = config
        self.filter_timer = self.create_timer(self.filter_time_step, self.filter_step)
        return SetParametersResult(successful=True)

    def reset_filter_cb(self, req, response) -> Tuple[bool, str]:
        pass
        #todo make it so it resets only 1 filter? or all of them?

    def robot_callback(self, robot_msg: RobotArray) -> None:
        """
        Assigns each detected robot to existing or new filter
        #todo the original just decides which ball to take since it doesnt need to assign filters
        #todo so do I already assign filters here or later?

        :param robot_msg: List of robot-detections
        """
        if robot_msg.objects:
            pass

    def _get_closest_ball_to_previous_prediction(self, ball_array: BallArray) -> Union[Ball, None]:
        pass
        #todo do we need this?

    def _get_transform(self,
                       header: Header,
                       point: Point,
                       frame: Union[None, str] = None,
                       timeout: float = 0.3) -> Union[PointStamped, None]:
        # todo
        pass

    def filter_step(self) -> None:
        pass
    #todo (because we only do it whne we have a measurement anyway) why is the original filter step done with a timer and not every time you get a new measurement?
    #todo explain that in paper
        if self.robot:  # Robot measurement exists
            pass


    def get_ball_measurement(self) -> Tuple[float, float]:
        pass

    def init_filter(self, x: float, y: float) -> None:
        # how do we deal with the multiple filters
        pass

    def publish_data(self, state_vec: np.array, cov_mat: np.array) -> None:
        pass

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

