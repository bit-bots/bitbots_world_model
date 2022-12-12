import glob
import os

from setuptools import find_packages
from setuptools import setup

package_name = 'bitbots_ball_filter'


setup(
    name=package_name,
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + "/config",
            glob.glob('config/*.yaml')),
        ('share/' + package_name + '/launch',
            glob.glob('launch/*.launch')),
        *[('share/' + package_name + '/' + os.path.dirname(file), [file]) for file in glob.glob('models/**/**/*.*', recursive=True)],
    ],
    install_requires=[
        'launch',
        'setuptools',
    ],
    zip_safe=True,
    keywords=['ROS'],
    license='MIT',
    entry_points={
        'console_scripts': [
            'ball_filter = bitbots_ball_filter.ball_filter:main',
            'robot_filter = bitbots_ball_filter.robot_filter:main',
            'ball_sim = bitbots_ball_filter.ball_sim:main',
        ],
    }
)

#https://roboticsbackend.com/ros2-rclpy-parameter-callback/

