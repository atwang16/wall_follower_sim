#!/usr/bin/env python2

import numpy as np
from sklearn import linear_model
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from collections import deque

class PIDController:
    def __init__(self, k_p, k_i, k_d):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.error_accumulation = 0
        self.last_error = None

    def __call__(self, error):
        self.error_accumulation += error
        delta_error = 0 if self.last_error is None else (error - self.last_error)
        proportional = self.k_p * error
        integral = self.k_i * self.error_accumulation
        derivative = self.k_d * delta_error
        self.last_error = error
        return proportional + integral + derivative


class WallFollower:
    # Import ROS parameters from the "params.yaml" file.
    # Access these variables in class functions with self:
    # i.e. self.CONSTANT
    SCAN_TOPIC = rospy.get_param("wall_follower/scan_topic")
    DRIVE_TOPIC = rospy.get_param("wall_follower/drive_topic")
    SIDE = rospy.get_param("wall_follower/side")
    VELOCITY = rospy.get_param("wall_follower/velocity")
    DESIRED_DISTANCE = rospy.get_param("wall_follower/desired_distance")
    MAX_HEADING = 0.34

    def __init__(self):
        self.publisher = rospy.Publisher(WallFollower.DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)
        rospy.Subscriber(WallFollower.SCAN_TOPIC, LaserScan, self.callback)
        self.marker_publisher = rospy.Publisher("visualization_marker", Marker, queue_size=10)

        # tunable constants
        distance_k_p = rospy.get_param("wall_follower/distance_k_p", 2.0)
        distance_k_i = rospy.get_param("wall_follower/distance_k_i", 0.0)
        distance_k_d = rospy.get_param("wall_follower/distance_k_d", 3.0)
        self.front_dist_threshold_factor = 2
        self.distance_pid = PIDController(distance_k_p, distance_k_i, distance_k_d)
        self.extra_angle = np.pi/4
        self.ransac_residual_threshold = None 
        self.line_length = 2

    def callback(self, laser_scan):
        (estimated_distance, estimated_angle), front_dist = self.estimate_state(laser_scan)
        desired_heading = self.compute_heading(estimated_distance, estimated_angle, front_dist)
        out_msg = self.generate_message(desired_heading)
        self.publisher.publish(out_msg)

    def extract_laser_data(self, scan):
        # parse LaserScan ranges
        ranges = scan.ranges
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
        laser_data = np.stack([ranges, angles])
        valid_ranges = (laser_data[0, :] >= scan.range_min) & (laser_data[0, :] <= scan.range_max)
        valid_indices = np.where(valid_ranges)[0]
        laser_data = laser_data[:, valid_indices]
        valid_angles = ((laser_data[1, :]) * self.SIDE + self.extra_angle) >= 0  # positive angles if self.SIDE > 0 else negative angles
        side_data = laser_data[:, np.where(valid_angles)[0]]
        front_dist = np.mean(laser_data[0, np.where(np.abs(laser_data[1, :]) < scan.angle_increment * 4)[0]])
        return side_data, front_dist

    def estimate_state(self, scan):
        # parse LaserScan ranges
        laser_data, front_dist = self.extract_laser_data(scan)
        side_wall = self.ransac(laser_data, marker_id=0)

        return (side_wall, front_dist)

    def min_distance(self, laser_data):
        min_dist_idx = np.argmin(laser_data[0, :])
        min_dist, angle = laser_data[:, min_dist_idx]
        return min_dist, angle - self.SIDE * np.pi / 2.0

    def least_squares(self, laser_data):
        x = laser_data[0, :] * np.cos(laser_data[1, :])
        y = laser_data[0, :] * np.sin(laser_data[1, :])
        A = np.stack([x, np.ones(x.shape)], axis=1)
        slope, y_intercept = np.linalg.lstsq(A, y, rcond=1e-10)[0]

        self.generate_marker(slope, y_intercept)
        angle = np.arctan(slope)
        distance = np.abs(y_intercept) / np.sqrt(1 + slope ** 2)
        return distance, angle

    def ransac(self, laser_data, marker_id=0):
        x = laser_data[0, :] * np.cos(laser_data[1, :])
        x = x.reshape(-1, 1)
        y = laser_data[0, :] * np.sin(laser_data[1, :])
        ransac = linear_model.RANSACRegressor(residual_threshold=self.ransac_residual_threshold)
        ransac.fit(x, y)
        y_intercept, m_b = ransac.predict(np.array([[0], [1]]))
        slope = m_b - y_intercept

        self.generate_marker(slope, y_intercept, marker_id=marker_id)
        angle = np.arctan(slope)
        distance = np.abs(y_intercept) / np.sqrt(1 + slope ** 2)
        return distance, angle

    def compute_heading(self, estimated_distance, estimated_wall_angle, front_dist):
        if front_dist < self.front_dist_threshold_factor * self.DESIRED_DISTANCE:
            return -self.SIDE * self.MAX_HEADING 
        distance_error = self.DESIRED_DISTANCE - estimated_distance
        desired_angle = -self.SIDE * self.distance_pid(distance_error)
        desired_heading = desired_angle - (-1 * estimated_wall_angle)
        return desired_heading

    def generate_marker(self, m, b, marker_id=0):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "laser"
        marker.ns = "wall_estimation"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 0.5
        marker.scale.x = 0.1
        
        # color is blue
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        perp_point = b * np.array([-1.0 / (m + 1/m), 1.0 / (m ** 2 + 1)])
        endpoints_x = np.roots([1 + m ** 2, 
                                2 * m * (b - perp_point[1]) - 2 * perp_point[0],
                                perp_point[0] ** 2 + (b - perp_point[1]) ** 2 - (self.line_length / 2) ** 2])
        endpoints_y = m * endpoints_x + b
        start = Point()
        start.x = endpoints_x[0]
        start.y = endpoints_y[0]
        end = Point()
        end.x = endpoints_x[1]
        end.y = endpoints_y[1]
        marker.points.append(start)
        marker.points.append(end) 
        self.marker_publisher.publish(marker)

    def generate_message(self, desired_heading):
        out_msg = AckermannDriveStamped()
        out_msg.drive.steering_angle = np.clip(desired_heading, a_min=-self.MAX_HEADING, a_max=self.MAX_HEADING)
        out_msg.drive.steering_angle_velocity = 0
        out_msg.drive.speed = self.VELOCITY
        out_msg.drive.acceleration = 0
        out_msg.drive.jerk = 0

        return out_msg

if __name__ == "__main__":
    rospy.init_node('wall_follower')
    wall_follower = WallFollower()
    rospy.spin()

