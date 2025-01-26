# hsv_detector.py
import rclpy
from rclpy.node import Node
from rclpy import qos

import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, PolygonStamped, Point32
from cv_bridge import CvBridge
from std_msgs.msg import Int32

class DetectorHSV(Node):
    visualisation = True
    data_logging = True
    log_path = 'evaluation/data/'
    seq = 0

    def __init__(self):    
        super().__init__('detector_hsv')
        self.bridge = CvBridge()

        self.min_area_size = 100.0
        self.countour_color = (255, 255, 0)  # Cyan
        self.countour_width = 1  # Pixels

        self.object_pub = self.create_publisher(PolygonStamped, '/object_polygon', 10)
        self.object_count_pub = self.create_publisher(Int32, '/object_count', 10)
        self.image_sub = self.create_subscription(
            Image, '/limo/depth_camera_link/image_raw', 
            self.image_color_hsv, qos_profile=qos.qos_profile_sensor_data
        )

    def image_color_hsv(self, data):
        bgr_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # Convert to HSV
        hsv_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)
        
        # HSV ranges for red
        lower_red1 = (0, 100, 100)
        upper_red1 = (10, 255, 255)
        lower_red2 = (170, 100, 100)
        upper_red2 = (180, 255, 255)

        mask1 = cv.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv.bitwise_or(mask1, mask2)

        self.process_contours(data, red_mask, bgr_image)

    def process_contours(self, data, mask, bgr_image):
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        object_count = 0

        for contour in contours:
            area = cv.contourArea(contour)
            if area > self.min_area_size:
                object_count += 1
                bbx, bby, bbw, bbh = cv.boundingRect(contour)
                if self.visualisation:
                    cv.rectangle(bgr_image, (bbx, bby), (bbx + bbw, bby + bbh), self.countour_color, self.countour_width)

        print(f"Frame {self.seq}: Detected {object_count} objects.")
        self.object_count_pub.publish(Int32(data=object_count))

        if self.visualisation:
            cv.imshow("HSV - Colour Image", bgr_image)
            cv.imshow("HSV - Detection Mask", mask)
            cv.waitKey(1)

        self.seq += 1


def main(args=None):
    rclpy.init(args=args)
    detector_hsv = DetectorHSV()
    rclpy.spin(detector_hsv)
    detector_hsv.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
