# rgb_detector.py
import rclpy
from rclpy.node import Node
from rclpy import qos

import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, PolygonStamped, Point32
from cv_bridge import CvBridge
from std_msgs.msg import Int32

class DetectorRGB(Node):
    visualisation = True
    data_logging = True
    log_path = 'evaluation/data/'
    seq = 0

    def __init__(self):    
        super().__init__('detector_rgb')
        self.bridge = CvBridge()

        self.min_area_size = 100.0
        self.countour_colour = (255, 255, 0)  # Cyan
        self.countour_width = 1  # Pixels

        self.object_pub = self.create_publisher(PolygonStamped, '/object_polygon', 10)
        self.object_count_pub = self.create_publisher(Int32, '/object_count', 10)
        self.image_sub = self.create_subscription(Image, '/limo/depth_camera_link/image_raw', 
                                                  self.image_colour_rgb, qos_profile=qos.qos_profile_sensor_data
        )

    def image_colour_rgb(self, data):
        bgr_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # RGB colour range for red
        lower_red = (0, 0, 100)
        upper_red = (50, 50, 255)
        red_mask = cv.inRange(bgr_image, lower_red, upper_red)

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
                    cv.rectangle(bgr_image, (bbx, bby), (bbx + bbw, bby + bbh), self.countour_colour, self.countour_width)

        print(f"Frame {self.seq}: Detected {object_count} objects.")
        self.object_count_pub.publish(Int32(data=object_count))

        if self.visualisation:
            cv.imshow("RGB - Colour Image", bgr_image)
            cv.imshow("RGB - Detection Mask", mask)
            cv.waitKey(1)

        self.seq += 1


def main(args=None):
    rclpy.init(args=args)
    detector_rgb = DetectorRGB()
    rclpy.spin(detector_rgb)
    detector_rgb.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
