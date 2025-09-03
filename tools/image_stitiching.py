import cv2
import numpy as np

image1 = cv2.imread("tiles/tile_0001_x409_y0.jpg", 
                    #cv2.IMREAD_GRAYSCALE
                    )
image2 = cv2.imread("tiles/tile_0002_x818_y0.jpg", 
                    #cv2.IMREAD_GRAYSCALE
                    )


def stitch_two_images(image1, image2):
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    raw_matches = bf.match(des1, des2)

    # Sort matches by distance
    raw_matches = sorted(raw_matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in raw_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in raw_matches]).reshape(-1, 1, 2)

    # Find homography matrix and inlier mask. This finds the matrix to map Image1 to Image2's coordinate plane
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # Play with the ransacReprojThreshold value (5.0)
    # Use the mask to select only the inlier matches

    # Get the height and width of both images
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Get the canvas size for the warped image
    corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    x_min = warped_corners[:, 0, 0].min()
    y_min = warped_corners[:, 0, 1].min()
    x_max = warped_corners[:, 0, 0].max()
    y_max = warped_corners[:, 0, 1].max()
    offset_x = max(0, -x_min)
    offset_y = max(0, -y_min)


    # Calculate proper canvas size
    warped_width = int(x_max - x_min + offset_x)
    warped_height = int(y_max - y_min + offset_y)
    canvas_width = max(warped_width, w2 + int(offset_x))
    canvas_height = max(warped_height, h2 + int(offset_y))
    translation_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
    combined = translation_matrix @ H

    final_image = cv2.warpPerspective(image1, combined, (canvas_width, canvas_height))

    # Place Image 2 at the correct offset position
    end_x = min(int(offset_x) + w2, canvas_width)
    end_y = min(int(offset_y) + h2, canvas_height)
    print(end_x, end_y)
    final_image[int(offset_y):end_y, int(offset_x):end_x] = image2[:end_y-int(offset_y), :end_x-int(offset_x)]

    return final_image
