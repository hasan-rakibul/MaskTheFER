import os
import cv2
import numpy as np
import dlib


def rotate_image(image, angle, center=None, scale=1.0):
    """
        Rotate the input image by the given angle.
    """
    (h, w) = image.shape[:2]
    # If no centre of rotation is specified, the centre of the image is set as the centre of rotation
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    return rotated_image


# Load Dlib face detector and face landmark predictor.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def crop_and_save(input_path, output_path):
    """
        Crop the regions near the eyes
        and rotate the images so that the eyes are on a horizontal line.
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    # -0.7 here is the threshold (ranging from -1 to 1 ) of the face detector.
    faces, scores, idx = detector.run(img, 1, -0.7)
    # This if statement will remove some poorly masked images in the masked datasets (mostly in FER2013),
    # which cannot be detected by Dlib even using a lenient threshold (-0.7).
    if len(faces) == 1:

        # get the face landmarks
        landmarks = predictor(img, faces[0])

        # get the positions of eyes
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        # calculate the angle between the two eyes
        angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
        rotated_img = rotate_image(img, angle)

        # The corresponding region of interest(ROI) and landmarks in the rotated images can also be calculated
        # mathematically by the angle. In this implementation, I just reuse the Dlib library to detect them again in
        # the rotated images for convenience.
        faces, scores, idx = detector.run(img, 1, -0.7)
        if len(faces) == 1:
            # get the face landmarks
            landmarks = predictor(rotated_img, faces[0])
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            nose = (landmarks.part(30).x, landmarks.part(30).y)
            # calculate the bounding box for cropping.
            top = max(0, faces[0].top())
            bottom = max(left_eye[1] + 8, right_eye[1] + 8, nose[1])
            left = max(0, faces[0].left(), left_eye[0] - 6)
            right = min(faces[0].left() + faces[0].width(), img.shape[1] - 1, right_eye[0] + 7)

            cropped_img = rotated_img[top: bottom, left:right]
            try:
                cropped_img = cv2.resize(cropped_img, (48, 24))
                cv2.imwrite(output_path, cropped_img)
            except:
                pass


input_folder = "..\datasets\M-FER2013"
output_folder = "..\datasets\M-FER2013_cropped"
# input_folder = "..\datasets\M-CK+"
# output_folder = "..\datasets\M-CK+_cropped"
input_folder = "..\datasets\FER2013_masked"
output_folder = "..\datasets\M-FER2013_masked_cropped"
# Crop the masked datasets to preprocess images.
for sub_folder in os.listdir(input_folder):
    sub_folder_path = os.path.join(input_folder, sub_folder)
    if os.path.isdir(sub_folder_path):
        output_folder_path = os.path.join(output_folder, sub_folder)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        for filename in os.listdir(sub_folder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                input_path = os.path.join(sub_folder_path, filename)
                output_path = os.path.join(output_folder_path, filename)
                crop_and_save(input_path, output_path)
