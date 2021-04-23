import os
import argparse
import time

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",
	help="path to input image or video")
parser.add_argument("-p", "--prototxt", default="./models/deploy.prototxt",
	help="path to Caffe prototxt file")
parser.add_argument("-m", "--model", default="./models/res10_300x300_ssd_iter_140000.caffemodel",
	help="path to Caffe pre-trained model")
parser.add_argument("-th", "--threshold", type=float, default=0.6,
	help="face confidense threshold.")
args = parser.parse_args()

class FaceDetector:
    def __init__(self, prototxt, model, threshold):
        # Load a model imported from Tensorflow
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.threshold = threshold

    def _blob(self, image):
        # Input image
        _image = cv2.resize(image, (300, 300))
        blob = cv2.dnn.blobFromImage(_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        return blob

    def __call__(self, image):
        time_s = time.time()
        rows, cols = image.shape[:2]

        blob = self._blob(image)

        # pass the blob through the network and obtain the detections and
        # predictions
        print("[INFO] computing object detections...")
        self.net.setInput(blob)
        detections = self.net.forward()

        # Loop on the outputs
        bboxes = []
        for detection in detections[0,0]:
            score = float(detection[2])
            if score > self.threshold:
                left = max(0, int(detection[3] * cols + 0.5))
                top = max(0, int(detection[4] * rows + 0.5))
                right = min(cols-1, int(detection[5] * cols + 0.5))
                bottom = min(rows-1, int(detection[6] * rows + 0.5))
                bboxes.append((left, top, right, bottom, score))

        print(time.time()-time_s, "[sec]")
        return bboxes
        
    @classmethod
    def clip_bbox(cls, image, bbox):
        left, top, right, bottom, _ = bbox
        clip_img = image[top:bottom, left:right]
        return clip_img

    @classmethod
    def draw_bbox(cls, image, bbox):
        left, top, right, bottom, _ = bbox
        #draw a red rectangle around detected objects
        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
        return image

def execute_image(fd:FaceDetector, image, dst_path:str=None):
    bboxes = fd(image)
    for i, bbox in enumerate(bboxes):
        FaceDetector.draw_bbox(image, bbox)

    if dst_path is not None:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, image)

def execute_video(fd:FaceDetector, video_path:str, dst_path:str=None):

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, img = cap.read()
        if not ret:
            print("fin.")
            break
        
        _dst_path = None
        if dst_path is not None:
            _dst_path = os.path.join(dst_path, f"{frame_count:08d}.jpg")
        execute_image(fd, img, _dst_path)

        frame_count += 1

    cap.release()

IMAGE_EXT_LIST = [
    ".jpg", ".jpeg", ".png", ".bmp"
]
VIDEO_EXT_LIST = [
    ".mp4", ".avi"
]

if __name__ == "__main__":
    src = args.input

    fd = FaceDetector(args.prototxt, args.model, args.threshold)

    _, ext = os.path.splitext(src)
    if ext in IMAGE_EXT_LIST:
        image = cv2.imread(src)
        execute_image(fd, image, "./result/lena.jpg")
    elif ext in VIDEO_EXT_LIST:
        execute_video(fd, src, "./result")
    else:
        print("invalid input data.")
        exit()
