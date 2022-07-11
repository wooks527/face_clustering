"""Face detector

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""

from typing import List, Tuple

import os
import shutil
import cv2
import pickle
import face_recognition as fr

from tqdm import tqdm
from models.retinaface import RetinaFace
from models.yolov5_face import YOLOv5Face


class FaceDetector:
    """Detect faces."""

    def __init__(self, model: str = "hog"):
        """Initialize instances.

        Args:
            model: Face Detector (e.g. hog, cnn, harr)
        """
        self.model = model
        self.frame_ids = []
        self.face_bboxes = []

    def load_face_bboxes(self, out_dir: str, filename: str) -> Tuple[List, List]:
        """Load face bounding boxes.

        Args:
            out_dir: directory of results
            filename: filename of face_bboxes instances

        Returns:
            frame_ids: extracted frame IDs
            face_bboxes: extracted face bounding boxes
        """
        print("Load Face Bboxes...")
        with open(f"{out_dir}/{filename}", "rb") as f:
            data = f.read()
            frame_ids, face_bboxes = pickle.loads(data)
        return frame_ids, face_bboxes

    def save_face_bboxes(self, out_dir: str, filename: str) -> None:
        """Load face bounding boxes.

        Args:
            out_dir: directory of results
            filename: filename of face_bboxes instances
        """
        with open(f"{out_dir}/{filename}", "wb") as f:
            f.write(pickle.dumps((self.frame_ids, self.face_bboxes)))

    def detect(
        self, src_path: str, out_dir: str, cps: int, device: str
    ) -> Tuple[List, List]:
        """Detect faces.

        Args:
            src_path: source path (e.g. video file path)
            cps: capture per second
            out_dir: directory of results
            device: device for model

        Returns:
            frame_ids: extracted frame IDs
            face_bboxes: extracted face bounding boxes
        """
        if os.path.isdir(f"{out_dir}/captured_imgs"):
            shutil.rmtree(f"{out_dir}/captured_imgs")
        os.makedirs(f"{out_dir}/captured_imgs")

        src = cv2.VideoCapture(src_path)
        assert src.isOpened(), "Fail to load."

        width, height, fps = src.get(3), src.get(4), src.get(5)
        skip_num = int(fps / cps)
        frame_num = int(src.get(cv2.CAP_PROP_FRAME_COUNT) / skip_num)
        print("Video Information:")
        print(f"- Resolution: {int(width)}x{int(height)}, FPS: {fps}, CPS: {cps}")

        if self.model == "harr":
            model = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
        elif self.model == "retinaface":
            use_cpu = True if device == "cpu" else False
            model = RetinaFace(
                network="mobile0.25",
                trained_model="./Pytorch_Retinaface/weights/mobilenet0.25_Final.pth",
                cpu=use_cpu,
            )
        elif self.model == "yolov5_face":
            model = YOLOv5Face(
                weight="./yolov5_face/weights/yolov5n-face.pt",
                # weight="./yolov5_face/weights/yolov5n-0.5.pt",
                device="cpu",
            )
        else:
            assert False, "Not supported model."
        print("\nLoad Model...")

        print("\nDetect Faces...")
        frame_ids, face_bboxes = [], []
        frame_id = 0
        pbar = tqdm(total=frame_num)
        while True:
            _, frame = src.read()
            if frame is None:
                break

            frame_id += 1
            if frame_id % skip_num != 0:
                continue

            frame_rgb = frame[:, :, ::-1]
            if self.model in ("hog", "cnn"):  # dlib face detectors
                face_locations = fr.face_locations(frame_rgb, model=self.model)
            elif self.model == "harr":  # Harr cascade classifier
                face_locations = model.detectMultiScale(
                    frame_rgb, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
                )
                if len(face_locations) != 0:
                    # (top, right, bottom, left) = bbox
                    face_locations = face_locations.tolist()
                    for i, face_location in enumerate(face_locations):
                        left, top = face_location[:2]
                        right, bottom = left + face_location[2], top + face_location[3]
                        face_locations[i] = (top, right, bottom, left)
            elif self.model == "retinaface":
                face_locations = model.detect(img_raw=frame)
            elif self.model == "yolov5_face":
                face_locations = model.detect(image=frame)
            else:
                assert False, "Not supported model."

            # print(f"frame ({frame_id}): {face_locations}")
            if not face_locations:
                pbar.update(1)
                continue

            frame_ids.append(frame_id)
            face_bboxes.append(face_locations)

            img_path = f"{out_dir}/captured_imgs/frame-{frame_id}.jpg"
            cv2.imwrite(img_path, frame)

            pbar.update(1)

        src.release()
        self.frame_ids = frame_ids
        self.face_bboxes = face_bboxes
        return frame_ids, face_bboxes
