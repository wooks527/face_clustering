"""Face detector

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""

from typing import List, Tuple

import os
import shutil
import cv2
import pickle
import time
import face_recognition as fr

from tqdm import tqdm


class FaceDetector:
    """Detect faces."""

    def __init__(self):
        """Initialize instances."""
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

    def detect(self, src_path: str, out_dir: str, cps: int) -> Tuple[List, List]:
        """Detect faces.

        Args:
            src_path: source path (e.g. video file path)
            cps: capture per second
            out_dir: directory of results

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
            face_locations = fr.face_locations(frame_rgb, model="hog")
            # print(f"frame ({frame_id}): {face_locations}")
            if not face_locations:
                time.sleep(0.01)
                pbar.update(1)
                continue

            frame_ids.append(frame_id)
            face_bboxes.append(face_locations)

            img_path = f"{out_dir}/captured_imgs/frame-{frame_id}.jpg"
            cv2.imwrite(img_path, frame)

            time.sleep(0.25)
            pbar.update(1)

        src.release()
        self.frame_ids = frame_ids
        self.face_bboxes = face_bboxes
        return frame_ids, face_bboxes
