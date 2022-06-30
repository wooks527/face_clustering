"""Face classes.

- Author: Hyunwook Kim
- Contact: hyunwook@rippleai.co
"""

from typing import List, Tuple
from sklearn.cluster import DBSCAN
from tqdm import tqdm

import cv2
import face_recognition as fr
import argparse
import numpy as np
import os
import shutil
import pickle
import json


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--src-path",
        type=str,
        default="sample_video.mkv",
        help="source path",
    )
    parser.add_argument(
        "--cps",
        type=int,
        default=1,
        help="capture per second",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/sample-test",
        help="directory of results",
    )
    return parser.parse_args()


class Face():
    """Face class with frame, bounding box, encoding information."""

    def __init__(self, frame_id: int, bbox: List[Tuple], encoding) -> None:
        """Initialize instance.

        Args:
            frame_id: frame index
            bbox: bounding box coordinates
            encoding: face encoding vector
        """
        self.frame_id = frame_id
        self.bbox = bbox
        self.encoding = encoding


class FaceClustering():
    """Cluster faces using encoder and cluster methods."""

    def __init__(self) -> None:
        """Initialize instance.

        Args:
            faces: list of face instances
        """
        self.faces = []

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(pickle.dumps(self.faces))

    def load(self, filename):
        with open(filename, "rb") as f:
            data = f.read()
            self.faces = pickle.loads(data)

    def encode(self, src_path: str, cps: int, out_dir: str) -> None:
        """Encode frames to encoding vectors.

        Args:
            src_path: source path (e.g. video file path)
            cps: capture per second
            out_dir: directory of results
        """
        if os.path.isdir(f"{out_dir}/captured_imgs"):
            shutil.rmtree(f"{out_dir}/captured_imgs")
        os.mkdir(f"{out_dir}/captured_imgs")

        src = cv2.VideoCapture(src_path)
        assert src.isOpened(), "Fail to load."

        self.faces = []
        width, height, fps = src.get(3), src.get(4), src.get(5)
        skip_num = int(fps/cps)
        print("Video Information:")
        print(f"- Resolution: {width}x{height}, FPS: {fps}, CPS: {cps}")

        print("\nStart Encoding...")
        frame_id = 0
        while True:
            _, frame = src.read()
            if frame is None:
                break

            frame_id += 1
            if frame_id % skip_num != 0:
                continue

            frame_rgb = frame[:, :, ::-1]
            bboxes = fr.face_locations(frame_rgb, model="hog")
            print(f"frame ({frame_id}): {bboxes}")
            if not bboxes:
                continue

            encodings = fr.face_encodings(frame_rgb, bboxes)
            new_faces = [Face(frame_id, b, e) for b, e in zip(bboxes, encodings)]
            img_path = f"{out_dir}/captured_imgs/frame-{frame_id}.jpg"
            cv2.imwrite(img_path, frame)
            self.faces.extend(new_faces)

        src.release()

    def get_encodings(self) -> List:
        return [face.encoding for face in self.faces]

    def get_face_img_with_bbox(self, img: np.array, bbox: List[Tuple]) -> np.array:
        img_h, img_w = img.shape[:2]
        (top, right, bottom, left) = bbox
        bbox_w, bbox_h = right - left, bottom - top
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        top = max(top - bbox_h, 0)
        bottom = min(bottom + bbox_h, img_h - 1)
        left = max(left - bbox_w, 0)
        right = min(right + bbox_w, img_w - 1)
        return img[top:bottom, left:right]

    def cluster(self, src_path: str, out_dir: str) -> None:
        """Cluster face encodings.
        
        Args:
            src_path: source path (e.g. video file path)
            out_dir: directory of results
        """
        assert len(self.faces) != 0, "There is no cluster."
        if os.path.isdir(f"{out_dir}/clustered_imgs"):
            shutil.rmtree(f"{out_dir}/clustered_imgs")
        os.mkdir(f"{out_dir}/clustered_imgs")

        src = cv2.VideoCapture(src_path)
        assert src.isOpened(), "Fail to load."
        fps = int(src.get(5))
        src.release()

        print("Start Clustering...")
        cluster = DBSCAN(metric="euclidean")
        cluster.fit(self.get_encodings())

        cluster_ids = np.unique(cluster.labels_).tolist()
        cluster_info = {cluster_id: [] for cluster_id in cluster_ids}
        src = cv2.VideoCapture(src_path)
        for cluster_id in tqdm(cluster_ids):
            filtered_idxes = np.where(cluster.labels_ == cluster_id)[0]
            prev_frame_id = self.faces[filtered_idxes[0]].frame_id - fps
            range_idxes = []
            for cur_idx in filtered_idxes:
                cur_frame_id = int(self.faces[cur_idx].frame_id)
                if cur_frame_id - prev_frame_id == fps:
                    range_idxes.append(cur_frame_id)
                else:
                    cluster_info[cluster_id].append(range_idxes)
                    range_idxes = [cur_frame_id]

                img_path = f"{out_dir}/captured_imgs/frame-{cur_frame_id}.jpg"
                img = cv2.imread(img_path)
                bbox = self.faces[cur_idx].bbox
                face_img = self.get_face_img_with_bbox(img, bbox)
                face_img_fname = f"fid-{cluster_id}-{len(cluster_info[cluster_id])}-{cur_frame_id}.jpg"
                face_img_path = f"{out_dir}/clustered_imgs/{face_img_fname}"
                cv2.imwrite(face_img_path, face_img)

                prev_frame_id = cur_frame_id

        with open(f"{out_dir}/clustered_results.json", "w") as f:
            json.dump(cluster_info, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    fc = FaceClustering()

    # Encode faces
    if not os.path.isfile(f"{args.out_dir}/encodings.pickle"):
        fc.encode(args.src_path, args.cps, args.out_dir)
        fc.save(f"{args.out_dir}/encodings.pickle")
    else:
        fc.load(f"{args.out_dir}/encodings.pickle")

    # Cluster faces
    fc.cluster(args.src_path, args.out_dir)
