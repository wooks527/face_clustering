"""Face cluster

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""

from typing import List, Tuple

import cv2
import os
import shutil
import json
import numpy as np

from sklearn.cluster import DBSCAN
from tqdm import tqdm
from utils.face import Face


class FaceCluster:
    """Cluster faces using encoder and cluster methods."""

    def __init__(self, faces: List[Face] = []) -> None:
        """Initialize instance.

        Args:
            faces: list of face instances
        """
        self.faces = faces

    def get_embeddings(self) -> List:
        """Generate list of face embdding vectors.

        Returns:
            list of face embedding vectors
        """
        return [face.embedding for face in self.faces]

    def get_face_img_with_bbox(self, img: np.array, bbox: List[Tuple]) -> np.array:
        """Generate face images with bounding boxes.

        Args:
            img: original image
            bbox: face bounding box coordinates

        Returns:
            cropped face patch with face bounding box
        """
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

        print("\nStart Clustering...")
        cluster = DBSCAN(metric="euclidean")
        cluster.fit(self.get_embeddings())

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
                face_img_fname = (
                    f"fid-{cluster_id}"
                    f"-{len(cluster_info[cluster_id])}"
                    f"-{cur_frame_id}.jpg"
                )
                face_img_path = f"{out_dir}/clustered_imgs/{face_img_fname}"
                cv2.imwrite(face_img_path, face_img)

                prev_frame_id = cur_frame_id

        out_path = f"{out_dir}/clustered_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cluster_info, f, indent=4)
