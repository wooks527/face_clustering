"""Face encoder.

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""

from typing import List

import cv2
import pickle
import face_recognition as fr

from tqdm import tqdm
from utils.face import Face


class FaceEncoder:
    """Encode faces as embedding vectors."""

    def __init__(self) -> None:
        """Initialize instances."""
        self.faces = []

    def save_faces(self, out_dir: str, filename: str) -> None:
        """Save face embedding vectors.

        Args:
            out_dir: directory of results
            filename: output filename of face embedding vectors
        """
        with open(f"{out_dir}/{filename}", "wb") as f:
            f.write(pickle.dumps(self.faces))

    def load_faces(self, out_dir: str, filename: str) -> None:
        """Save face embedding vectors.

        Args:
            out_dir: directory of results
            filename: filename of face embedding vectors

        Returns:
            self.faces: list of face instances
        """
        print("\nLoad Embeddings...")
        with open(f"{out_dir}/{filename}", "rb") as f:
            data = f.read()
            self.faces = pickle.loads(data)

        return self.faces

    def encode(self, frame_ids: List, face_bboxes: List, out_dir: str) -> List[Face]:
        """Encode face bounding boxes to encoding vectors.

        Args:
            frame_ids: extracted frame IDs
            face_bboxes: extracted face bounding boxes
            out_dir: directory of results

        Returns:
            faces: list of face instances
        """
        print("\nStart Encoding...")
        faces = []
        for frame_id, face_bboxes_per_img in tqdm(
            zip(frame_ids, face_bboxes), total=len(frame_ids)
        ):
            img_path = f"{out_dir}/captured_imgs/frame-{frame_id}.jpg"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            embeddings = fr.face_encodings(img, face_bboxes_per_img)
            for face_bbox, embedding in zip(face_bboxes_per_img, embeddings):
                faces.append(Face(frame_id, face_bbox, embedding))

        self.faces = faces
        return faces
