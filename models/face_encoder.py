"""Face encoder.

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""

from typing import List

import cv2
import pickle
import numpy as np
import face_recognition as fr

from tqdm import tqdm
from deepface import DeepFace
from utils.face import Face


class FaceEncoder:
    """Encode faces as embedding vectors."""

    def __init__(self, model: str) -> None:
        """Initialize instances.

        Args:
            model: Face encoding model
        """
        self.model = model
        if self.model == "dlib":
            self.shape = None
        elif self.model == "arcface":
            self.input_shape = (112, 112)
        elif self.model == "facenet":
            self.input_shape = (160, 160)
        else:
            assert False, "Not supported model."
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
        if self.model == "dlib":
            pass
        elif self.model == "arcface":
            model = DeepFace.build_model("ArcFace")
        elif self.model == "facenet":
            model = DeepFace.build_model("Facenet")
        else:
            assert False, "Not supported model."

        faces = []
        for frame_id, face_bboxes_per_img in tqdm(
            zip(frame_ids, face_bboxes), total=len(frame_ids)
        ):
            img_path = f"{out_dir}/captured_imgs/frame-{frame_id}.jpg"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.model == "dlib":
                embeddings = fr.face_encodings(img, face_bboxes_per_img)
            elif self.model in ("arcface", "facenet"):
                embeddings = []
                for face_bbox in face_bboxes_per_img:
                    face_bbox = [max(0, coord) for coord in face_bbox]
                    top, right, bottom, left = face_bbox
                    face_img = img[top:bottom, left:right]
                    face_img = cv2.resize(face_img, self.input_shape)
                    face_img = np.expand_dims(face_img, axis=0)
                    embedding = model.predict(face_img, verbose=0)[0]
                    embeddings.append(embedding)

            assert len(face_bboxes_per_img) == len(embeddings), "Encoding error!!"
            for face_bbox, embedding in zip(face_bboxes_per_img, embeddings):
                faces.append(Face(frame_id, face_bbox, embedding))

        self.faces = faces
        return faces
