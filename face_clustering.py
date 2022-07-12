"""Face clustering.

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""

import argparse
import os

from models.face_detector import FaceDetector
from models.face_encoder import FaceEncoder
from models.face_cluster import FaceCluster


def get_args() -> argparse.Namespace:
    """Parse arguments."""
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
        "--detector",
        type=str,
        default="hog",
        help="Face Detector (e.g. hog, cnn, harr)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dlib",
        help="Face Encoder (e.g. dlib, arcface)",
    )
    parser.add_argument(
        "--cps",
        type=int,
        default=1,
        help="capture per second",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device for model",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/sample-test",
        help="directory of results",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="log level",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Detect faces
    fd = FaceDetector(model=args.detector)
    if not os.path.isfile(f"{args.out_dir}/face_bboxes.pickle"):  # Detect faces
        frame_ids, face_bboxes = fd.detect(
            src_path=args.src_path,
            out_dir=args.out_dir,
            cps=args.cps,
            device=args.device,
            verbose=args.verbose,
        )
        fd.save_face_bboxes(args.out_dir, "face_bboxes.pickle")
    else:
        frame_ids, face_bboxes = fd.load_face_bboxes(args.out_dir, "face_bboxes.pickle")

    # Encode faces
    fe = FaceEncoder(model=args.encoder)
    if not os.path.isfile(f"{args.out_dir}/faces.pickle"):
        faces = fe.encode(frame_ids, face_bboxes, args.out_dir)
        fe.faces = faces
        fe.save_faces(args.out_dir, "faces.pickle")
    else:
        faces = fe.load_faces(args.out_dir, "faces.pickle")

    # Cluster faces
    fc = FaceCluster(faces)
    fc.cluster(args.src_path, args.out_dir)
