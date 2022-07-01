"""Face Class

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""

from typing import List, Tuple


class Face:
    """Face class with frame, bounding box, embedding information."""

    def __init__(self, frame_id: int, bbox: List[Tuple], embedding: List) -> None:
        """Initialize instance.

        Args:
            frame_id: frame index
            face_bbox: bounding box coordinates
            embedding: face embedding vector
        """
        self.frame_id = frame_id
        self.bbox = bbox
        self.embedding = embedding
