# Face Clustering

## Get Started

(WIP)

## Results

### Sample Test ([Video Link](https://www.youtube.com/watch?v=bUQj7Ng7PCs))

- [HOG-dlib-DBSCAN](results/HOG-dlib-DBSCAN/visualize_results.ipynb)

  - Face Detection: [face_recognition.face_locations](https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.face_locations) -> Hog + Linear SVM
  - Face Recognition: [face_recognition.face_encodings](https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.face_encodings) -> dlib.face_recognition_model_v1
  - Face Clustering: [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)