# Face Clustering

## Get Started

### Download Codes and Sample Video

```
git clone https://github.com/wooks527/face_clustering.git
cd face_clustering
```

```
youtube-dl https://www.youtube.com/watch?v=bUQj7Ng7PCs
mv "I Ran - Pool Party Scene [La La Land _ 2016] - Movie Clip HD-bUQj7Ng7PCs.mkv" sample_video.mkv
```

### Face Clustering

```
python face_clustering.py --src-path sample_video.mkv \
                          --cps 1 \
                          --out-dir results/HOG-dlib-DBSCAN
```
```
Video Information:
- Resolution: 1920.0x1080.0, FPS: 29.0, CPS: 1

Detect Faces...
100%|██████████████████████████████████████████████████████████████████▋| 231/232 [05:10<00:01,  1.34s/it]

Start Encoding...
100%|████████████████████████████████████████████████████████████████████| 68/68 [00:04<00:00, 14.54it/s]

Start Clustering...
100%|████████████████████████████████████████████████████████████████████| 5/5 [00:02<00:00,  2.48it/s]
```


## Results

### Sample Test ([Video Link](https://www.youtube.com/watch?v=bUQj7Ng7PCs))

- [Comparison All Methods](visualize_results.ipynb)

- [HOG-dlib-DBSCAN](results/HOG-dlib-DBSCAN/visualize_results.ipynb)

  - Face Detection: [face_recognition.face_locations](https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.face_locations) -> Hog + Linear SVM
  - Face Recognition: [face_recognition.face_encodings](https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.face_encodings) -> dlib.face_recognition_model_v1
  - Face Clustering: [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

- [harr-dlib-DBSCAN](results/harr-dlib-DBSCAN/visualize_results.ipynb)

  - Face Detection: [cv2.CascadeClassifier](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) -> Harr Cascade Classifier
  - Face Recognition: [face_recognition.face_encodings](https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.face_encodings) -> dlib.face_recognition_model_v1
  - Face Clustering: [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)