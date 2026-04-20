# Lab 9 — Face Detection

## Data files setup

Two binary files are not tracked by git and must be placed manually in `data/` before running either notebook.

### 1. Haar cascade (`haarcascade_frontalface_default.xml`)

Download from the OpenCV repository:

```
https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml
```

Click **Raw**, then save the file as `data/haarcascade_frontalface_default.xml`.

### 2. dlib 68-point shape predictor (`shape_predictor_68_face_landmarks.dat`)

Download from:

```
https://github.com/GuoQuanhao/68_points/blob/master/shape_predictor_68_face_landmarks.dat
```

Click **Raw**, then save the file as `data/shape_predictor_68_face_landmarks.dat`.

---

After both files are in place the `data/` directory should contain:

```
data/
├── haarcascade_frontalface_default.xml
├── shape_predictor_68_face_landmarks.dat
└── *.jpg   ← photo dataset
```
