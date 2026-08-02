# FOCUS — AI-Based Student Engagement and Attention Tracking System

## Setup Instructions (any new laptop)

### 1. Prerequisites
- Python 3.10 or higher installed
- Git installed
- A working webcam

Check both are installed:
```
python --version
git --version
```

### 2. Clone the repository
```
git clone https://github.com/harrypotter1808/FOCUS.git
cd FOCUS
```

### 3. Create and activate a virtual environment
Windows:
```
python -m venv venv
venv\Scripts\activate
```
Mac/Linux:
```
python -m venv venv
source venv/bin/activate
```
You'll know it worked if you see `(venv)` at the start of your terminal line.

### 4. Install dependencies
```
pip install -r requirements.txt
```

### 5. Confirm the model file is present
Run `dir` (Windows) or `ls` (Mac/Linux) inside the FOCUS folder — you should see `face_landmarker.task` (~3.6MB) listed. It's already included in the repo, so no separate download is needed.

If it's missing for any reason, re-download it with:
```
curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 6. Run the individual test scripts (optional, for debugging one module at a time)
```
python test_webcam.py
python test_facemesh.py
python test_ear.py
python test_headpose.py
python test_expression.py
```
Each opens a webcam window testing one piece of the pipeline. Press `q` to close any of them.

### 7. Run the full combined pipeline (console window version)
```
python engagement_combined.py
```
Shows live Engagement score + status (ENGAGED/DISENGAGED) plus the individual expr/pose_dev/drowsy values, in a plain OpenCV window.

### 8. Run the full dashboard (browser version)
```
streamlit run dashboard.py
```
Opens a browser tab. Check "Start Camera" to begin. Shows live video with per-face engagement scores, classroom engagement index, and focus alerts.

## Syncing changes between laptops

**Before starting work on any laptop**, pull the latest changes:
```
git pull
```

**After making changes you want saved**, push them:
```
git add .
git commit -m "describe what you changed"
git push
```

## Calibration note
EAR (drowsiness) and yaw (head pose) thresholds may need re-tuning per camera/laptop, since camera quality and angle affect the raw values. See `engagement_combined.py` — adjust the `WEIGHTS` and `ALERT_THRESHOLD` values, and the `closed`/`opened` parameters in `drowsiness_from_ear()`, based on live testing on each machine.
