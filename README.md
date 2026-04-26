# human_demo_singlearm

Single-arm robot control from human demonstration video.  
Records a human performing a task → extracts 3D joint positions → generates a smooth robot arm trajectory via IK.

---

## Pipeline (4 stages)

```
Video
  Stage 1  extract_poses.py       → raw 3D landmarks (T, 33, 3)  +  raw joint angles (T, 6)
  Stage 2  vlm_analyzer.py        → Gemini VLM selects keyframes + task structure
  Stage 3  keyframe_processor.py  → averaged 3D positions per keyframe (K, 9)  +  IK angles (K, 6)
  Stage 4  trajectory_optimizer.py→ Cartesian min-jerk interpolation → IK at each point → trajectory (T2, 6)
```

Joint order: `[pan, lift, elbow, wflx, wrol, grip]`

---

## Requirements

```bash
conda create -n lehome python=3.12
conda activate lehome

pip install mediapipe opencv-python numpy scipy matplotlib
pip install mujoco
pip install google-generativeai          # Gemini API (Stage 2)

# Depth estimation (Stage 1, optional)
pip install transformers torch           # Depth Anything V2
```

**Model files** (place in `../human_demo/`):
- `pose_landmarker_full.task` — [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- `hand_landmarker.task` — [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) (optional)

---

## Quick Start

```bash
cd human_demo_singlearm
conda activate lehome
export GEMINI_API_KEY="your_key_here"

# 1. Record calibration video  (raise/lower arm slowly, ~30s)
python record_video.py --out calib.mp4

# 2. Measure arm lengths  →  arm_calib.json  +  MuJoCo viewer
python calibrate_arm.py --video calib.mp4 --arm right

# 3. Record demonstration video
python record_video.py --out demo.mp4

# 4. Run full pipeline
python pipeline.py --video demo.mp4 --arm right

# 5. Visualize result
python visualize_skeleton.py --out output/<session_folder> --arm right

# 6. Debug / diagnose
python debug_visualize.py --out output/<session_folder> --arm right
```

### Resume from a specific stage

```bash
# Re-run from Stage 3 onward (reuse Stage 1 & 2 results)
python pipeline.py --video demo.mp4 --arm right --from_stage 3 --out output/<session_folder>
```

---

## File Structure

```
human_demo_singlearm/
  pipeline.py              Main entry point (4-stage orchestrator)
  extract_poses.py         Stage 1: video → MediaPipe landmarks → IK
  vlm_analyzer.py          Stage 2: Gemini VLM keyframe selection  (shared from human_demo_v2)
  keyframe_processor.py    Stage 3: keyframe 3D positions + IK angles
  trajectory_optimizer.py  Stage 4: Cartesian interpolation + per-point IK
  human_ik.py              IK / FK math  (mp_to_mujoco, solve_arm_ik, fk_arm, ...)
  calibrate_arm.py         Arm length measurement + MuJoCo XML generation
  record_video.py          Webcam recording with mirror preview
  visualize_skeleton.py    Animated 3-panel skeleton viewer
  debug_visualize.py       3 debug plots: timeline / wrist trajectory / pipeline animation
```

### Pipeline output (per session)

```
output/<session>/
  frames/                  Extracted JPEG frames
  raw_landmarks.npy        (T, 33, 3)  MediaPipe world landmarks
  raw_landmarks_2d.npy     (T, 33, 2)  Normalized image coordinates
  raw_poses.npy            (T, 6)      Per-frame IK joint angles
  timestamps.npy           (T,)
  analysis.json            Gemini VLM output
  keyframe_captures.png    Keyframe grid with 3D position overlay
  keyframes.npy            (K, 6)      Keyframe joint angles
  keyframes_pos.npy        (K, 9)      Keyframe 3D positions [sh, el, wr]
  keyframe_meta.json
  trajectory.npy           (T2, 6)     Final robot trajectory
  trajectory_meta.json
  debug_1_joint_timeline.png
  debug_2_wrist_trajectory.png
```

---

## Key Design Decisions

**Cartesian-space interpolation (not joint-space)**  
Keyframe 3D positions are interpolated in Cartesian space with a minimum-jerk profile.  
IK is applied at each interpolated point — this avoids the unnatural arcing that occurs when interpolating joint angles directly.

**Visibility-based filtering + velocity outlier rejection**  
MediaPipe occasionally swaps left/right landmark assignments.  
`extract_poses.py` filters frames with low landmark visibility and interpolates over velocity outliers (> 1.2 rad/frame).

**Coordinate transform**  
`mp_to_mujoco(v) = [-v[2], -v[0], v[1]]`  
MediaPipe world: x=right, y=up, z=toward-camera  
MuJoCo: x=forward, y=side, z=up

---

## Joint Limits

| Joint | Min (rad) | Max (rad) | Notes |
|-------|-----------|-----------|-------|
| pan   | -1.8 | 1.8 | shoulder horizontal rotation |
| lift  | -2.0 | 1.8 | shoulder vertical — allows arm fully raised |
| elbow | 0.0  | 2.5 | elbow flex |
| wflx  | 0.0  | 1.5 | wrist flex |
| wrol  | -1.5 | 1.5 | wrist roll |
| grip  | 0.0  | 1.57 | gripper open/close |

---

## Notes

- `arm_calib.json` and `calib_robot_*.xml` are machine-specific and excluded from version control.
- Demo videos and `output/` folders are excluded (personal data + large files).
- Gemini API key must be set as `GEMINI_API_KEY` environment variable before running Stage 2.
