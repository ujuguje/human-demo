"""
extract_poses.py  —  Stage 1: 영상 → 단일 팔 포즈 시퀀스

출력:
  out_dir/
    ├── raw_poses.npy      (T, 6)  float32, rad
    ├── raw_landmarks.npy  (T, 33, 3)
    ├── timestamps.npy     (T,)
    └── frames/
"""

import sys
from pathlib import Path

V1_DIR = Path(__file__).parent.parent / "human_demo"
sys.path.insert(0, str(V1_DIR))

import cv2
import numpy as np

from camera_utils import backproject_landmarks, get_intrinsics
from depth_estimator import DepthEstimator
from pose_retarget import pixel_landmarks_to_np, world_landmarks_to_np

from human_ik import pose_to_single_arm_state_with_calib, load_calib


def _cam_to_body(pts: np.ndarray) -> np.ndarray:
    out = pts.copy(); out *= -1; return out


def _arm_visible(pose_lms, arm: str, threshold: float = 0.55) -> bool:
    """활성 팔의 핵심 랜드마크(어깨/팔꿈치/손목) 가시성이 모두 충분한지 확인."""
    from pose_retarget import LM
    for key in (f"{arm}_shoulder", f"{arm}_elbow", f"{arm}_wrist"):
        if pose_lms[LM[key]].visibility < threshold:
            return False
    return True


def _reject_velocity_outliers(poses: np.ndarray, threshold: float = 1.2) -> np.ndarray:
    """
    프레임 간 관절 각도 변화가 threshold(rad)를 초과하면 outlier로 간주하고
    인접 유효 프레임 사이를 선형 보간으로 대체.
    기준: 30fps에서 1.2 rad/frame ≈ 약 2000°/s (팔 교체 수준의 점프)
    """
    poses = poses.copy()
    T = len(poses)

    # 전 관절이 0인 프레임 = 미검출 프레임 → invalid
    valid = ~np.all(poses == 0, axis=1)

    # 속도 초과 프레임 = outlier → invalid
    for t in range(1, T):
        if valid[t] and valid[t - 1]:
            if np.abs(poses[t] - poses[t - 1]).max() > threshold:
                valid[t] = False

    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2:
        return poses  # 유효 프레임이 너무 없으면 그대로

    for t in range(T):
        if valid[t]:
            continue
        left  = valid_idx[valid_idx < t]
        right = valid_idx[valid_idx > t]
        if len(left) == 0:
            poses[t] = poses[right[0]]
        elif len(right) == 0:
            poses[t] = poses[left[-1]]
        else:
            l, r = int(left[-1]), int(right[0])
            alpha = (t - l) / (r - l)
            poses[t] = poses[l] + alpha * (poses[r] - poses[l])

    return poses


def _get_hand_lms(hand_result, person_side: str):
    if not hand_result.handedness:
        return None
    for i, hl in enumerate(hand_result.handedness):
        mp_side = hl[0].category_name
        ps = "right" if mp_side == "Left" else "left"
        if ps == person_side and i < len(hand_result.hand_world_landmarks):
            return hand_result.hand_world_landmarks[i]
    return None


def extract_poses(
    video_path: str,
    out_dir: Path,
    arm: str = "right",
    device: str = "cuda",
    depth_skip: int = 3,
) -> None:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    load_calib()

    pose_model = str(V1_DIR / "pose_landmarker_full.task")
    landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_model),
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
    )

    hand_model = str(V1_DIR / "hand_landmarker.task")
    hand_landmarker = None
    if Path(hand_model).exists():
        hand_landmarker = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_model),
                num_hands=1,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                running_mode=mp_vision.RunningMode.VIDEO,
            )
        )
        print(f"  [Stage1] Hands 활성화 ({arm} arm)")
    else:
        print(f"  [Stage1] hand_landmarker.task 없음 → Hands 스킵")

    print(f"  [Stage1] Depth Estimator 로드 중 (device={device})...")
    depth_est = DepthEstimator(device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = get_intrinsics(str(video_path), W, H)
    print(f"  [Stage1] {Path(video_path).name}  {W}×{H}  {total_frames}프레임  [{arm} arm]")

    raw_poses  = []
    raw_lms    = []
    raw_lms_2d = []
    timestamps = []
    depth_map  = None
    frame_idx  = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        ts = frame_idx / orig_fps
        ts_ms = int(ts * 1000)

        if frame_idx % depth_skip == 0:
            try:
                depth_map = depth_est.predict(frame_bgr)
            except Exception:
                pass

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        pose_result = landmarker.detect_for_video(mp_image, ts_ms)

        hand_lms = None
        if hand_landmarker is not None:
            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            hand_lms = _get_hand_lms(hand_result, arm)

        arm_ok = (
            pose_result.pose_landmarks
            and pose_result.pose_world_landmarks
            and _arm_visible(pose_result.pose_landmarks[0], arm)
        )
        if arm_ok:
            try:
                has_depth = depth_map is not None
                if has_depth:
                    lms_2d_px = pixel_landmarks_to_np(pose_result.pose_landmarks[0], W, H)
                    pts_cam   = backproject_landmarks(lms_2d_px, depth_map, K)
                    pts_3d    = _cam_to_body(pts_cam)
                else:
                    pts_3d = world_landmarks_to_np(pose_result.pose_world_landmarks[0])

                state = pose_to_single_arm_state_with_calib(
                    pts_3d, side=arm, has_depth=has_depth, hand_lms=hand_lms)
                wlm = world_landmarks_to_np(pose_result.pose_world_landmarks[0])
                lm2d = np.array([[lm.x, lm.y] for lm in pose_result.pose_landmarks[0]],
                                 dtype=np.float32)
            except Exception:
                state = np.zeros(6, dtype=np.float32)
                wlm   = np.zeros((33, 3), dtype=np.float32)
                lm2d  = np.zeros((33, 2), dtype=np.float32)
        else:
            state = np.zeros(6, dtype=np.float32)
            wlm   = np.zeros((33, 3), dtype=np.float32)
            lm2d  = np.zeros((33, 2), dtype=np.float32)

        raw_poses.append(state)
        raw_lms.append(wlm)
        raw_lms_2d.append(lm2d)
        timestamps.append(ts)

        cv2.imwrite(str(out_dir / "frames" / f"{frame_idx:06d}.jpg"),
                    frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  [Stage1] {frame_idx}/{total_frames}", end="\r")

    cap.release()
    landmarker.close()
    if hand_landmarker:
        hand_landmarker.close()

    raw_poses  = np.array(raw_poses,  dtype=np.float32)
    raw_lms    = np.array(raw_lms,    dtype=np.float32)
    raw_lms_2d = np.array(raw_lms_2d, dtype=np.float32)
    timestamps = np.array(timestamps, dtype=np.float64)

    # 팔 교체 수준의 점프를 보간으로 대체
    n_before = int(np.all(raw_poses == 0, axis=1).sum())
    raw_poses = _reject_velocity_outliers(raw_poses, threshold=1.2)
    n_after   = int(np.all(raw_poses == 0, axis=1).sum())
    print(f"  [Stage1] outlier 보간: 미검출={n_before}f → 보간 후 zero={n_after}f")

    np.save(out_dir / "raw_poses.npy",        raw_poses)
    np.save(out_dir / "raw_landmarks.npy",    raw_lms)
    np.save(out_dir / "raw_landmarks_2d.npy", raw_lms_2d)
    np.save(out_dir / "timestamps.npy",       timestamps)

    print(f"\n  [Stage1] 완료: {frame_idx}프레임  raw_poses {raw_poses.shape}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video",      required=True)
    p.add_argument("--out",        required=True)
    p.add_argument("--arm",        default="right", choices=["right", "left"])
    p.add_argument("--device",     default="cuda")
    p.add_argument("--depth_skip", type=int, default=3)
    args = p.parse_args()
    extract_poses(args.video, Path(args.out), args.arm, args.device, args.depth_skip)
