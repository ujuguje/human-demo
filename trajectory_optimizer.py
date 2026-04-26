"""
trajectory_optimizer.py  —  Stage 4: 단일 팔 키프레임 → Minimum Jerk Trajectory

Cartesian 보간 방식:
  키프레임 3D 위치(shoulder, elbow, wrist)를 Cartesian 공간에서 Min-Jerk 보간 →
  각 보간 포인트에서 IK 계산 → 매끄러운 관절 각도 trajectory

  관절각도 공간에서 직접 보간하면 팔이 물리적으로 이상한 경로를 만들 수 있음.
  Cartesian → IK 는 팔이 실제 직선(또는 부드러운 곡선)으로 움직이게 보장.

출력:
  out_dir/trajectory.npy   (T, 6) float32
"""

import json
from pathlib import Path

import numpy as np

_LO = np.array([-1.8, -2.0, 0.0, 0.0, -1.5, 0.0],   dtype=np.float32)
_HI = np.array([ 1.8,  1.8, 2.5, 1.5,  1.5, 1.57], dtype=np.float32)


def _min_jerk(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return 10*t**3 - 15*t**4 + 6*t**5


def _duration(delta_max: float, max_vel: float) -> float:
    return float(np.clip(delta_max / max(max_vel, 1e-6), 0.3, 5.0))


def optimize_trajectory(
    keyframes: np.ndarray,       # (K, 6) — grip/wflx/wrol 등 보간용
    kf_meta: list[dict],
    out_dir: Path,
    fps: float = 30.0,
    max_velocity: float = 1.0,
    hold_frames: int = 10,
    arm: str = "right",
) -> np.ndarray:

    out_dir = Path(out_dir)
    K = len(keyframes)

    if K == 0:
        raise ValueError("키프레임이 없습니다.")
    if K == 1:
        traj = np.tile(keyframes[0], (hold_frames, 1)).astype(np.float32)
        np.save(out_dir / "trajectory.npy", traj)
        return traj

    # ── Cartesian 위치 기반 IK 사용 가능 여부 확인 ──────────────
    pos_path = out_dir / "keyframes_pos.npy"
    use_cartesian = pos_path.exists()

    if use_cartesian:
        kf_pos = np.load(pos_path)   # (K, 9)  [sh(3), el(3), wr(3)]
        # 위치가 모두 0인 키프레임은 각도 보간 fallback
        use_cartesian = not np.all(kf_pos == 0)

    if use_cartesian:
        traj = _cartesian_ik_trajectory(keyframes, kf_pos, kf_meta, fps,
                                         max_velocity, hold_frames, arm, out_dir)
    else:
        print("  [Stage4] keyframes_pos.npy 없음 → 관절각도 보간 사용")
        traj = _joint_space_trajectory(keyframes, kf_meta, fps,
                                        max_velocity, hold_frames)

    np.save(out_dir / "trajectory.npy", traj)
    total_sec = len(traj) / fps
    print(f"  [Stage4] trajectory: {traj.shape}  ({total_sec:.1f}s @ {fps}fps)")

    summary = {
        "n_keyframes": K, "n_frames": int(len(traj)),
        "duration_sec": float(total_sec), "fps": float(fps),
        "max_velocity": float(max_velocity), "hold_frames": hold_frames,
        "method": "cartesian_ik" if use_cartesian else "joint_space",
        "joint_names": ["pan", "lift", "elbow", "wflx", "wrol", "grip"],
    }
    with open(out_dir / "trajectory_meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return traj


# ──────────────────────────────────────────────────────────────
# Cartesian 보간 + IK
# ──────────────────────────────────────────────────────────────

def _cartesian_ik_trajectory(
    keyframes: np.ndarray,   # (K, 6) — 손목/그리퍼 각도
    kf_pos:    np.ndarray,   # (K, 9) — sh+el+wr 3D positions (MediaPipe world)
    kf_meta:   list[dict],
    fps:       float,
    max_vel:   float,
    hold_frames: int,
    arm:       str,
    out_dir:   Path,
) -> np.ndarray:

    from human_ik import ik_arm_only, load_calib
    calib = load_calib()
    L1 = calib.get(f"{arm}_upper_arm", 0.25)
    L2 = calib.get(f"{arm}_forearm",   0.22)

    K = len(keyframes)
    segments = []

    for i in range(K - 1):
        pos0, pos1 = kf_pos[i], kf_pos[i + 1]
        q0,   q1   = keyframes[i], keyframes[i + 1]

        label0 = kf_meta[i  ].get("label", "?") if i   < len(kf_meta) else "?"
        label1 = kf_meta[i+1].get("label", "?") if i+1 < len(kf_meta) else "?"

        # 이동 거리(손목)로 구간 시간 결정
        wr0 = pos0[6:9]; wr1 = pos1[6:9]
        wrist_dist = float(np.linalg.norm(wr1 - wr0))
        dur = _duration(wrist_dist, max_vel * 0.5)   # 위치 기반 속도
        n_move = max(2, int(dur * fps))

        print(f"  [Stage4] {i}: {label0} → {label1}  "
              f"wrist_dist={wrist_dist:.3f}m  {dur:.2f}s  ({n_move}f)  [Cartesian IK]")

        # Min-Jerk 위치 보간
        s = _min_jerk(n_move)                   # (n_move,)
        pos_seg = pos0 + s[:, None] * (pos1 - pos0)   # (n_move, 9)

        # 손목·그리퍼 각도 보간 (wflx, wrol, grip)
        hand_seg = q0[3:] + s[:, None] * (q1[3:] - q0[3:])  # (n_move, 3)

        # 각 포인트마다 팔 IK
        arm_seg = np.zeros((n_move, 3), dtype=np.float64)
        for t in range(n_move):
            sh = pos_seg[t, 0:3]; el = pos_seg[t, 3:6]; wr = pos_seg[t, 6:9]
            arm_seg[t] = ik_arm_only(sh, el, wr, arm, L1, L2)

        seg = np.concatenate([arm_seg, hand_seg], axis=1).astype(np.float32)
        seg = seg[:-1]   # 마지막 포인트는 다음 구간 시작과 겹치므로 제외
        segments.append(seg)

        # 키프레임 홀드
        q_hold = _ik_pose_from_pos(pos1, q1, arm, L1, L2)
        segments.append(np.tile(q_hold, (hold_frames, 1)))

    # 마지막 키프레임
    segments.append(keyframes[-1][np.newaxis, :])

    traj = np.concatenate(segments, axis=0).astype(np.float32)
    traj = np.clip(traj, _LO, _HI)

    # 가벼운 스무딩 (IK 노이즈 제거)
    try:
        from scipy.ndimage import gaussian_filter1d
        traj = gaussian_filter1d(traj, sigma=1.0, axis=0).astype(np.float32)
        traj = np.clip(traj, _LO, _HI)
    except ImportError:
        pass

    return traj


def _ik_pose_from_pos(pos9: np.ndarray, q6: np.ndarray,
                       arm: str, L1: float, L2: float) -> np.ndarray:
    from human_ik import ik_arm_only
    sh = pos9[0:3]; el = pos9[3:6]; wr = pos9[6:9]
    arm_q = ik_arm_only(sh, el, wr, arm, L1, L2)
    return np.clip(
        np.array([*arm_q, *q6[3:]], dtype=np.float32),
        _LO, _HI,
    )


# ──────────────────────────────────────────────────────────────
# 관절각도 공간 보간 (fallback)
# ──────────────────────────────────────────────────────────────

def _joint_space_trajectory(
    keyframes:   np.ndarray,
    kf_meta:     list[dict],
    fps:         float,
    max_vel:     float,
    hold_frames: int,
) -> np.ndarray:
    K = len(keyframes)
    segments = []
    for i in range(K - 1):
        q0, q1 = keyframes[i], keyframes[i + 1]
        dur    = _duration(float(np.abs(q1 - q0).max()), max_vel)
        n_move = max(2, int(dur * fps))
        label0 = kf_meta[i  ].get("label", "?") if i   < len(kf_meta) else "?"
        label1 = kf_meta[i+1].get("label", "?") if i+1 < len(kf_meta) else "?"
        print(f"  [Stage4] {i}: {label0} → {label1}  "
              f"Δ={np.abs(q1-q0).max():.2f}rad  {dur:.2f}s  ({n_move}f)")
        s = _min_jerk(n_move + 1)
        seg = (q0 + s[:, None] * (q1 - q0))[:-1]
        segments.append(seg)
        segments.append(np.tile(q1, (hold_frames, 1)))

    segments.append(keyframes[-1][np.newaxis, :])
    traj = np.concatenate(segments, axis=0).astype(np.float32)
    traj = np.clip(traj, _LO, _HI)

    try:
        from scipy.ndimage import gaussian_filter1d
        traj = gaussian_filter1d(traj, sigma=1.5, axis=0).astype(np.float32)
        traj = np.clip(traj, _LO, _HI)
    except ImportError:
        pass

    return traj


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out",          required=True)
    p.add_argument("--arm",          default="right", choices=["right", "left"])
    p.add_argument("--fps",          type=float, default=30.0)
    p.add_argument("--max_velocity", type=float, default=1.0)
    args = p.parse_args()
    out_dir = Path(args.out)
    kfs = np.load(out_dir / "keyframes.npy")
    with open(out_dir / "keyframe_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    traj = optimize_trajectory(kfs, meta, out_dir, args.fps, args.max_velocity, arm=args.arm)
    print(f"trajectory: {traj.shape}")
