"""
human_ik.py  —  Single-arm position-based IK (human_demo_v2에서 가져옴)

좌표계:
  MediaPipe → MuJoCo:  [-v[2], -v[0], v[1]]

MuJoCo FK:
  오른팔: T-pose=[0,-1,0]  lift=Rx(lift)  pan=Rz(+pan)  elbow=R_localZ(+elbow)
  왼팔:   T-pose=[0,+1,0]  lift=Rx(lift)  pan=Rz(-pan)  elbow=R_localZ(-elbow)

  ua_right = [cos(l)*sin(p), -cos(l)*cos(p), sin(l)]
  ua_left  = [cos(l)*sin(p),  cos(l)*cos(p), sin(l)]
"""

import json
from pathlib import Path

import numpy as np

_LM = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow":    13, "right_elbow":    14,
    "left_wrist":    15, "right_wrist":    16,
    "left_index":    19, "right_index":    20,
    "left_pinky":    17, "right_pinky":    18,
    "left_thumb":    21, "right_thumb":    22,
}

_LIMITS = [(-1.8, 1.8), (-2.0, 1.8), (0.0, 2.5), (0.0, 1.5), (-1.5, 1.5), (0.0, 1.57)]


def mp_to_mujoco(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return np.array([-v[2], -v[0], v[1]])


def _n(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > eps else np.zeros_like(v)


def hand_to_gripper(hand_world_lms) -> float:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_world_lms], dtype=np.float32)
    mcp_idx = [5, 9, 13, 17]; pip_idx = [6, 10, 14, 18]; dip_idx = [7, 11, 15, 19]
    angles = []
    for mcp, pip, dip in zip(mcp_idx, pip_idx, dip_idx):
        v1 = pts[pip] - pts[mcp]; v2 = pts[dip] - pts[pip]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            angles.append(np.arccos(float(np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0))))
    if not angles:
        return float(np.pi / 2)
    return float(np.clip(np.pi / 2 - np.mean(angles), 0.0, np.pi / 2))


def solve_arm_ik(
    shoulder_mp, elbow_mp, wrist_mp, index_mp, pinky_mp, thumb_mp,
    side: str, L1: float, L2: float,
    has_depth: bool = True, hand_world_lms=None,
) -> list[float]:
    if float(np.linalg.norm(np.asarray(elbow_mp) - np.asarray(shoulder_mp))) < 0.05:
        return [0.0, 0.0, 0.5, 0.0, 0.0, float(np.pi / 2)]

    S = mp_to_mujoco(shoulder_mp); E = mp_to_mujoco(elbow_mp)
    W = mp_to_mujoco(wrist_mp);    I = mp_to_mujoco(index_mp)
    P = mp_to_mujoco(pinky_mp)

    ua = _n(E - S); fa = _n(W - E)

    shoulder_lift = float(np.arcsin(np.clip(ua[2], -1.0, 1.0)))
    cos_lift = float(np.cos(shoulder_lift))

    if abs(cos_lift) < 1e-6:
        shoulder_pan = 0.0
    elif side == "left":
        shoulder_pan = float(np.arctan2(ua[0], ua[1]))
    else:
        shoulder_pan = float(np.arctan2(ua[0], -ua[1]))

    L1_h = float(np.linalg.norm(E - S)); L2_h = float(np.linalg.norm(W - E))
    d_h  = float(np.linalg.norm(W - S)); total_h = L1_h + L2_h
    if total_h > 1e-6:
        scale = (L1 + L2) / total_h
        d_r = float(np.clip(d_h * scale, abs(L1 - L2) + 0.005, L1 + L2 - 0.005))
        cos_alpha = (L1**2 + L2**2 - d_r**2) / (2.0 * L1 * L2)
        elbow_flex = float(np.pi - np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
    else:
        elbow_flex = 0.5

    ha = _n(I - W)
    wrist_flex = float(np.arccos(np.clip(np.dot(fa, ha), -1.0, 1.0)))

    if has_depth:
        fa_ax = _n(W - E)
        v1 = _n(I - W); v2 = _n(P - W)
        hn = _n(np.cross(v1, v2))
        hn_p = _n(hn - np.dot(hn, fa_ax) * fa_ax)
        up = np.array([0., 0., 1.])
        up_p = _n(up - np.dot(up, fa_ax) * fa_ax)
        if np.linalg.norm(up_p) > 1e-6 and np.linalg.norm(hn_p) > 1e-6:
            cross = np.cross(up_p, hn_p)
            wrist_roll = float(np.sign(np.dot(cross, fa_ax)) *
                               np.arccos(np.clip(np.dot(up_p, hn_p), -1.0, 1.0)))
        else:
            wrist_roll = 0.0
    else:
        wrist_roll = 0.0

    if hand_world_lms is not None:
        gripper = hand_to_gripper(hand_world_lms)
    elif thumb_mp is not None:
        T_m = mp_to_mujoco(thumb_mp)
        gripper = float(np.clip(np.linalg.norm(T_m - I) / (float(np.linalg.norm(E - S)) + 1e-8),
                                0.0, 1.0) * (np.pi / 2))
    else:
        gripper = float(np.pi / 2)

    return [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]


def clamp_to_limits(angles: list[float]) -> list[float]:
    return [float(np.clip(a, lo, hi)) for a, (lo, hi) in zip(angles, _LIMITS)]


def ik_arm_only(sh_mp, el_mp, wr_mp, side: str, L1: float, L2: float) -> np.ndarray:
    """
    어깨/팔꿈치/손목 3D 위치(MediaPipe) → 팔 관절 3개만 (pan, lift, elbow) 반환.
    손목/그리퍼는 포함하지 않음 — Cartesian 보간 trajectory에서 사용.
    """
    sh = np.asarray(sh_mp, dtype=np.float64)
    el = np.asarray(el_mp, dtype=np.float64)
    wr = np.asarray(wr_mp, dtype=np.float64)

    if np.linalg.norm(el - sh) < 0.05:
        return np.array([0.0, 0.0, 0.5])

    S = mp_to_mujoco(sh); E = mp_to_mujoco(el); W = mp_to_mujoco(wr)
    ua = _n(E - S)

    shoulder_lift = float(np.arcsin(np.clip(ua[2], -1.0, 1.0)))
    cos_lift = float(np.cos(shoulder_lift))
    if abs(cos_lift) < 1e-6:
        shoulder_pan = 0.0
    elif side == "left":
        shoulder_pan = float(np.arctan2(ua[0], ua[1]))
    else:
        shoulder_pan = float(np.arctan2(ua[0], -ua[1]))

    L1_h = float(np.linalg.norm(E - S)); L2_h = float(np.linalg.norm(W - E))
    d_h  = float(np.linalg.norm(W - S)); total_h = L1_h + L2_h
    if total_h > 1e-6:
        scale = (L1 + L2) / total_h
        d_r = float(np.clip(d_h * scale, abs(L1 - L2) + 0.005, L1 + L2 - 0.005))
        cos_alpha = (L1**2 + L2**2 - d_r**2) / (2.0 * L1 * L2)
        elbow_flex = float(np.pi - np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
    else:
        elbow_flex = 0.5

    pan_lo, pan_hi   = _LIMITS[0]
    lift_lo, lift_hi = _LIMITS[1]
    el_lo, el_hi     = _LIMITS[2]
    return np.array([
        np.clip(shoulder_pan,  pan_lo,  pan_hi),
        np.clip(shoulder_lift, lift_lo, lift_hi),
        np.clip(elbow_flex,    el_lo,   el_hi),
    ], dtype=np.float64)


def pose_to_single_arm_state(
    pts_3d: np.ndarray,
    side: str = "right",
    L1: float = 0.25, L2: float = 0.22,
    has_depth: bool = True,
    hand_lms=None,
) -> np.ndarray:
    """MediaPipe (33,3) → 단일 팔 관절 각도 (6,)."""
    def _g(key): return pts_3d[_LM[key]]
    angles = solve_arm_ik(
        _g(f"{side}_shoulder"), _g(f"{side}_elbow"), _g(f"{side}_wrist"),
        _g(f"{side}_index"),    _g(f"{side}_pinky"), _g(f"{side}_thumb"),
        side=side, L1=L1, L2=L2, has_depth=has_depth, hand_world_lms=hand_lms,
    )
    return np.array(clamp_to_limits(angles), dtype=np.float32)


# ── 캘리브레이션 ──────────────────────────────────────────

_calib_cache: dict | None = None

def load_calib(path: str | Path | None = None) -> dict:
    global _calib_cache
    if _calib_cache is not None:
        return _calib_cache
    default = {"left_upper_arm": 0.25, "left_forearm": 0.22,
                "right_upper_arm": 0.25, "right_forearm": 0.22}
    if path is None:
        path = Path(__file__).parent / "arm_calib.json"
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            _calib_cache = json.load(f)
        print(f"  [IK] 캘리브레이션 로드: {path}")
    else:
        _calib_cache = default
        print("  [IK] arm_calib.json 없음 → 기본값 사용")
    return _calib_cache


def pose_to_single_arm_state_with_calib(
    pts_3d: np.ndarray,
    side: str = "right",
    has_depth: bool = True,
    hand_lms=None,
) -> np.ndarray:
    c = load_calib()
    return pose_to_single_arm_state(
        pts_3d, side=side,
        L1=c[f"{side}_upper_arm"], L2=c[f"{side}_forearm"],
        has_depth=has_depth, hand_lms=hand_lms,
    )


# ── FK (MuJoCo 정확 재현) ────────────────────────────────

def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def _rodrigues(axis, angle, v):
    axis = _n(np.asarray(axis, dtype=np.float64))
    v    = np.asarray(v, dtype=np.float64)
    return (v * np.cos(angle) + np.cross(axis, v) * np.sin(angle)
            + axis * np.dot(axis, v) * (1 - np.cos(angle)))

def fk_arm(q6, shoulder_world, L1, L2, HL=0.07, is_right=True):
    """MuJoCo FK → [Shoulder, Elbow, Wrist, Tip] (4, 3)."""
    pan, lift, elbow_ang = float(q6[0]), float(q6[1]), float(q6[2])
    t_pose = np.array([0., -1. if is_right else 1., 0.])
    R = _rot_z(pan if is_right else -pan) @ _rot_x(lift)
    ua = _n(R @ t_pose)
    ep = np.asarray(shoulder_world, dtype=np.float64) + ua * L1
    local_z = R @ np.array([0., 0., 1.])
    ea = local_z if is_right else -local_z
    fa = _n(_rodrigues(ea, elbow_ang, ua))
    wp = ep + fa * L2
    return np.array([shoulder_world, ep, wp, wp + fa * HL], dtype=np.float64)
