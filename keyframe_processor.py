"""
keyframe_processor.py  —  Stage 3: 단일 팔 키프레임 추출

출력:
  out_dir/keyframes.npy       (K, 6)
  out_dir/keyframe_meta.json
"""

import json
from pathlib import Path

import numpy as np

from human_ik import pose_to_single_arm_state_with_calib, load_calib, clamp_to_limits

GRIP_CLOSED = 0.0
GRIP_OPEN   = float(np.pi / 2)
NEUTRAL_ARM = np.array([0.0, 0.0, 0.3, 0.0, 0.0, GRIP_OPEN], dtype=np.float32)

_LO = np.array([-1.8, -2.0, 0.0, 0.0, -1.5, 0.0], dtype=np.float32)
_HI = np.array([ 1.8,  0.5, 2.5, 1.5,  1.5, 1.57], dtype=np.float32)


def _clamp(state): return np.clip(state, _LO, _HI)


def _window_avg(data, center, window):
    T = len(data); half = window // 2
    lo = max(0, center - half); hi = min(T, center + half + 1)
    return data[lo:hi].mean(axis=0).astype(np.float32)


def process_keyframes(
    out_dir: Path,
    analysis: dict,
    arm: str = "right",
    smooth_window: int = 7,
) -> tuple[np.ndarray, list[dict]]:

    out_dir   = Path(out_dir)
    raw_poses = np.load(out_dir / "raw_poses.npy")   # (T, 6)
    T = len(raw_poses)

    lms_path = out_dir / "raw_landmarks.npy"
    raw_lms  = np.load(lms_path) if lms_path.exists() else None  # (T, 33, 3)

    vlm_kfs = analysis.get("keyframes", [])
    if not vlm_kfs:
        raise ValueError("VLM 분석 결과에 keyframes가 없습니다.")

    vlm_kfs = sorted(vlm_kfs, key=lambda x: float(x.get("importance", 0)), reverse=True)
    vlm_kfs = [kf for kf in vlm_kfs if float(kf.get("importance", 0)) >= 4]
    vlm_kfs = sorted(vlm_kfs, key=lambda x: float(x.get("timestamp", 0)))
    print(f"  [Stage3] importance≥4 키프레임: {len(vlm_kfs)}개  [{arm} arm]")

    SH_I = 12 if arm == "right" else 11
    EL_I = 14 if arm == "right" else 13
    WR_I = 16 if arm == "right" else 15

    keyframes  = []
    kf_pos     = []   # (K, 9)  [sh(3), el(3), wr(3)] in MediaPipe world space
    kf_meta    = []

    for vkf in vlm_kfs:
        frame_idx = min(max(int(vkf.get("frame_idx", 0)), 0), T - 1)

        # position-based IK: 랜드마크 평균 후 IK (각도 평균보다 정확)
        if raw_lms is not None:
            avg_pts = _window_avg(raw_lms, frame_idx, smooth_window)
            pose = pose_to_single_arm_state_with_calib(avg_pts, side=arm, has_depth=True)
            sh = avg_pts[SH_I]; el = avg_pts[EL_I]; wr = avg_pts[WR_I]
            kf_pos.append(np.concatenate([sh, el, wr]).astype(np.float32))
        else:
            pose = _window_avg(raw_poses, frame_idx, smooth_window)
            kf_pos.append(np.zeros(9, dtype=np.float32))

        # VLM 손 상태 반영 (그리퍼)
        hand_key = f"{arm}_hand"
        hand = vkf.get(hand_key) or vkf.get("left_hand" if arm == "left" else "right_hand") or {}

        def _not_visible(h):
            if not h: return True
            if h.get("state") is None: return True
            hint = h.get("spatial_hint", "")
            return isinstance(hint, str) and "not visible" in hint.lower()

        if _not_visible(hand):
            if abs(pose[:3]).max() < 0.05:
                pose = NEUTRAL_ARM.copy()
                print(f"  [Stage3] {arm}팔 미감지 → neutral")
            else:
                pose[5] = GRIP_OPEN
        else:
            state_map = {"closed": GRIP_CLOSED, "open": GRIP_OPEN, "partial": GRIP_OPEN * 0.5}
            lstate = (hand.get("state") or "").lower()
            if lstate in state_map:
                pose[5] = state_map[lstate]

            # pose_correction
            correction = vkf.get("pose_correction", {})
            ctype = correction.get("type", "free")
            if ctype in ("contact", "grasp"):
                pose[5] = GRIP_CLOSED
            elif ctype == "release":
                pose[5] = GRIP_OPEN

        pose = _clamp(pose)
        keyframes.append(pose)

        ts  = vkf.get("timestamp", 0)
        lbl = vkf.get("label", "?")
        imp = vkf.get("importance", "?")
        print(f"  [Stage3] [{ts:5.1f}s] {lbl:10s} imp={imp}  "
              f"hand={hand.get('state','?')}/{hand.get('contact','?')}")

        kf_meta.append({
            "frame_idx":   frame_idx,
            "timestamp":   float(ts),
            "label":       lbl,
            "description": vkf.get("description", ""),
            "importance":  int(vkf.get("importance", 5)),
            "hand":        hand,
        })

    if not keyframes:
        raise ValueError("조건을 만족하는 키프레임이 없습니다.")

    kf_arr     = np.array(keyframes, dtype=np.float32)
    kf_pos_arr = np.array(kf_pos,    dtype=np.float32)  # (K, 9)
    np.save(out_dir / "keyframes.npy",     kf_arr)
    np.save(out_dir / "keyframes_pos.npy", kf_pos_arr)
    with open(out_dir / "keyframe_meta.json", "w", encoding="utf-8") as f:
        json.dump(kf_meta, f, ensure_ascii=False, indent=2)

    print(f"  [Stage3] 저장: keyframes.npy {kf_arr.shape}  keyframes_pos.npy {kf_pos_arr.shape}")
    return kf_arr, kf_meta


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out",           required=True)
    p.add_argument("--arm",           default="right", choices=["right", "left"])
    p.add_argument("--smooth_window", type=int, default=7)
    args = p.parse_args()
    out_dir = Path(args.out)
    with open(out_dir / "analysis.json", encoding="utf-8") as f:
        analysis = json.load(f)
    kf, meta = process_keyframes(out_dir, analysis, args.arm, args.smooth_window)
    print(f"키프레임: {kf.shape}")
