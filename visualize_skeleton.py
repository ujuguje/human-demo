"""
visualize_skeleton.py  —  단일 팔: 원본 영상 / Raw 3D / FK trajectory 비교

키보드: Space=일시정지  q=종료
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from human_ik import fk_arm, load_calib

# MediaPipe 연결선 — arm 파라미터에 따라 필터링
_CONNECTIONS_LEFT  = [(11,13),(13,15),(15,17),(15,19)]
_CONNECTIONS_RIGHT = [(12,14),(14,16),(16,18),(16,20)]
_CONNECTIONS_BASE  = [(11,12),(11,0),(12,0)]   # 어깨연결 + 목 (항상 표시)

LEFT_IDX  = {11, 13, 15, 17, 19}
RIGHT_IDX = {12, 14, 16, 18, 20}

def _make_connections(arm: str):
    if arm == "right":
        return _CONNECTIONS_BASE + _CONNECTIONS_RIGHT
    else:
        return _CONNECTIONS_BASE + _CONNECTIONS_LEFT

def _make_labels(arm: str):
    base = {0: "Nose", 11: "L.Shldr", 12: "R.Shldr"}
    if arm == "right":
        base.update({14: "R.Elbow", 16: "R.Wrist"})
    else:
        base.update({13: "L.Elbow", 15: "L.Wrist"})
    return base

FK_LABELS = ["Shldr", "Elbow", "Wrist", "Tip"]
L_HAND = 0.07


def _draw_overlay(frame_rgb, lm2d, connections, labels, arm="right"):
    img = frame_rgb.copy()
    H, W = img.shape[:2]
    active_idx = RIGHT_IDX if arm == "right" else LEFT_IDX
    center_idx = set(range(33)) - LEFT_IDX - RIGHT_IDX
    for a, b in connections:
        pa = (int(lm2d[a, 0]*W), int(lm2d[a, 1]*H))
        pb = (int(lm2d[b, 0]*W), int(lm2d[b, 1]*H))
        if pa == (0,0) or pb == (0,0): continue
        col = (76,195,247) if (a in LEFT_IDX or b in LEFT_IDX) else \
              (255,138,101) if (a in RIGHT_IDX or b in RIGHT_IDX) else (150,170,180)
        cv2.line(img, pa, pb, col, 2, cv2.LINE_AA)
    for i, (nx, ny) in enumerate(lm2d):
        if nx == 0 and ny == 0: continue
        if i not in active_idx and i not in center_idx:
            continue  # 비활성 팔 점 숨기기
        px, py = int(nx*W), int(ny*H)
        col = (76,195,247) if i in LEFT_IDX else \
              (255,138,101) if i in RIGHT_IDX else (180,200,210)
        cv2.circle(img, (px,py), 4, col, -1, cv2.LINE_AA)
        if i in labels:
            cv2.putText(img, labels[i], (px+6,py-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1, cv2.LINE_AA)
    return img


def visualize(out_dir: Path, arm: str = "right", speed: float = 1.0, save_path: str | None = None):
    out_dir = Path(out_dir)

    calib    = load_calib()
    L1 = calib.get(f"{arm}_upper_arm", 0.25)
    L2 = calib.get(f"{arm}_forearm",   0.22)
    is_right = (arm == "right")

    # arm에 맞는 연결선/라벨 생성
    connections = _make_connections(arm)
    labels      = _make_labels(arm)

    # 어깨 위치 (MuJoCo 좌표)
    SW_half = 0.18
    shoulder_world = np.array([0., -SW_half if is_right else SW_half, 0.])

    lms      = np.load(out_dir / "raw_landmarks.npy")   # (T,33,3)
    ts_arr   = np.load(out_dir / "timestamps.npy")
    lms_2d_p = out_dir / "raw_landmarks_2d.npy"
    lms_2d   = np.load(lms_2d_p) if lms_2d_p.exists() else None

    traj_path = out_dir / "trajectory.npy"
    traj = np.load(traj_path) if traj_path.exists() else None

    kf_meta = []
    if (out_dir / "keyframe_meta.json").exists():
        with open(out_dir / "keyframe_meta.json", encoding="utf-8") as f:
            kf_meta = json.load(f)
    kf_set = {int(kf["frame_idx"]) for kf in kf_meta}

    T_raw  = len(lms)
    T_traj = len(traj) if traj is not None else 0
    frame_files = sorted((out_dir / "frames").glob("*.jpg"))

    ncols = 3 if traj is not None else 2
    fig = plt.figure(figsize=(6*ncols, 6.5))
    fig.patch.set_facecolor("#12141a")
    BG = "#12141a"

    ax_vid = fig.add_subplot(1, ncols, 1)
    ax_vid.set_facecolor(BG); ax_vid.axis("off")
    ax_raw = fig.add_subplot(1, ncols, 2, projection="3d")
    ax_raw.set_facecolor(BG)
    ax_fk  = None
    if traj is not None:
        ax_fk = fig.add_subplot(1, ncols, 3, projection="3d")
        ax_fk.set_facecolor(BG)
    plt.tight_layout(pad=1.2)

    def _style3d(ax, title):
        ax.set_xlabel("X", color="#607d8b", fontsize=7)
        ax.set_ylabel("Y", color="#607d8b", fontsize=7)
        ax.set_zlabel("Z", color="#607d8b", fontsize=7)
        ax.tick_params(colors="#455a64", labelsize=6)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
        return ax.set_title(title, color="white", fontsize=10, pad=4)

    t_vid = ax_vid.set_title("", color="white", fontsize=10, pad=4)
    _style3d(ax_raw, "Raw Skeleton")
    t_fk = _style3d(ax_fk, f"FK Trajectory ({arm} arm)") if ax_fk else None

    pts_all = lms[lms.any(axis=(1,2))]
    if len(pts_all):
        c = pts_all.mean(axis=(0,1)); rng = max(pts_all.std(axis=(0,1)).max()*3, 0.4) + 0.15
    else:
        c = np.zeros(3); rng = 0.6
    ax_raw.set_xlim(c[0]-rng, c[0]+rng)
    ax_raw.set_ylim(c[1]-rng, c[1]+rng)
    ax_raw.set_zlim(c[2]-rng, c[2]+rng)

    frame0 = cv2.cvtColor(cv2.imread(str(frame_files[0])), cv2.COLOR_BGR2RGB)
    im_vid = ax_vid.imshow(frame0)

    sl = ax_raw.scatter([],[],[], c="#4fc3f7", s=22, depthshade=False)
    sr = ax_raw.scatter([],[],[], c="#ff8a65", s=22, depthshade=False)
    sc = ax_raw.scatter([],[],[], c="#90a4ae", s=12, depthshade=False)
    raw_lines   = [ax_raw.plot([],[],[], lw=1.2)[0] for _ in connections]
    kf_star     = ax_raw.scatter([],[],[], c="yellow", s=100, marker="*", depthshade=False)
    raw_labels  = {i: ax_raw.text(0,0,0, n, fontsize=5.5, color="white")
                   for i, n in labels.items()}

    arm_line = arm_scat = None
    arm_texts = []
    if ax_fk is not None:
        ax_fk.set_xlim(-0.6, 0.6); ax_fk.set_ylim(-0.6, 0.6); ax_fk.set_zlim(-0.5, 0.5)
        col = "#ff4081" if is_right else "#00e676"
        arm_line, = ax_fk.plot([],[],[], c=col, lw=2.5)
        arm_scat  = ax_fk.scatter([],[],[], c=col, s=45, depthshade=False)
        arm_texts = [ax_fk.text(0,0,0, n, fontsize=6, color=col) for n in FK_LABELS]

    state = {"paused": False}
    fig.canvas.mpl_connect("key_press_event",
        lambda e: state.update(paused=not state["paused"]) if e.key==" " else
                  plt.close(fig) if e.key=="q" else None)

    N = max(T_raw, T_traj)

    def update(fi):
        if state["paused"]: return

        vi = fi % len(frame_files)
        bgr = cv2.imread(str(frame_files[vi]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        is_kf = vi in kf_set
        if lms_2d is not None:
            rgb = _draw_overlay(rgb, lms_2d[vi % T_raw], connections, labels, arm)
        if is_kf:
            cv2.rectangle(rgb, (0,0), (rgb.shape[1]-1, rgb.shape[0]-1), (255,220,0), 5)
        im_vid.set_data(rgb)
        kf_lbl = next((f"  ★ {kf.get('label','')}" for kf in kf_meta
                        if int(kf["frame_idx"]) == vi), "")
        t_vid.set_text(f"Video  [{ts_arr[vi % T_raw]:.2f}s]{kf_lbl}")

        ri = fi % T_raw
        pts = lms[ri]
        px, py, pz = pts[:,0], -pts[:,2], -pts[:,1]
        active_i   = sorted(RIGHT_IDX if arm == "right" else LEFT_IDX)
        c_i        = [i for i in range(33) if i not in LEFT_IDX and i not in RIGHT_IDX]
        if arm == "right":
            sl._offsets3d = ([], [], [])             # 왼팔 숨기기
            sr._offsets3d = (px[active_i], py[active_i], pz[active_i])
        else:
            sl._offsets3d = (px[active_i], py[active_i], pz[active_i])
            sr._offsets3d = ([], [], [])             # 오른팔 숨기기
        sc._offsets3d = (px[c_i], py[c_i], pz[c_i])
        for line, (a, b) in zip(raw_lines, connections):
            col = "#4fc3f7" if (a in LEFT_IDX or b in LEFT_IDX) else \
                  "#ff8a65" if (a in RIGHT_IDX or b in RIGHT_IDX) else "#78909c"
            line.set_data_3d([px[a],px[b]], [py[a],py[b]], [pz[a],pz[b]]); line.set_color(col)
        kf_star._offsets3d = (px[[11,12,15,16]], py[[11,12,15,16]], pz[[11,12,15,16]]) \
                             if is_kf else ([],[],[])
        for idx, txt in raw_labels.items():
            txt.set_position_3d((px[idx]+0.01, py[idx]+0.01, pz[idx]+0.01))

        if traj is not None and ax_fk is not None:
            ti = fi % T_traj
            arm_pts = fk_arm(traj[ti], shoulder_world, L1, L2, L_HAND, is_right=is_right)
            arm_line.set_data_3d(arm_pts[:,0], arm_pts[:,1], arm_pts[:,2])
            arm_scat._offsets3d = (arm_pts[:,0], arm_pts[:,1], arm_pts[:,2])
            for j, txt in enumerate(arm_texts):
                txt.set_position_3d((arm_pts[j,0]+0.02, arm_pts[j,1], arm_pts[j,2]))
            t_fk.set_text(f"FK  {ti}/{T_traj-1}")

    interval_ms = max(8, int(1000/30/speed))
    ani = animation.FuncAnimation(fig, update, frames=N, interval=interval_ms, blit=False)

    if save_path:
        print(f"저장 중: {save_path}")
        ani.save(save_path, writer="ffmpeg", fps=int(30*speed), dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out",   required=True)
    p.add_argument("--arm",   default="right", choices=["right", "left"])
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--save",  default=None)
    args = p.parse_args()
    visualize(Path(args.out), args.arm, args.speed, args.save)
