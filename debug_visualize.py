"""
debug_visualize.py  --  Pipeline debug visualization (3 methods)

Method 1: Joint angle timeline  (raw / keyframes / trajectory + clamp limits)
Method 2: 3D wrist trajectory   (MediaPipe raw / keyframe positions / trajectory FK)
Method 3: Pipeline skeleton animation  (original video | Raw FK | Trajectory FK)

Usage:
  python debug_visualize.py --out output/my_session --arm right
  python debug_visualize.py --out output/my_session --arm right --method 1
  python debug_visualize.py --out output/my_session --arm right --method 3

Keys (method 3):
  Space : pause / resume
  q     : quit
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import sys
SA_DIR = Path(__file__).parent
V2_DIR = SA_DIR.parent / "human_demo_v2"
V1_DIR = SA_DIR.parent / "human_demo"
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(SA_DIR))

from human_ik import fk_arm, load_calib

# ── constants ─────────────────────────────────────────────────
JOINT_NAMES  = ["pan", "lift", "elbow", "wflx", "wrol", "grip"]
JOINT_COLORS = ["#4fc3f7", "#ff8a65", "#a5d6a7", "#f48fb1", "#ce93d8", "#ffcc80"]
_LO = np.array([-1.8, -2.0, 0.0, 0.0, -1.5, 0.0])
_HI = np.array([ 1.8,  1.8, 2.5, 1.5,  1.5, 1.57])
BG  = "#0d1117"
BG2 = "#12141a"


def _style3d(ax, title):
    ax.set_facecolor(BG2)
    ax.set_xlabel("X", color="#607d8b", fontsize=7)
    ax.set_ylabel("Y", color="#607d8b", fontsize=7)
    ax.set_zlabel("Z", color="#607d8b", fontsize=7)
    ax.tick_params(colors="#455a64", labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.set_title(title, color="white", fontsize=9, pad=4)


def _shoulder_world(arm: str):
    is_right = (arm == "right")
    return np.array([0., -0.18 if is_right else 0.18, 0.20])


# ══════════════════════════════════════════════════════════════
# Method 1: Joint Angle Timeline
# ══════════════════════════════════════════════════════════════

def plot_joint_timeline(out_dir: Path, arm: str = "right"):
    """
    Overlay raw_poses / keyframes / trajectory on a shared time axis.
    Red dashed lines = clamp limits.  Gold vertical lines = keyframe timestamps.
    If 'lift' does not touch the 1.8 limit, clamping is no longer an issue.
    """
    raw_poses  = np.load(out_dir / "raw_poses.npy")   # (T, 6)
    timestamps = np.load(out_dir / "timestamps.npy")  # (T,)
    trajectory = np.load(out_dir / "trajectory.npy")  # (T2, 6)
    keyframes  = np.load(out_dir / "keyframes.npy")   # (K, 6)

    with open(out_dir / "keyframe_meta.json", encoding="utf-8") as f:
        kf_meta = json.load(f)
    with open(out_dir / "trajectory_meta.json", encoding="utf-8") as f:
        traj_fps = json.load(f).get("fps", 30.0)

    traj_ts = np.arange(len(trajectory)) / traj_fps

    fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"[Method 1]  Joint Angle Timeline  [{arm} arm]\n"
        "dim=raw  bright=trajectory  gold=keyframe  red-dashed=clamp limit",
        color="white", fontsize=11)

    for j, (ax, name, col) in enumerate(zip(axes, JOINT_NAMES, JOINT_COLORS)):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2d36")

        ax.plot(timestamps, raw_poses[:, j],
                color=col, alpha=0.28, lw=0.8, label="raw")
        ax.plot(traj_ts, trajectory[:, j],
                color=col, alpha=0.95, lw=1.4, label="trajectory")

        for ki, km in enumerate(kf_meta):
            kf_ts = float(km.get("timestamp", 0))
            ax.axvline(kf_ts, color="#ffd700", lw=0.8, alpha=0.55)
            if ki < len(keyframes):
                ax.plot(kf_ts, keyframes[ki, j], 'o',
                        color="#ffd700", markersize=5, zorder=5)

        ax.axhline(_LO[j], color="#ff5555", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(_HI[j], color="#ff5555", lw=0.8, ls="--", alpha=0.6)

        ax.set_ylabel(name, color=col, fontsize=8)
        ax.tick_params(colors="#607d8b", labelsize=6)
        if j == 0:
            ax.legend(fontsize=6, loc="upper right",
                      facecolor="#1a1d24", labelcolor="white")
        if j == 5:
            ax.set_xlabel("time (s)", color="#607d8b", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = out_dir / "debug_1_joint_timeline.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=BG)
    print(f"  Saved: {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# Method 2: 3D Wrist Trajectory Comparison
# ══════════════════════════════════════════════════════════════

def plot_wrist_trajectory_3d(out_dir: Path, arm: str = "right"):
    """
    Left panel:  MediaPipe world-space wrist positions (shoulder-relative)
      - gray dots : all raw frames
      - gold stars: keyframes
    Right panel: MuJoCo FK wrist positions from final trajectory
      - plasma colormap: time -> color

    Check: wrist should rise along MP_y (left) and MuJoCo_z (right)
    when the arm is raised.
    """
    calib = load_calib()
    L1 = calib.get(f"{arm}_upper_arm", 0.25)
    L2 = calib.get(f"{arm}_forearm",   0.22)
    is_right = (arm == "right")
    sh_world = _shoulder_world(arm)

    WR_I = 16 if arm == "right" else 15
    SH_I = 12 if arm == "right" else 11

    fig = plt.figure(figsize=(14, 6.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"[Method 2]  3D Wrist Trajectory Comparison  [{arm} arm]\n"
        "Left: MediaPipe world (shoulder-relative)  /  Right: MuJoCo FK (trajectory)",
        color="white", fontsize=11)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    _style3d(ax1, "MediaPipe world\n(shoulder-relative,  y=up)")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _style3d(ax2, "MuJoCo FK wrist\n(absolute,  z=up)")

    # ── left: MediaPipe raw + keyframe wrists ────────────────
    lms_path = out_dir / "raw_landmarks.npy"
    if lms_path.exists():
        lms   = np.load(lms_path)
        valid = ~np.all(lms[:, SH_I] == 0, axis=1)
        wr_raw = lms[valid, WR_I] - lms[valid, SH_I]
        step = max(1, len(wr_raw) // 400)
        ax1.scatter(wr_raw[::step, 0], -wr_raw[::step, 2], wr_raw[::step, 1],
                    c="#607d8b", s=5, alpha=0.35, label="raw (all frames)")

    kf_pos_path = out_dir / "keyframes_pos.npy"
    if kf_pos_path.exists():
        kf_pos    = np.load(kf_pos_path)
        kf_wr_rel = kf_pos[:, 6:9] - kf_pos[:, 0:3]
        ax1.scatter(kf_wr_rel[:, 0], -kf_wr_rel[:, 2], kf_wr_rel[:, 1],
                    c="#ffd700", s=90, marker="*", zorder=6, label="keyframe")
        for ki, r in enumerate(kf_wr_rel):
            ax1.text(r[0], -r[2], r[1], f" KF{ki}", color="#ffd700", fontsize=7)

    ax1.set_xlabel("MP_x (right->)", color="white", fontsize=7)
    ax1.set_ylabel("MP_z (->cam)",   color="white", fontsize=7)
    ax1.set_zlabel("MP_y (up)",      color="white", fontsize=7)
    ax1.legend(fontsize=7, facecolor="#1a1d24", labelcolor="white")

    # ── right: trajectory FK wrist ───────────────────────────
    traj = np.load(out_dir / "trajectory.npy")
    traj_wrists = np.array([
        fk_arm(q, sh_world, L1, L2, is_right=is_right)[2]
        for q in traj
    ])

    step = max(1, len(traj_wrists) // 600)
    sc = ax2.scatter(traj_wrists[::step, 0], traj_wrists[::step, 1],
                     traj_wrists[::step, 2],
                     c=np.arange(len(traj_wrists[::step])),
                     cmap="plasma", s=7, alpha=0.8)
    ax2.plot(traj_wrists[:, 0], traj_wrists[:, 1], traj_wrists[:, 2],
             color="#ff4081", lw=1.0, alpha=0.45)
    ax2.scatter(*sh_world, c="white", s=70, marker="^", zorder=6, depthshade=False)
    ax2.text(*sh_world, "  Shoulder", color="white", fontsize=7)
    plt.colorbar(sc, ax=ax2, label="time ->", shrink=0.5, pad=0.1)

    ax2.set_xlabel("MuJoCo x (fwd)",  color="white", fontsize=7)
    ax2.set_ylabel("MuJoCo y (side)", color="white", fontsize=7)
    ax2.set_zlabel("MuJoCo z (up)",   color="white", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = out_dir / "debug_2_wrist_trajectory.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=BG)
    print(f"  Saved: {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# Method 3: Pipeline Skeleton Panel Animation
# ══════════════════════════════════════════════════════════════

def animate_pipeline_panels(out_dir: Path, arm: str = "right", speed: float = 1.0):
    """
    Synchronized 3-panel animation:
      [Original video + 2D skeleton]  |  [Raw Poses FK]  |  [Trajectory FK]

    If Raw FK matches the video but Trajectory FK does not,
    the issue is in the trajectory optimizer.
    If Raw FK is already wrong, the issue is in the IK.
    Space=pause  q=quit
    """
    calib = load_calib()
    L1 = calib.get(f"{arm}_upper_arm", 0.25)
    L2 = calib.get(f"{arm}_forearm",   0.22)
    is_right = (arm == "right")
    sh_world = _shoulder_world(arm)

    raw_poses  = np.load(out_dir / "raw_poses.npy")
    trajectory = np.load(out_dir / "trajectory.npy")
    timestamps = np.load(out_dir / "timestamps.npy")

    lms_2d = None
    p2d = out_dir / "raw_landmarks_2d.npy"
    if p2d.exists():
        lms_2d = np.load(p2d)

    with open(out_dir / "keyframe_meta.json", encoding="utf-8") as f:
        kf_meta = json.load(f)
    kf_set = {int(k["frame_idx"]) for k in kf_meta}

    frame_files = sorted((out_dir / "frames").glob("*.jpg"))
    T_raw  = len(raw_poses)
    T_traj = len(trajectory)
    N      = max(T_raw, T_traj)

    SH_I = 12 if arm == "right" else 11
    EL_I = 14 if arm == "right" else 13
    WR_I = 16 if arm == "right" else 15
    CONNS = [(SH_I, EL_I), (EL_I, WR_I), (11, 12)]

    # ── figure setup ─────────────────────────────────────────
    fig = plt.figure(figsize=(17, 6.5))
    fig.patch.set_facecolor(BG)

    ax_vid  = fig.add_subplot(1, 3, 1)
    ax_vid.set_facecolor(BG); ax_vid.axis("off")

    ax_raw  = fig.add_subplot(1, 3, 2, projection="3d")
    ax_traj = fig.add_subplot(1, 3, 3, projection="3d")
    _style3d(ax_raw,  "[Raw Poses FK]\nMediaPipe -> IK result")
    _style3d(ax_traj, "[Trajectory FK]\nFinal path")

    LIM = 0.55
    for ax in (ax_raw, ax_traj):
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_zlim(-LIM * 0.5, LIM * 1.2)

    frame0 = cv2.cvtColor(cv2.imread(str(frame_files[0])), cv2.COLOR_BGR2RGB)
    im_vid = ax_vid.imshow(frame0)
    t_vid  = ax_vid.set_title("", color="white", fontsize=9, pad=3)

    col_raw  = "#4fc3f7"
    col_traj = "#ff4081" if is_right else "#00e676"

    def _init_arm(ax, color):
        line, = ax.plot([], [], [], c=color, lw=2.8)
        scat  = ax.scatter([], [], [], c=color, s=35, depthshade=False)
        ax.scatter(*sh_world, c="white", s=60, marker="^",
                   depthshade=False, zorder=6)
        return line, scat

    raw_line,  raw_scat  = _init_arm(ax_raw,  col_raw)
    traj_line, traj_scat = _init_arm(ax_traj, col_traj)

    for ki, km in enumerate(kf_meta):
        for ax in (ax_raw, ax_traj):
            ax.text2D(0.02, 0.98 - ki * 0.07,
                      f"* {km.get('label','')[:12]}",
                      transform=ax.transAxes,
                      color="#ffd700", fontsize=6, va="top")

    state = {"paused": False}
    fig.canvas.mpl_connect("key_press_event",
        lambda e: state.update(paused=not state["paused"]) if e.key == " " else
                  plt.close(fig) if e.key == "q" else None)

    def _update_arm(line, scat, pts4):
        line.set_data_3d(pts4[:, 0], pts4[:, 1], pts4[:, 2])
        scat._offsets3d = (pts4[:, 0], pts4[:, 1], pts4[:, 2])

    def update(fi):
        if state["paused"]:
            return

        # panel 1: original video
        vi  = fi % len(frame_files)
        bgr = cv2.imread(str(frame_files[vi]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if lms_2d is not None:
            lm = lms_2d[vi % T_raw]
            H, W = rgb.shape[:2]
            for a, b in CONNS:
                pa = (int(lm[a, 0]*W), int(lm[a, 1]*H))
                pb = (int(lm[b, 0]*W), int(lm[b, 1]*H))
                if pa != (0, 0) and pb != (0, 0):
                    cv2.line(rgb, pa, pb, (80, 200, 80), 2, cv2.LINE_AA)
            for idx in (SH_I, EL_I, WR_I):
                px2 = int(lm[idx, 0]*W); py2 = int(lm[idx, 1]*H)
                if px2 > 0 or py2 > 0:
                    cv2.circle(rgb, (px2, py2), 6, (255, 180, 0), -1, cv2.LINE_AA)
        is_kf = vi in kf_set
        if is_kf:
            cv2.rectangle(rgb, (0, 0),
                          (rgb.shape[1]-1, rgb.shape[0]-1), (255, 220, 0), 5)
        im_vid.set_data(rgb)
        t_vid.set_text(
            f"Video  [{timestamps[vi % T_raw]:.2f}s]" + ("  * KF" if is_kf else ""))

        # panel 2: raw poses FK
        ri   = fi % T_raw
        rpts = fk_arm(raw_poses[ri], sh_world, L1, L2, is_right=is_right)
        _update_arm(raw_line, raw_scat, rpts)

        # panel 3: trajectory FK
        ti   = fi % T_traj
        tpts = fk_arm(trajectory[ti], sh_world, L1, L2, is_right=is_right)
        _update_arm(traj_line, traj_scat, tpts)

    interval_ms = max(8, int(1000 / 30 / speed))
    ani = animation.FuncAnimation(
        fig, update, frames=N, interval=interval_ms, blit=False)

    plt.suptitle(
        f"[Method 3]  Pipeline Step Comparison  [{arm} arm]\n"
        "Space=pause  q=quit  |  gold border = keyframe",
        color="white", fontsize=10)
    plt.tight_layout(pad=0.8)
    plt.show()


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Pipeline debug visualization (3 methods)")
    p.add_argument("--out",    required=True, help="Pipeline output folder")
    p.add_argument("--arm",    default="right", choices=["right", "left"])
    p.add_argument("--speed",  type=float, default=1.0, help="Playback speed (method 3)")
    p.add_argument("--method", type=int, default=0,
                   help="0=all  1=timeline  2=wrist-traj  3=animation")
    args = p.parse_args()
    out_dir = Path(args.out)

    required = ["raw_poses.npy", "timestamps.npy", "trajectory.npy",
                "keyframes.npy", "keyframe_meta.json", "trajectory_meta.json"]
    missing = [f for f in required if not (out_dir / f).exists()]
    if missing:
        print(f"Error: missing files: {missing}")
        print(f"  Run the pipeline first: python pipeline.py --video ... --out {out_dir}")
        return

    if args.method in (0, 1):
        print("\n[Method 1] Joint angle timeline...")
        plot_joint_timeline(out_dir, args.arm)

    if args.method in (0, 2):
        print("\n[Method 2] 3D wrist trajectory...")
        plot_wrist_trajectory_3d(out_dir, args.arm)

    if args.method in (0, 3):
        print("\n[Method 3] Pipeline skeleton animation...")
        animate_pipeline_panels(out_dir, args.arm, args.speed)


if __name__ == "__main__":
    main()
