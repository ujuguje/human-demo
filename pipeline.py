"""
pipeline.py  —  human_demo_singlearm 메인 진입점

파이프라인 4단계:
  Stage 1  extract_poses.py      영상 → 단일 팔 포즈 (T, 6)
  Stage 2  vlm_analyzer.py       Gemini 영상 분석 → 키프레임/태스크 구조
  Stage 3  keyframe_processor.py VLM 결과 → 클린 키프레임 (K, 6)
  Stage 4  trajectory_optimizer  키프레임 → Minimum Jerk Trajectory (T2, 6)

사용법:
  export GEMINI_API_KEY="your_key_here"
  python pipeline.py --video my_video.mp4 --arm right

  # 특정 단계부터 재실행
  python pipeline.py --video my_video.mp4 --from_stage 3 --out output/my_session
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SA_DIR = Path(__file__).parent
V2_DIR = Path(__file__).parent.parent / "human_demo_v2"
V1_DIR = Path(__file__).parent.parent / "human_demo"
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(V2_DIR))   # vlm_analyzer 등 공용 모듈
sys.path.insert(0, str(SA_DIR))   # singlearm 우선 (v2보다 앞)


def show_keyframe_captures(out_dir: Path, analysis: dict, arm: str = "right"):
    """
    Stage 2 완료 후: Gemini 선택 키프레임 그리드.

    각 셀:
      - 원본 프레임 + 관절 연결선
      - 어깨(빨강) / 팔꿈치(주황) / 손목(하늘) 컬러 원 + 3D 좌표 텍스트
      - 관절 간 거리 라벨 (MediaPipe world 미터)
      - Gemini 라벨 / 손 상태 / 설명
    좌표계: origin=hip중점  x=오른쪽  y=위  z=카메라쪽(음수=멀리)
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    frames_dir  = out_dir / "frames"
    lms_2d_path = out_dir / "raw_landmarks_2d.npy"
    lms_3d_path = out_dir / "raw_landmarks.npy"
    lms_2d = np.load(lms_2d_path) if lms_2d_path.exists() else None
    lms_3d = np.load(lms_3d_path) if lms_3d_path.exists() else None

    SH_I = 12 if arm == "right" else 11
    EL_I = 14 if arm == "right" else 13
    WR_I = 16 if arm == "right" else 15

    vlm_kfs = sorted(analysis.get("keyframes", []),
                     key=lambda x: float(x.get("timestamp", 0)))
    n = len(vlm_kfs)
    if n == 0:
        print("  키프레임 없음 — 그리드 스킵")
        return

    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        print("  frames/ 없음 — 그리드 스킵")
        return

    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    fig.patch.set_facecolor("#0d1117")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    CONNS = [(12,14),(14,16),(11,12)] if arm == "right" else [(11,13),(13,15),(11,12)]
    JT    = [("Sh", SH_I, "#ff4444"),
             ("El", EL_I, "#ffaa00"),
             ("Wr", WR_I, "#44ddff")]

    for i, kf in enumerate(vlm_kfs):
        ax = axes_flat[i]
        fidx = int(kf.get("frame_idx", 0))
        fidx = min(fidx, len(frame_files) - 1)

        bgr = cv2.imread(str(frame_files[fidx]))
        if bgr is None:
            ax.axis("off"); continue
        H, W = bgr.shape[:2]
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 스켈레톤 연결선 (초록)
        if lms_2d is not None and fidx < len(lms_2d):
            lm = lms_2d[fidx]
            for a, b in CONNS:
                pa = (int(lm[a,0]*W), int(lm[a,1]*H))
                pb = (int(lm[b,0]*W), int(lm[b,1]*H))
                if pa != (0,0) and pb != (0,0):
                    cv2.line(rgb, pa, pb, (80, 200, 80), 2, cv2.LINE_AA)

        ax.imshow(rgb); ax.axis("off")

        # 3D 위치 오버레이
        pos_str = ""
        if lms_2d is not None and lms_3d is not None \
                and fidx < len(lms_2d) and fidx < len(lms_3d):
            lm2 = lms_2d[fidx]; pts = lms_3d[fidx]
            px = {}
            for name, idx, color in JT:
                x2d = lm2[idx, 0] * W
                y2d = lm2[idx, 1] * H
                p3  = pts[idx]
                px[name] = (x2d, y2d)
                ax.plot(x2d, y2d, 'o', color=color, markersize=9, zorder=5,
                        markeredgecolor="white", markeredgewidth=0.7)
                ax.text(x2d + 7, y2d - 7,
                        f"{name}({p3[0]:+.2f},{p3[1]:+.2f},{p3[2]:+.2f})",
                        fontsize=5, color=color, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.1",
                                  fc="#0d1117", alpha=0.80, ec="none"))

            # 거리 측정선
            segs = [("Sh","El", EL_I, SH_I, "#ff8844"),
                    ("El","Wr", WR_I, EL_I, "#44aaff")]
            for n0, n1, i1, i0, col in segs:
                p0, p1 = px[n0], px[n1]
                ax.plot([p0[0],p1[0]], [p0[1],p1[1]], '-',
                        color=col, lw=2, zorder=4, alpha=0.85)
                dist = float(np.linalg.norm(pts[i1] - pts[i0]))
                mx, my = (p0[0]+p1[0])/2, (p0[1]+p1[1])/2
                ax.text(mx, my - 10, f"{dist:.3f}m", fontsize=7.5, color=col,
                        fontweight="bold", ha="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="#0d1117", alpha=0.82, ec="none"))

            ua_m = float(np.linalg.norm(pts[EL_I] - pts[SH_I]))
            fa_m = float(np.linalg.norm(pts[WR_I] - pts[EL_I]))
            pos_str = f"ua={ua_m:.3f}m  fa={fa_m:.3f}m"

        # 골든 테두리
        for sp in ax.spines.values():
            sp.set_edgecolor("#ffd700"); sp.set_linewidth(2.5)

        ts       = kf.get("timestamp", 0)
        lbl      = kf.get("label", "?")
        imp      = kf.get("importance", "?")
        hand     = (kf.get("right_hand") or kf.get("left_hand") or {})
        hstate   = hand.get("state", "?")
        hcontact = hand.get("contact", "none")
        desc     = kf.get("description", "")[:55]
        title    = (f"[{ts:.1f}s] {lbl}  (imp={imp})\n"
                    f"hand: {hstate}/{hcontact}   {pos_str}\n"
                    f"{desc}")
        ax.set_title(title, color="white", fontsize=7, pad=3, loc="left")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle(
        f"Gemini Keyframes  [{arm} arm]  total={n}\n"
        "Coords: origin=hip-mid  x=right  y=up  z=toward-cam(neg=far)",
        color="white", fontsize=11)
    plt.tight_layout(pad=1.5)
    save_path = out_dir / "keyframe_captures.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Keyframe captures saved: {save_path}")
    plt.show()


def run_pipeline(args):
    out_dir = Path(args.out) if args.out else (
        Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  human_demo_singlearm 파이프라인  [{args.arm} arm]")
    print(f"  출력: {out_dir}")
    print(f"{'='*60}\n")

    from_stage = args.from_stage

    # ── Stage 1 ──────────────────────────────────────────────
    if from_stage <= 1:
        print("[ Stage 1 ] 포즈 추출 시작...")
        from extract_poses import extract_poses
        extract_poses(
            video_path=args.video,
            out_dir=out_dir,
            arm=args.arm,
            device=args.device,
            depth_skip=args.depth_skip,
        )
        print("[ Stage 1 ] 완료\n")
    else:
        print("[ Stage 1 ] 스킵\n")

    # ── Stage 2 ──────────────────────────────────────────────
    if from_stage <= 2:
        print("[ Stage 2 ] Gemini 영상 분석 시작...")
        from vlm_analyzer import analyze_video
        analysis = analyze_video(
            video_path=args.video,
            out_dir=out_dir,
            task_description=args.task,
            sample_fps=args.sample_fps,
        )
        print(f"[ Stage 2 ] 완료 — 키프레임 {len(analysis['keyframes'])}개\n")
        show_keyframe_captures(out_dir, analysis, arm=args.arm)
    else:
        print("[ Stage 2 ] 스킵 — analysis.json 로드...")
        with open(out_dir / "analysis.json", encoding="utf-8") as f:
            analysis = json.load(f)
        print(f"            키프레임 {len(analysis['keyframes'])}개\n")
        show_keyframe_captures(out_dir, analysis, arm=args.arm)

    # ── Stage 3 ──────────────────────────────────────────────
    if from_stage <= 3:
        print("[ Stage 3 ] 키프레임 처리 시작...")
        from keyframe_processor import process_keyframes
        keyframes, kf_meta = process_keyframes(
            out_dir=out_dir,
            analysis=analysis,
            arm=args.arm,
            smooth_window=args.smooth_window,
        )
        print(f"[ Stage 3 ] 완료 — 키프레임 {len(keyframes)}개\n")
    else:
        print("[ Stage 3 ] 스킵 — keyframes.npy 로드...")
        keyframes = np.load(out_dir / "keyframes.npy")
        with open(out_dir / "keyframe_meta.json", encoding="utf-8") as f:
            kf_meta = json.load(f)
        print(f"            키프레임 {len(keyframes)}개\n")

    # ── Stage 4 ──────────────────────────────────────────────
    print("[ Stage 4 ] Trajectory 최적화 시작...")
    from trajectory_optimizer import optimize_trajectory
    trajectory = optimize_trajectory(
        keyframes=keyframes,
        kf_meta=kf_meta,
        out_dir=out_dir,
        fps=args.fps,
        max_velocity=args.max_velocity,
        arm=args.arm,
    )
    print(f"[ Stage 4 ] 완료 — {len(trajectory)}프레임 ({len(trajectory)/args.fps:.1f}s)\n")

    print("="*60)
    print(f"  파이프라인 완료  [{args.arm} arm]")
    print(f"  키프레임  : {len(keyframes)}개")
    print(f"  Trajectory: {len(trajectory)}프레임 ({len(trajectory)/args.fps:.1f}s)")
    print(f"  저장 위치 : {out_dir}")
    print("="*60)
    print()
    print("다음 단계:")
    print(f"  # 시각화")
    print(f"  python visualize_skeleton.py --out {out_dir} --arm {args.arm}")
    print(f"  # MuJoCo 재생")
    print(f"  python calibrate_arm.py --view-only --arm {args.arm} --trajectory {out_dir}/trajectory.npy")


def main():
    p = argparse.ArgumentParser(description="human_demo_singlearm 파이프라인")
    p.add_argument("--video",         required=True)
    p.add_argument("--arm",           default="right", choices=["right", "left"])
    p.add_argument("--task",          default="")
    p.add_argument("--out",           default=None)
    p.add_argument("--from_stage",    type=int, default=1)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--depth_skip",    type=int, default=3)
    p.add_argument("--sample_fps",    type=float, default=2.0)
    p.add_argument("--smooth_window", type=int, default=7)
    p.add_argument("--fps",           type=float, default=30.0)
    p.add_argument("--max_velocity",  type=float, default=1.0)
    args = p.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
