"""
calibrate_arm.py  —  팔 길이 측정 → 단일 팔 MuJoCo 시각화

사용법:
  python calibrate_arm.py --video calib_video.mp4 [--arm right]
  python calibrate_arm.py --view-only [--arm right]
  python calibrate_arm.py --view-only --trajectory output/my_session/trajectory.npy
"""

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

V1_DIR = Path(__file__).parent.parent / "human_demo"
sys.path.insert(0, str(V1_DIR))
from pose_retarget import world_landmarks_to_np, LM


def measure_arm_lengths(video_path: str, arm: str = "right") -> dict:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = str(V1_DIR / "pose_landmarker_full.task")
    landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
    )

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    upper_vals, lower_vals = [], []
    debug_samples = []   # (frame_bgr, lm2d_norm, world3d) — 최대 9개
    sample_every  = max(1, total // 9)
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        ts = frame_idx / orig_fps
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, int(ts * 1000))

        if result.pose_world_landmarks and result.pose_landmarks:
            pts = world_landmarks_to_np(result.pose_world_landmarks[0])
            upper = np.linalg.norm(pts[LM[f"{arm}_elbow"]]   - pts[LM[f"{arm}_shoulder"]])
            lower = np.linalg.norm(pts[LM[f"{arm}_wrist"]]   - pts[LM[f"{arm}_elbow"]])
            if upper > 0.05 and lower > 0.05:
                upper_vals.append(upper)
                lower_vals.append(lower)
                # 균등 간격으로 디버그 샘플 수집
                if len(debug_samples) < 9 and frame_idx % sample_every == 0:
                    lm2d = np.array([[lm.x, lm.y]
                                     for lm in result.pose_landmarks[0]], dtype=np.float32)
                    debug_samples.append((frame_bgr.copy(), lm2d, pts.copy()))

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  [{frame_idx}/{total}]", end="\r")

    cap.release()
    landmarker.close()
    print()

    def _robust(vals):
        if not vals:
            return 0.25, 0.0
        arr = np.array(vals)
        thresh = np.percentile(arr, 85)
        sel = arr[arr >= thresh * 0.9]
        return float(sel.mean()), float(sel.std())

    u_mean, u_std = _robust(upper_vals)
    l_mean, l_std = _robust(lower_vals)

    return {
        f"{arm}_upper_arm":     u_mean, f"{arm}_upper_arm_std": u_std,
        f"{arm}_forearm":       l_mean, f"{arm}_forearm_std":   l_std,
        "arm":                  arm,
        "measured_frames":      frame_idx,
        "valid_frames":         len(upper_vals),
        "_debug_samples":       debug_samples,   # JSON 저장 전에 pop 할 것
    }


def print_measurements(calib: dict):
    arm = calib.get("arm", "right")
    print(f"\n{'='*50}")
    print(f"  팔 길이 측정 결과 ({arm} arm)")
    print(f"{'='*50}")
    print(f"  상완 (어깨→팔꿈치): {calib[f'{arm}_upper_arm']:.3f}m ±{calib.get(f'{arm}_upper_arm_std',0):.3f}m")
    print(f"  전완 (팔꿈치→손목): {calib[f'{arm}_forearm']:.3f}m ±{calib.get(f'{arm}_forearm_std',0):.3f}m")
    reach = calib[f"{arm}_upper_arm"] + calib[f"{arm}_forearm"]
    print(f"  최대 뻗기: {reach:.3f}m")
    print(f"  유효 프레임: {calib.get('valid_frames',0)} / {calib.get('measured_frames',0)}")
    print(f"{'='*50}")


def show_measurement_debug(calib: dict, save_path: str = "calib_debug.png"):
    """
    캘리브레이션 측정 디버그 시각화.

    각 샘플 프레임에 표시:
      ● 어깨 / 팔꿈치 / 손목  → 색상 원
      — 연결선 + 3D 거리 레이블 (m)
      좌표 텍스트: MediaPipe world_landmarks (x, y, z) in meters
        x = 오른쪽, y = 위, z = 카메라 방향 (음수 = 카메라 가까운 쪽)

    좌표 기준:
      원점 = 좌우 hip 중간점 (MediaPipe 고정 기준)
      단위 ≈ 미터 (MediaPipe가 몸통 비율로 스케일 추정)
      → 절대 위치는 부정확, 관절 간 상대 거리는 신뢰 가능
    """
    import matplotlib
    matplotlib.use("TkAgg")  # 또는 "Qt5Agg"
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    arm     = calib.get("arm", "right")
    samples = calib.get("_debug_samples", [])
    if not samples:
        print("  [Debug] 샘플 없음 (캘리브레이션 영상에서 팔이 감지되지 않음)")
        return

    sh_i = LM[f"{arm}_shoulder"]
    el_i = LM[f"{arm}_elbow"]
    wr_i = LM[f"{arm}_wrist"]

    n    = len(samples)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    fig.patch.set_facecolor("#0d1117")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, (frame_bgr, lm2d, world3d) in enumerate(samples):
        ax = axes_flat[i]
        H, W = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb); ax.axis("off")

        joints = {
            "Shoulder": (sh_i, "#ff4444"),
            "Elbow":    (el_i, "#ffaa00"),
            "Wrist":    (wr_i, "#44ddff"),
        }

        px = {}
        for name, (idx, color) in joints.items():
            x2d = int(lm2d[idx, 0] * W)
            y2d = int(lm2d[idx, 1] * H)
            px[name] = (x2d, y2d)
            # 원
            circ = plt.Circle((x2d, y2d), max(W//80, 6), color=color, zorder=5)
            ax.add_patch(circ)
            # 좌표 텍스트 (MediaPipe world: x,y,z in meters)
            p3 = world3d[idx]
            ax.text(x2d + 8, y2d - 8,
                    f"{name}\n({p3[0]:+.3f}, {p3[1]:+.3f}, {p3[2]:+.3f})m",
                    fontsize=5.5, color=color, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#0d1117", alpha=0.75, ec="none"))

        # 연결선 + 거리 레이블
        segs = [
            ("Shoulder", "Elbow", "#ff8844",
             float(np.linalg.norm(world3d[el_i] - world3d[sh_i]))),
            ("Elbow",    "Wrist", "#44aaff",
             float(np.linalg.norm(world3d[wr_i] - world3d[el_i]))),
        ]
        for pa_n, pb_n, col, dist in segs:
            pa, pb = px[pa_n], px[pb_n]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], "-", color=col, lw=2.5, zorder=4)
            mx, my = (pa[0]+pb[0])//2, (pa[1]+pb[1])//2
            ax.text(mx, my - 10, f"{dist:.3f} m",
                    fontsize=8, color=col, fontweight="bold", ha="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117", alpha=0.8, ec="none"))

        u = float(np.linalg.norm(world3d[el_i] - world3d[sh_i]))
        l = float(np.linalg.norm(world3d[wr_i] - world3d[el_i]))
        ax.set_title(f"upper={u:.3f}m  forearm={l:.3f}m",
                     color="white", fontsize=8, pad=3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    u_final = calib[f"{arm}_upper_arm"]
    l_final = calib[f"{arm}_forearm"]
    plt.suptitle(
        f"Calibration Debug  [{arm} arm]\n"
        f"Final: upper={u_final:.3f}m  forearm={l_final:.3f}m  (85th-pct mean)\n"
        f"Coords: origin=hip-mid  x=right  y=up  z=toward-cam(neg=far)",
        color="white", fontsize=10, y=1.01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  [Debug] 저장: {save_path}")
    plt.show()


def generate_mujoco_xml(calib: dict) -> str:
    arm = calib.get("arm", "right")
    L1  = calib[f"{arm}_upper_arm"]
    L2  = calib[f"{arm}_forearm"]
    HL  = 0.07
    FL  = 0.045
    SH  = 0.20  # 어깨 높이 오프셋
    SW_half = 0.18  # 어깨 절반 너비

    is_right = (arm == "right")
    sign = -1.0 if is_right else 1.0   # 오른팔: -Y, 왼팔: +Y
    sy = sign * SW_half                 # 어깨 Y 위치

    if is_right:
        # 오른팔: pan=+Z, elbow=+Z, T-pose=-Y
        pan_axis    = "0 0 1"
        elbow_axis  = "0 0 1"
        arm_dir     = f"0 {-L1:.4f} 0"
        farm_dir    = f"0 {-L2:.4f} 0"
        hand_dir    = f"0 {-HL:.4f} 0"
        fing_a      = f"0 {-FL:.4f}  0.018"
        fing_b      = f"0 {-FL:.4f} -0.018"
        farm_pos    = f"0 {-L1:.4f} 0"
        hand_pos    = f"0 {-L2:.4f} 0"
        fing_pos    = f"0 {-HL:.4f} 0"
        rgba_sh     = "0.95 0.44 0.32 1"
        rgba_u      = "0.95 0.44 0.32 0.9"
        rgba_uj     = "0.95 0.44 0.32 1"
        rgba_f      = "1.00 0.56 0.44 0.9"
        rgba_fj     = "1.00 0.56 0.44 1"
        rgba_h      = "1.00 0.66 0.55 1"
        rgba_fi     = "1.00 0.76 0.66 1"
    else:
        pan_axis    = "0 0 -1"
        elbow_axis  = "0 0 -1"
        arm_dir     = f"0 {L1:.4f} 0"
        farm_dir    = f"0 {L2:.4f} 0"
        hand_dir    = f"0 {HL:.4f} 0"
        fing_a      = f"0 {FL:.4f}  0.018"
        fing_b      = f"0 {FL:.4f} -0.018"
        farm_pos    = f"0 {L1:.4f} 0"
        hand_pos    = f"0 {L2:.4f} 0"
        fing_pos    = f"0 {HL:.4f} 0"
        rgba_sh     = "0.22 0.60 0.92 1"
        rgba_u      = "0.22 0.60 0.92 0.9"
        rgba_uj     = "0.22 0.60 0.92 1"
        rgba_f      = "0.32 0.72 1.00 0.9"
        rgba_fj     = "0.32 0.72 1.00 1"
        rgba_h      = "0.42 0.82 1.00 1"
        rgba_fi     = "0.55 0.90 1.00 1"

    prefix = "R" if is_right else "L"

    # 쇄골(collarbone): 몸통 중심 상단 → 어깨 구체까지 연결
    # 몸통 박스 Y 반너비=0.09, 어깨 Y=sy → 공백을 캡슐로 메움
    collarbone = f'<geom type="capsule" fromto="0 0 {SH:.4f} 0 {sy:.4f} {SH:.4f}" size="0.025" rgba="0.50 0.50 0.60 0.9"/>'

    return f"""<mujoco model="single_arm_{arm}">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.18 0.22 0.28" rgb2="0.05 0.05 0.07" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".65 .65 .65" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="2 1 3" dir="-1 -0.5 -1" diffuse="0.3 0.3 0.35"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid" contype="1" conaffinity="1"/>

    <body name="torso" pos="0 0 1.3">
      <geom type="box"    size="0.13 0.09 0.18" rgba="0.55 0.55 0.65 1"/>
      <geom type="sphere" pos="0 0 0.27"         size="0.10" rgba="0.75 0.70 0.65 1"/>
      <!-- 쇄골: 몸통→어깨 연결 -->
      {collarbone}

      <!-- {arm} arm -->
      <body name="{prefix}_smount" pos="0 {sy:.4f} {SH:.4f}">
        <geom type="sphere" size="0.040" rgba="{rgba_sh}"/>

        <body name="{prefix}_uarm" pos="0 0 0">
          <joint name="{prefix}_lift"  type="hinge" axis="1 0 0"      range="-2.0 1.8" damping="1.0"/>
          <joint name="{prefix}_pan"   type="hinge" axis="{pan_axis}"  range="-1.8 1.8" damping="1.0"/>
          <geom type="capsule" fromto="0 0 0 {arm_dir}"  size="0.022" rgba="{rgba_u}"/>
          <geom type="sphere"  pos="{farm_pos}"           size="0.028" rgba="{rgba_uj}"/>

          <body name="{prefix}_farm" pos="{farm_pos}">
            <joint name="{prefix}_elbow" type="hinge" axis="{elbow_axis}" range="0.0 2.5" damping="0.6"/>
            <geom type="capsule" fromto="0 0 0 {farm_dir}" size="0.017" rgba="{rgba_f}"/>
            <geom type="sphere"  pos="{hand_pos}"           size="0.022" rgba="{rgba_fj}"/>

            <body name="{prefix}_hand" pos="{hand_pos}">
              <joint name="{prefix}_wflx" type="hinge" axis="1 0 0" range="-1.5 1.5" damping="0.3"/>
              <joint name="{prefix}_wrol" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="0.3"/>
              <geom type="capsule" fromto="0 0 0 {hand_dir}" size="0.015" rgba="{rgba_h}"/>

              <body name="{prefix}_fingers" pos="{fing_pos}">
                <joint name="{prefix}_grip" type="hinge" axis="0 0 1" range="0 1.57" damping="0.2"/>
                <geom type="capsule" fromto="0 0  0.018 {fing_a}" size="0.006" rgba="{rgba_fi}"/>
                <geom type="capsule" fromto="0 0 -0.018 {fing_b}" size="0.006" rgba="{rgba_fi}"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- 순서: pan, lift, elbow, wflx, wrol, grip -->
    <motor joint="{prefix}_pan"   gear="1" ctrllimited="true" ctrlrange="-1.8 1.8"/>
    <motor joint="{prefix}_lift"  gear="1" ctrllimited="true" ctrlrange="-2.0 1.8"/>
    <motor joint="{prefix}_elbow" gear="1" ctrllimited="true" ctrlrange="0.0 2.5"/>
    <motor joint="{prefix}_wflx"  gear="1" ctrllimited="true" ctrlrange="0.0 1.5"/>
    <motor joint="{prefix}_wrol"  gear="1" ctrllimited="true" ctrlrange="-1.5 1.5"/>
    <motor joint="{prefix}_grip"  gear="1" ctrllimited="true" ctrlrange="0.0 1.57"/>
  </actuator>
</mujoco>"""


def launch_mujoco_viewer(xml_str: str, trajectory_path: str | None, arm: str = "right"):
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("mujoco 없음. pip install mujoco")
        return

    model = mujoco.MjModel.from_xml_string(xml_str)
    data  = mujoco.MjData(model)

    traj = None
    if trajectory_path and Path(trajectory_path).exists():
        traj = np.load(trajectory_path)  # (T, 6)
        print(f"  trajectory 재생: {traj.shape}")

    prefix = "R" if arm == "right" else "L"
    joint_names = [f"{prefix}_pan", f"{prefix}_lift", f"{prefix}_elbow",
                   f"{prefix}_wflx", f"{prefix}_wrol", f"{prefix}_grip"]
    neutral = [0.0, -0.3, 0.5, 0.0, 0.0, float(np.pi / 2)]
    jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]

    print("\nMuJoCo 뷰어 실행 중... (창 닫으면 종료)")
    frame_idx = [0]
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 135

        import time
        while viewer.is_running():
            if traj is not None:
                fi = frame_idx[0] % len(traj)
                q  = traj[fi]
                for ji, jid in enumerate(jnt_ids):
                    if jid >= 0:
                        data.qpos[model.jnt_qposadr[jid]] = float(q[ji])
                frame_idx[0] += 1
            else:
                for ji, jid in enumerate(jnt_ids):
                    if jid >= 0:
                        data.qpos[model.jnt_qposadr[jid]] = neutral[ji]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1/30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video",      default=None)
    p.add_argument("--arm",        default="right", choices=["right", "left"])
    p.add_argument("--out",        default="arm_calib.json")
    p.add_argument("--trajectory", default=None)
    p.add_argument("--no-viewer",  action="store_true")
    p.add_argument("--view-only",  action="store_true")
    args = p.parse_args()

    out_path = Path(args.out)

    if args.view_only:
        if not out_path.exists():
            print(f"오류: {out_path} 없음. 먼저 --video 로 측정하세요."); return
        with open(out_path, encoding="utf-8") as f:
            calib = json.load(f)
        arm = calib.get("arm", args.arm)
    else:
        if not args.video:
            p.error("--video 가 필요합니다 (또는 --view-only 사용)")
        if not Path(args.video).exists():
            p.error(f"비디오 파일이 없습니다: {args.video}\n"
                    f"  먼저 python record_video.py --out calib.mp4 으로 녹화하세요.")
        print(f"팔 길이 측정 중: {args.video}  [{args.arm} arm]")
        calib = measure_arm_lengths(args.video, args.arm)
        if calib["valid_frames"] == 0:
            print("경고: 유효 프레임이 0입니다. 영상에 팔이 잘 보이는지 확인하세요.")
            print("     기본값(0.25m / 0.22m)으로 계속합니다.")
        calib["arm"] = args.arm
        # 디버그 시각화 (JSON 저장 전에 pop)
        debug_samples = calib.pop("_debug_samples", [])
        calib["_debug_samples"] = debug_samples   # show_measurement_debug용으로 임시 복원
        show_measurement_debug(calib, save_path=str(out_path).replace(".json", "_debug.png"))
        calib.pop("_debug_samples", None)         # JSON에는 저장 안 함
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)
        print(f"저장: {out_path}")
        arm = args.arm

    print_measurements(calib)
    xml_str = generate_mujoco_xml(calib)
    xml_path = out_path.with_name(f"calib_robot_{arm}.xml")
    xml_path.write_text(xml_str, encoding="utf-8")
    print(f"MuJoCo XML 저장: {xml_path}")

    if not args.no_viewer:
        launch_mujoco_viewer(xml_str, args.trajectory, arm)


if __name__ == "__main__":
    main()
