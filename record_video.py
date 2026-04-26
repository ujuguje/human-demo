"""
record_video.py  —  웹캠 녹화 (미러 미리보기)

키:
  Space  : 녹화 시작 / 중지
  m      : 미러 모드 켜기/끄기 (저장은 항상 원본)
  q      : 종료 + 저장
"""

import argparse
from pathlib import Path

import cv2


def record(out_path: str, cam_id: int = 0, fps: float = 30.0):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise IOError(f"카메라 {cam_id}를 열 수 없습니다. --cam 옵션으로 다른 ID를 시도해보세요.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    recording = False
    mirror    = True   # 기본: 미러(좌우 반전) 미리보기
    frames_written = 0

    WIN = "Webcam  [Space:녹화  m:미러  q:저장+종료]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(W, 960), min(H, 540))

    print(f"웹캠 열림: {W}x{H} @ {fps}fps")
    print("  Space : 녹화 시작/중지")
    print("  m     : 미러 모드 켜기/끄기")
    print("  q     : 종료 + 저장")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 저장은 항상 원본(미러 없는) 프레임
        if recording:
            if writer is None:
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
            writer.write(frame)
            frames_written += 1

        # 미리보기용 디스플레이
        display = cv2.flip(frame, 1) if mirror else frame.copy()

        # UI 오버레이
        if recording:
            cv2.circle(display, (28, 28), 13, (0, 0, 220), -1)
            cv2.putText(display, f"REC  {frames_written/fps:.1f}s",
                        (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 2, cv2.LINE_AA)
        else:
            cv2.putText(display, "PREVIEW  [Space] 녹화 시작",
                        (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 230, 180), 2, cv2.LINE_AA)

        mirror_txt = "Mirror ON" if mirror else "Mirror OFF"
        cv2.putText(display, mirror_txt,
                    (W - 130, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            recording = not recording
            if recording:
                print("● 녹화 시작...")
            else:
                print(f"■ 일시정지 ({frames_written}프레임 / {frames_written/fps:.1f}s)")
        elif key == ord('m'):
            mirror = not mirror
            print(f"  미러 모드: {'ON' if mirror else 'OFF'}")
        elif key == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if frames_written > 0:
        print(f"\n저장 완료: {out_path}")
        print(f"  {frames_written}프레임  {frames_written/fps:.1f}초  {W}x{H}")
    else:
        print("녹화된 프레임 없음.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/recording/input.mp4")
    p.add_argument("--cam", type=int, default=0, help="카메라 ID (기본 0, 안 되면 1 시도)")
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()
    record(args.out, args.cam, args.fps)
