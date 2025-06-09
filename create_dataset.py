import cv2
import time
import os
import random
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# --- 유저 지정 변수 ---
LABEL_NAME      = 1                       # 이 포즈의 클래스 ID
TRAIN_RATIO     = 0.8                     # train/test 분할 비율
CONF_THRESH     = 0.3                     # low-confidence 키포인트 컷오프
UPPER_BODY_IDX  = list(range(0,11))       # 상반신 11개 kpt 인덱스
FLIP_IDX        = [0,2,1,4,3,6,5,8,7,10,9] # 좌우 대칭 시 keypoint 재배치
OUTPUT_DIR      = "dataset_ad"  # 결과 저장 디렉토리

# 증강 파라미터
NOISE_STD       = 0.02
ROT_ANGLES  = list(range(-45, 50, 5))
# → [-45, -40, …, 0, …, 40, 45]

# Shift: -0.1부터 +0.1까지 9단계(0.025 간격)로
SHIFT_VALUES = np.linspace(-0.1, 0.1, 5)
# → [-0.10, -0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.10]

# (dx,dy) 페어 생성, (0,0)은 원본에 이미 포함돼 있으니 제외
SHIFTS = [
    (dx, dy)
    for dx in SHIFT_VALUES
    for dy in SHIFT_VALUES
    if not (dx == 0 and dy == 0)
]

# --- augmentation functions ---
def augment_noise(kp):
    return [np.clip(x + np.random.normal(0, NOISE_STD), 0, 1) for x in kp]

def augment_flip(kp):
    xs = kp[0::2]; ys = kp[1::2]
    xs_flipped = [1 - x for x in xs]
    xs_r = [xs_flipped[i] for i in FLIP_IDX]
    ys_r = [ys[i]         for i in FLIP_IDX]
    out = []
    for x,y in zip(xs_r, ys_r): out += [x,y]
    return out

def augment_rotate(kp, angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    out=[]
    for i in range(0, len(kp), 2):
        x, y = kp[i] - 0.5, kp[i+1] - 0.5
        xr, yr = c*x - s*y + 0.5, s*x + c*y + 0.5
        out += [np.clip(xr, 0, 1), np.clip(yr, 0, 1)]
    return out

def augment_shift(kp, dx, dy):
    out=[]
    for i in range(0, len(kp), 2):
        x = np.clip(kp[i]   + dx, 0, 1)
        y = np.clip(kp[i+1] + dy, 0, 1)
        out += [x, y]
    return out

# --- 초기화 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam", 1280, 720)

model = YOLO("yolo11m-pose.pt")

capturing = False
t0        = 0

# keypoint 벡터만 모을 리스트
raw_kp_vectors = []

print("스페이스: 3초 뒤 keypoint 캡처 • q: 종료")

while True:
    ret, frame = cap.read()
    if not ret: break

    res = model(frame, imgsz=640)[0]
    vis = res.plot() if len(res) > 0 else frame

    # capture countdown
    if capturing:
        elapsed = time.time() - t0
        if elapsed < 3:
            cv2.putText(vis, f"Capturing in {3-int(elapsed)}...",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        else:
            if len(res.boxes) and res.keypoints is not None:
                h, w = frame.shape[:2]
                # 첫 번째 인스턴스만
                xy   = res.keypoints.xy.cpu().numpy()[0]   # (17,2)
                conf = res.keypoints.conf.cpu().numpy()[0] # (17,)
                kp_vec = []
                for idx in UPPER_BODY_IDX:
                    x_px, y_px = xy[idx]
                    if conf[idx] < CONF_THRESH:
                        kp_vec += [0.0, 0.0]
                    else:
                        kp_vec += [float(x_px/w), float(y_px/h)]
                raw_kp_vectors.append(kp_vec)
                print(f"Captured: {len(raw_kp_vectors)} vectors")
            capturing = False

    cv2.imshow("Webcam", vis)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        capturing = True
        t0         = time.time()
    elif k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# --- augmentation & split ---
all_samples = []
for vec in raw_kp_vectors:
    all_samples.append(vec)
    all_samples.append( augment_noise(vec) )
    all_samples.append( augment_flip(vec) )
    for ang in ROT_ANGLES:
        all_samples.append(augment_rotate(vec, ang))
    for dx, dy in SHIFTS:
        all_samples.append(augment_shift(vec, dx, dy))

# train/test split
train, test = train_test_split(all_samples, train_size=TRAIN_RATIO, random_state=42)

# --- DataFrame 생성 & 저장 ---
col_names = []
for i in range(len(UPPER_BODY_IDX)):
    col_names += [f"kp{i}_x", f"kp{i}_y"]
col_names += ["label"]

df_train = pd.DataFrame(train, columns=col_names[:-1])
df_train["label"] = LABEL_NAME
df_test  = pd.DataFrame(test,  columns=col_names[:-1])
df_test["label"]  = LABEL_NAME

train_path = os.path.join(OUTPUT_DIR, f"train_{LABEL_NAME}.csv")
test_path  = os.path.join(OUTPUT_DIR, f"test_{LABEL_NAME}.csv")

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path,   index=False)

print(f"✅ 저장 완료:\n - {train_path} ({len(df_train)})\n - {test_path} ({len(df_test)})")
