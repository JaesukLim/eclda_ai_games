import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ---------------------------
# 설정
# ---------------------------
YOLO_WEIGHTS        = "./yolo11m-pose.pt"
CLS_MODEL_PATH      = "./pose_classifier.pth"
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESH         = 0.3   # keypoint 검출 신뢰도 컷오프
CLASS_CONF_THRESH   = 0.5   # 분류기 softmax 컷오프
UPPER_BODY_IDX      = list(range(0, 13))

ENERGY_TEMP        = 1.0        # energy score temperature T
ENERGY_THRESHOLD   = 5.0

FONT_PATH = "BMHANNAPro.ttf"
FONT_SIZE = 32
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

labels = ["군인", "하트", "복싱", "가수", "닭", "꽃", "배꼽인사", "양궁", "야구선수(타자)", "없음"]

def energy_score(logits, T=1.0):
    # logits: Tensor of shape [B, C]
    return -T * torch.logsumexp(logits / T, dim=1)  # returns [B]

# ---------------------------
# 1) YOLOv11-Pose 모델 로드
# ---------------------------
pose_model = YOLO(YOLO_WEIGHTS)
pose_model.fuse()

# ---------------------------
# 2) Classifier 모델 정의 & 로드
# ---------------------------
ckpt = torch.load(CLS_MODEL_PATH, map_location=DEVICE)
input_dim   = ckpt["input_dim"]
num_classes = ckpt["num_classes"]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, n_classes)
        )
    def forward(self, x):
        return self.net(x)

cls_model = MLP(input_dim, 64, num_classes).to(DEVICE)
cls_model.load_state_dict(ckpt["model_state_dict"])
cls_model.eval()

# ---------------------------
# 3) 웹캠 열기
# ---------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Annotated Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotated Frame", 1280, 720)

# ---------------------------
# 4) 실시간 추론 + 분류
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4.1) YOLO-Pose 추론 & 키포인트+스켈레톤 그리기
    result = pose_model(frame, imgsz=640)[0]
    annotated = result.plot() if len(result.keypoints or []) > 0 else frame.copy()

    energy = None
    pred_cls = None

    # 4.2) 키포인트 벡터 추출
    if result.keypoints is not None and result.keypoints.xy.shape[0] > 0:
        if result.keypoints.conf is None:
            continue
        xy   = result.keypoints.xy[0].cpu().numpy()    # (K,2)
        conf = result.keypoints.conf[0].cpu().numpy()  # (K,)
        h, w = frame.shape[:2]

        kp_vec = []
        for idx in UPPER_BODY_IDX:
            x_px, y_px = xy[idx]
            if conf[idx] < CONF_THRESH:
                kp_vec += [0.0, 0.0]
            else:
                kp_vec += [x_px / w, y_px / h]

        # 4.3) Classifier 예측
        x_tensor = torch.tensor(kp_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits = cls_model(x_tensor)  # [1, C]
            energy = -1 * energy_score(logits, ENERGY_TEMP).item()  # scalar
            pred_cls = int(logits.argmax(dim=1).item())


        # 4.4) “No Pose” 판정
        if energy is None or energy < ENERGY_THRESHOLD:
            text = f"No Pose (E={energy:.2f})" if energy is not None else "No Pose"
            color = (0, 0, 255)
        else:
            text = f"Class: {labels[pred_cls]} (E={energy:.2f})"
            color = (0, 255, 0)
    else:
        text = "No Pose"
        color = (0, 0, 255)

    pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((10, 10), text, font=font, fill=tuple(color[::-1]))
    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 4.6) 화면에 출력
    cv2.imshow("Annotated Frame", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
