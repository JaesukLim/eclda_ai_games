import pygame
import cv2
import torch
import numpy as np
import json
import time
import random
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ----------- 설정 -------------
SCREEN_W, SCREEN_H = 1920, 1080
FPS = 30

# UI 여백
MARGIN = 20

# 폰트
FONT_MAIN  = 32
FONT_SMALL = 24
FONT_PATH  = "BMHANNAPro.ttf"

# 모델 파일
YOLO_WEIGHTS   = "./yolo11m-pose.pt"
CLS_MODEL_PATH = "./pose_classifier.pth"

# 분류 설정
CONF_THRESH       = 0.3
CLASS_CONF_THRESH = 0.5
UPPER_BODY_IDX    = list(range(0, 13))
ENERGY_TEMP       = 1.0
ENERGY_THRESHOLD  = 15

# 게임 설정
TOTAL_STAGES     = 5
PAUSE_DURATION   = 1.5   # 정답 후 대기(초)
TOTAL_TIME_LIMIT = 60.0  # 본 게임 제한 시간(초)

# 레이블
LABELS = ["군인", "하트", "복싱", "가수", "닭",
          "꽃", "배꼽인사", "양궁", "야구선수(타자)", "없음"]

# 랭킹 파일
RANKING_FILE = "ranking.json"
# ------------------------------

def energy_score(logits, T=1.0):
    return -T * torch.logsumexp(logits / T, dim=1)   # [B]

class PoseGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("비전 박사와 함께하는 몸으로 말해요")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.RESIZABLE)
        self.clock  = pygame.time.Clock()

        # 폰트
        self.font_main  = pygame.font.Font(FONT_PATH, FONT_MAIN)
        self.font_small = pygame.font.Font(FONT_PATH, FONT_SMALL)
        self.font_large = pygame.font.Font(FONT_PATH, FONT_MAIN * 2)

        # 카메라
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 모델 로드
        self.pose_model = YOLO(YOLO_WEIGHTS)
        self.pose_model.fuse()

        ckpt = torch.load(CLS_MODEL_PATH, map_location="cpu")
        input_dim   = ckpt["input_dim"]
        num_cls     = ckpt["num_classes"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class MLP(torch.nn.Module):
            def __init__(self, in_dim, hidden, nc):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                    torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                    torch.nn.Linear(hidden // 2, nc)
                )
            def forward(self, x): return self.net(x)

        self.cls_model = MLP(input_dim, 64, num_cls).to(self.device)
        self.cls_model.load_state_dict(ckpt["model_state_dict"])
        self.cls_model.eval()

        # 비전 박사 이미지
        self.vision_img = pygame.image.load("vision.png")
        self.vision_img = pygame.transform.scale(self.vision_img, (180, 270))

        # 랭킹
        self.load_ranking()

        # ------------- 상태 변수 -------------
        # MAIN → TUTORIAL → PLAY → END
        self.state        = "MAIN"
        self.username     = ""
        self.input_active = False

        # 튜토리얼
        self.tutorial_done = False   # ★ 튜토리얼 성공 여부
        self.begin_btn     = None    # ★ 본 게임 시작 버튼 rect

        # 본 게임 변수
        self.stage_prompts = []      # ★ 본 게임 제시어 리스트
        self.stage_index   = 0
        self.prompt_idx    = 1       # 연습 = 하트
        self.global_start  = None
        self.stage_start   = None
        self.paused        = False
        self.pause_until   = 0
        self.bubble_text   = ""

        # 결과
        self.finish_time = None

    # ----------------- 랭킹 -----------------
    def load_ranking(self):
        try:
            with open(RANKING_FILE, "r", encoding="utf-8") as f:
                self.ranking = json.load(f)
        except:
            self.ranking = []

    def save_ranking(self):
        self.ranking.append({"name": self.username, "time": self.finish_time})
        self.ranking = sorted(self.ranking, key=lambda x: x["time"])[:5]
        with open(RANKING_FILE, "w", encoding="utf-8") as f:
            json.dump(self.ranking, f, ensure_ascii=False, indent=2)

    # ------------- 상태 전환 함수 -------------
    def start_tutorial(self):                       # ★
        self.state         = "TUTORIAL"
        self.prompt_idx    = 1          # 하트
        self.tutorial_done = False
        self.paused        = False
        self.bubble_text   = ""

    def start_play(self):                           # ★
        # 본 게임용 제시어 섞기
        self.stage_prompts = random.sample([i for i in range(len(LABELS)) if i != 1 and i != 9], TOTAL_STAGES)
        self.state         = "PLAY"
        self.stage_index   = 1
        self.prompt_idx    = self.stage_prompts[0]
        self.global_start  = time.time()
        self.stage_start   = self.global_start
        self.paused        = False
        self.bubble_text   = ""
        self.finish_time   = None

    # ---------------- 메인 루프 ----------------
    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return

                elif e.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(e.size, pygame.RESIZABLE)

                # -------- 키 입력 (이름) --------
                elif e.type == pygame.KEYDOWN and self.state == "MAIN" and self.input_active:
                    if e.key == pygame.K_BACKSPACE:
                        self.username = self.username[:-1]
                    elif e.key == pygame.K_RETURN:
                        if self.username.strip():
                            self.start_tutorial()
                    else:
                        if len(self.username) < 12 and e.unicode.isprintable():
                            self.username += e.unicode

                # -------- 마우스 클릭 --------
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = e.pos
                    if self.state == "MAIN":
                        self.input_active = self.input_box.collidepoint(mx, my)
                        if self.start_btn.collidepoint(mx, my) and self.username.strip():
                            self.start_tutorial()                     # ★

                    elif self.state == "TUTORIAL":                    # ★
                        if self.tutorial_done and self.begin_btn and self.begin_btn.collidepoint(mx, my):
                            self.start_play()

                    elif self.state == "END":
                        self.state = "MAIN"

            # ---------------- 그리기 ----------------
            self.screen.fill((240, 240, 255))
            if self.state == "MAIN":      self.draw_main()
            elif self.state == "TUTORIAL":self.draw_tutorial()        # ★
            elif self.state == "PLAY":    self.draw_play()
            elif self.state == "END":     self.draw_end()

            pygame.display.flip()

    # ---------------- MAIN 화면 ----------------
    def draw_main(self):
        w, h = self.screen.get_size()

        # 타이틀
        title = self.font_main.render("🕹 비전 박사와 함께하는 몸으로 말해요", True, (20, 20, 60))
        self.screen.blit(title, ((w - title.get_width()) // 2, MARGIN))

        # 이름 입력 박스
        box_w, box_h = 400, 50
        bx = (w - box_w) // 2
        by = title.get_height() + MARGIN * 2
        self.input_box = pygame.Rect(bx, by, box_w, box_h)
        pygame.draw.rect(self.screen, (255, 255, 255), self.input_box)
        pygame.draw.rect(self.screen, (0, 0, 0), self.input_box, 2)
        name_surf = self.font_small.render(self.username or "이름을 입력하세요", True, (50, 50, 50))
        self.screen.blit(name_surf, (bx + 10, by + (box_h - name_surf.get_height()) // 2))

        # ‘튜토리얼 시작’ 버튼
        btn_w, btn_h = 200, 50
        bx2 = (w - btn_w) // 2
        by2 = by + box_h + MARGIN
        self.start_btn = pygame.Rect(bx2, by2, btn_w, btn_h)
        pygame.draw.rect(self.screen, (100, 200, 255), self.start_btn, border_radius=8)
        start_txt = self.font_small.render("튜토리얼 시작", True, (255, 255, 255))
        self.screen.blit(start_txt, (bx2 + (btn_w - start_txt.get_width()) // 2,
                                     by2 + (btn_h - start_txt.get_height()) // 2))

        # 랭킹
        rank_y = by2 + btn_h + MARGIN * 2
        rank_title = self.font_small.render("🏆 오늘의 랭킹 (Top 5)", True, (80, 20, 20))
        self.screen.blit(rank_title, ((w - rank_title.get_width()) // 2, rank_y))
        for i, entry in enumerate(self.ranking):
            txt = f"{i + 1}. {entry['name']} - {entry['time']:.2f}s"
            surf = self.font_small.render(txt, True, (30, 30, 30))
            self.screen.blit(surf, ((w - surf.get_width()) // 2,
                                     rank_y + (i + 1) * (FONT_SMALL + 5)))

    # --------------- 튜토리얼 화면 ---------------
    def draw_tutorial(self):                                # ★
        w, h = self.screen.get_size()
        now = time.time()

        # 웹캠 출력
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_surf = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
            cam_w = int(w * 0.6)
            cam_h = int(cam_w * frame.shape[0] / frame.shape[1])
            cam_rect = cam_surf.get_rect(center=(w * 0.65, h * 0.55))
            cam_surf = pygame.transform.scale(cam_surf, (cam_w, cam_h))
            self.screen.blit(cam_surf, cam_rect)

        # 단계 & 제시어
        stage_txt = self.font_large.render("연습 단계", True, (20, 20, 20))
        self.screen.blit(stage_txt, ((w - stage_txt.get_width()) // 2, MARGIN))
        prompt = LABELS[self.prompt_idx]
        pr_txt = self.font_large.render(prompt, True, (200, 30, 30))
        self.screen.blit(pr_txt, ((w - pr_txt.get_width()) // 2, stage_txt.get_height() + MARGIN * 2))

        # 비전 박사 & 말풍선
        vx, vy = MARGIN, h - self.vision_img.get_height() - MARGIN
        self.screen.blit(self.vision_img, (vx, vy))
        if self.bubble_text:
            bubble = self.font_large.render(self.bubble_text, True, (0, 0, 0))
            bx = vx + self.vision_img.get_width() + 10
            by = vy + 20
            pygame.draw.rect(self.screen, (255, 255, 255),
                             (bx - 5, by - 5, bubble.get_width() + 10, bubble.get_height() + 10))
            pygame.draw.rect(self.screen, (0, 0, 0),
                             (bx - 5, by - 5, bubble.get_width() + 10, bubble.get_height() + 10), 2)
            self.screen.blit(bubble, (bx, by))

        # --- 튜토리얼 판정 (카운트다운 없음) ---
        if not self.tutorial_done and ret:
            result = self.pose_model(frame, imgsz=640)[0]
            kp_conf = getattr(result.keypoints, "conf", None)
            if kp_conf is not None and kp_conf.shape[0] > 0:
                xy   = result.keypoints.xy[0].cpu().numpy()
                conf = result.keypoints.conf[0].cpu().numpy()
                h_f, w_f = frame.shape[:2]
                kp_vec = []
                for idx in UPPER_BODY_IDX:
                    x_px, y_px = xy[idx]
                    if conf[idx] < CONF_THRESH:
                        kp_vec += [0.0, 0.0]
                    else:
                        kp_vec += [x_px / w_f, y_px / h_f]
                x_t = torch.tensor(kp_vec, device=self.device).float().unsqueeze(0)
                with torch.no_grad():
                    logits  = self.cls_model(x_t)
                    energy  = -energy_score(logits, ENERGY_TEMP).item()
                    pred_id = int(logits.argmax(dim=1).item())
                    print(LABELS[pred_id], energy)

                if energy >= ENERGY_THRESHOLD and pred_id == self.prompt_idx and pred_id != 9:
                    self.tutorial_done = True
                    self.bubble_text   = "잘했어! 연습 완료!"

        # 연습 성공 후 ‘게임 시작’ 버튼
        if self.tutorial_done:
            btn_w, btn_h = 250, 60
            bx = (w - btn_w) // 2
            by = stage_txt.get_height() + pr_txt.get_height() + MARGIN * 4
            self.begin_btn = pygame.Rect(bx, by, btn_w, btn_h)
            pygame.draw.rect(self.screen, (50, 180, 50), self.begin_btn, border_radius=8)
            btn_txt = self.font_small.render("게임 시작!", True, (255, 255, 255))
            self.screen.blit(btn_txt, (bx + (btn_w - btn_txt.get_width()) // 2,
                                       by + (btn_h - btn_txt.get_height()) // 2))

    # ---------------- 본 게임 화면 ----------------
    def draw_play(self):
        w, h = self.screen.get_size()
        now = time.time()

        # --- 웹캠 ---
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_surf = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
            cam_w = int(w * 0.6)
            cam_h = int(cam_w * frame.shape[0] / frame.shape[1])
            cam_rect = cam_surf.get_rect(topleft=(w * 0.35, h * 0.2))
            cam_surf = pygame.transform.scale(cam_surf, (cam_w, cam_h))
            self.screen.blit(cam_surf, cam_rect)

        # --- 카운트다운 ---
        elapsed = now - self.global_start
        remain  = max(0, TOTAL_TIME_LIMIT - elapsed)
        time_txt = self.font_large.render(f"⏱ {remain:05.2f}s", True, (20, 20, 20))
        self.screen.blit(time_txt, (MARGIN, MARGIN))

        # --- 단계 & 제시어 ---
        stage_txt = self.font_large.render(f"단계: {self.stage_index}/{TOTAL_STAGES}",
                                           True, (20, 20, 20))
        self.screen.blit(stage_txt, ((w - stage_txt.get_width()) // 2,
                                     MARGIN + time_txt.get_height() + 5))

        prompt = LABELS[self.prompt_idx]
        pr_txt = self.font_large.render(prompt, True, (200, 30, 30))
        self.screen.blit(pr_txt, (w - pr_txt.get_width() - MARGIN, MARGIN))

        # 비전 박사
        vx, vy = MARGIN, h - self.vision_img.get_height() - MARGIN
        self.screen.blit(self.vision_img, (vx, vy))
        if self.bubble_text:
            bubble = self.font_large.render(self.bubble_text, True, (0, 0, 0))
            bx = vx + self.vision_img.get_width() + 10
            by = vy + 20
            pygame.draw.rect(self.screen, (255, 255, 255),
                             (bx - 5, by - 5, bubble.get_width() + 10, bubble.get_height() + 10))
            pygame.draw.rect(self.screen, (0, 0, 0),
                             (bx - 5, by - 5, bubble.get_width() + 10, bubble.get_height() + 10), 2)
            self.screen.blit(bubble, (bx, by))

        # --- 게임 로직 ---
        if not self.paused and remain > 0:
            if ret:
                result = self.pose_model(frame, imgsz=640)[0]
                kp_conf = getattr(result.keypoints, "conf", None)
                if kp_conf is not None and kp_conf.shape[0] > 0:
                    xy   = result.keypoints.xy[0].cpu().numpy()
                    conf = result.keypoints.conf[0].cpu().numpy()
                    h_f, w_f = frame.shape[:2]
                    kp_vec = []
                    for idx in UPPER_BODY_IDX:
                        x_px, y_px = xy[idx]
                        if conf[idx] < CONF_THRESH:
                            kp_vec += [0.0, 0.0]
                        else:
                            kp_vec += [x_px / w_f, y_px / h_f]
                    x_t = torch.tensor(kp_vec, device=self.device).float().unsqueeze(0)
                    with torch.no_grad():
                        logits  = self.cls_model(x_t)
                        energy  = -energy_score(logits, ENERGY_TEMP).item()
                        pred_id = int(logits.argmax(dim=1).item())
                        print(LABELS[pred_id], energy)

                    if energy >= ENERGY_THRESHOLD and pred_id == self.prompt_idx and pred_id != 9:
                        self.paused      = True
                        self.pause_until = now + PAUSE_DURATION
                        self.bubble_text = f"정답! {prompt}"

        else:  # paused
            if self.paused and now >= self.pause_until:
                self.stage_index += 1
                if self.stage_index > TOTAL_STAGES or remain <= 0:
                    self.finish_time = elapsed
                    self.save_ranking()
                    self.state = "END"
                else:
                    # 다음 제시어로
                    if self.stage_index <= TOTAL_STAGES:
                        self.prompt_idx = self.stage_prompts[self.stage_index - 1]
                    self.stage_start = now
                    self.bubble_text = ""
                    self.paused      = False

        # 시간 초과
        if remain <= 0 and self.state == "PLAY":
            self.finish_time = TOTAL_TIME_LIMIT
            self.save_ranking()
            self.state = "END"

    # --------------- END 화면 ---------------
    def draw_end(self):
        w, h = self.screen.get_size()
        txt1 = self.font_main.render("게임 종료!", True, (20, 20, 60))
        txt2 = self.font_main.render(f"소요 시간: {self.finish_time:.2f}s", True, (20, 20, 60))
        self.screen.blit(txt1, ((w - txt1.get_width()) // 2, h * 0.3))
        self.screen.blit(txt2, ((w - txt2.get_width()) // 2, h * 0.4))
        cont = self.font_small.render("클릭하여 메인으로 돌아가기", True, (80, 80, 80))
        self.screen.blit(cont, ((w - cont.get_width()) // 2, h * 0.6))

# ------------------------------------------------
if __name__ == "__main__":
    PoseGame().run()
