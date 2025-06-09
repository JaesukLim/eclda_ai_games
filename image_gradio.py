import os, base64, uuid, requests, qrcode
from io import BytesIO
from pathlib import Path
from typing import Tuple

import openai
import gradio as gr
import time
from PIL import Image
from urllib.parse import quote
from fastapi.responses import FileResponse, Response

os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.getenv("OPENAI_API_KEY")

STYLE_PROMPT = {
    "지브리": "Studio Ghibli anime style",
    "짱구":   "Crayon Shin-chan cartoon illustration",
    "소묘":   "as a delicate black-and-white pencil sketch, chiaroscuro",
    "동물의 숲": "Nintendo Animal Crossing game style",
    "마블":   "as a cinematic Marvel comic hero poster, dramatic lighting",
    "수채화":   "soft and airy watercolor painting, transparent washes, subtle gradients",
    "반고흐":   "thick impasto brush strokes in the style of Vincent van Gogh, swirling night sky hues",
    "민화":     "traditional Korean Minhwa folk painting, flat perspective, vivid primary colors, auspicious motifs",
    "피카소":   "abstract cubist portrait in the style of Pablo Picasso, fractured planes and bold outlines",
}

# ────────────────────────── helpers ──────────────────────────
def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64_str)))

# 공개 URL(share_url) 저장용
BASE_URL: str | None = None
DIR_GEN = Path("generated_images").resolve()

def make_qr_pil(data: str) -> Image.Image:
    """
    문자열 → PIL.Image    (Gradio 호환 QR 코드)
    """
    qr = qrcode.QRCode(
        version=None,                        # 자동 크기
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=2,
    )
    qr.add_data(data)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white")
    # qr_img: qrcode.image.pil.PilImage  →  내부의 진짜 PIL.Image 꺼내기
    return qr_img.get_image() if hasattr(qr_img, "get_image") else qr_img.convert("RGB")

# ──────────────────────── main function ─────────────────────
def stylize(img: Image.Image, style: str) -> Tuple[Image.Image, Image.Image]:
    if img is None:
        raise gr.Error("웹캠 이미지를 먼저 찍어 주세요!")
    if BASE_URL is None:
        raise gr.Error("⚠️ 이 앱은 반드시 launch(share=True) 로 실행해야 합니다.")

    # 1. OpenAI 편집
    png_bytes_in = pil_to_png_bytes(img)
    resp = openai.images.edit(
        model="gpt-image-1",
        prompt=STYLE_PROMPT[style],
        image=("input.png", png_bytes_in, "image/png"),
        n=1,
        size="1024x1024",
    )
    out_b64 = resp.data[0].b64_json
    styled_img = b64_to_pil(out_b64)

    # 2. generated_images/<uuid>.png 저장
    out_dir = Path("generated_images")
    out_dir.mkdir(exist_ok=True)
    fname = f"{uuid.uuid4().hex}.png"
    fpath = DIR_GEN / fname
    styled_img.save(fpath)

    # 3. Gradio 파일 라우트 URL 생성
    file_url = f"{BASE_URL}/img/{fname}"
    qr_img = make_qr_pil(file_url)

    return styled_img, qr_img

def clear():
    return None, None, None

# ────────────────────────── UI 구성 ─────────────────────────
with gr.Blocks(title="AI 사진 스타일 변환기") as demo:
    gr.Markdown("## 📸 AI 사진 스타일 변환기")

    with gr.Row():
        webcam   = gr.Image(sources="webcam", label="웹캠", type="pil")
        style_rb = gr.Radio(list(STYLE_PROMPT.keys()), value="지브리", label="스타일 선택")

    run_btn = gr.Button("✨ 변환하기")

    with gr.Row():
        out_img = gr.Image(label="변환된 이미지")
        qr_img  = gr.Image(label="QR 코드")

    clear_btn = gr.Button("🗑️  Clear")

    run_btn.click(stylize, inputs=[webcam, style_rb], outputs=[out_img, qr_img])
    clear_btn.click(clear, outputs=[webcam, out_img, qr_img])

# ─────────────────────── 앱 실행 및 URL 확보 ───────────────────


app, local_url, share_url = demo.launch(
    share=True,
    prevent_thread_lock=True,                 # Gradio 4.x / 3.45↑
    server_name="0.0.0.0",
    server_port=7000,
)

def register_routes(app):
    @app.get("/img/ping")
    async def ping(): return {"ok": True}

    @app.get("/img/{fname}")
    async def download_img(fname: str):
        """생성된 이미지를 직접 내려주는 엔드포인트."""
        fp = DIR_GEN / fname
        if fp.exists() and fp.is_file():
            return FileResponse(
                fp,
                media_type="image/png",
                filename=fname,
            )
        return Response(status_code=404)



register_routes(app)

BASE_URL = share_url or local_url
print("★ Public URL:", BASE_URL)
print("📜 routes:", [r.path for r in app.routes])

# 메인 스크립트가 종료되면 터널도 닫히므로 루프로 유지
try:
    while True:
        time.sleep(72000)
except KeyboardInterrupt:
    print("▶️  종료합니다.")
