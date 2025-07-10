import gradio as gr
import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# --- Carregamento do modelo ---
MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)

# --- Função de processamento da imagem ---
def calculate_morphometrics(result_object):
    masks = result_object.masks
    if not masks:
        return pd.DataFrame()

    morph_data = []
    class_names = result_object.names

    for i, mask_data in enumerate(masks):
        mask_np = mask_data.data[0].cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

        aspect_ratio = 0
        if len(cnt) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            aspect_ratio = MA / ma if ma > 0 else 0

        class_id = int(result_object.boxes[i].cls)
        confidence = float(result_object.boxes[i].conf)

        morph_data.append({
            'cell_id': i + 1,
            'class_name': class_names[class_id],
            'confidence': round(confidence, 4),
            'area_pixels': area,
            'perimeter_pixels': perimeter,
            'circularity': round(circularity, 4),
            'aspect_ratio': round(aspect_ratio, 4)
        })
    return pd.DataFrame(morph_data)

def segment_and_analyze(image):
    # Converte imagem PIL para numpy
    img = np.array(image.convert("RGB"))
    
    results = model.predict(source=img, save=False, conf=0.25)
    result = results[0]
    
    # Imagem com as predições desenhadas
    predicted_img = result.plot(line_width=2, font_size=12)
    predicted_img_rgb = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB)

    # Análise morfométrica
    df_morph = calculate_morphometrics(result)

    return Image.fromarray(predicted_img_rgb), df_morph

# --- Interface Gradio ---
demo = gr.Interface(
    fn=segment_and_analyze,
    inputs=gr.Image(type="pil", label="Imagem de microscopia"),
    outputs=[
        gr.Image(type="pil", label="Células segmentadas"),
        gr.Dataframe(label="Métricas Morfométricas")
    ],
    title="🔬 Análise Morfométrica com YOLOv11",
    description="Faça upload de uma imagem de microscopia para segmentar células e obter métricas como área, perímetro, circularidade e razão de aspecto."
)

if __name__ == "__main__":
    demo.launch()
