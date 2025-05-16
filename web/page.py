import sys
import os
import streamlit as st
import supervision as sv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from predicts.predict_utils import predict_breed
from inference_sdk import InferenceHTTPClient
import time
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import json
import pandas as pd

def convert_uploadedfile_to_cv2(file):
    img = Image.open(file)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="ATSAdGLXCKfT6cavgKlq"
)
with open("web/assets/css/style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "..", "data", "dogs_cleaned.csv")


spinner_placeholder = None  
def loading_toogle(boolean):
    global spinner_placeholder

    if not boolean:
        if spinner_placeholder is not None:
            spinner_placeholder.empty()
        return
    spinner_html = f"""
        <div class="bg_loader">
            <div class="custom-loader"></div>
        </div>
    """
    # Show spinner
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown(spinner_html, unsafe_allow_html=True)

def classify_owner(row):
    if row["Energy Level"] >= 4 and row["Exercise Needs"] >= 4:
        return "B - Active Lifestyle"
    elif row["Tendency To Bark Or Howl"] <= 2 and row["Dog Size"] == "Small":
        return "C - Family with Kids"
    elif row["Energy Level"] <= 2:
        return "A - Stay-at-Home"
    else:
        return "D - General Owner"
    
with st.container():
    file = st.file_uploader("📁 อัปโหลดรูปภาพไฟล์พันธ์หมาที่ต้องการตรวจสอบ", accept_multiple_files=False, type=["jpg", "jpeg", "png"])
    predict_button = st.button("ตรวจสอบพันธ์หมา")
    if predict_button and file:
        loading_toogle(True)
        image_cv2 = convert_uploadedfile_to_cv2(file)
        file.seek(0)
        df = pd.read_csv(csv_path)
        result = predict_breed(CLIENT, file_obj=image_cv2, num_run=20)
        if result:
            loading_toogle(False)
            with st.container():
                boxes = []
                for pred in result:
                    boxes.append({
                        "bbox": [pred['x'], pred['y'], pred['width'], pred['height']],
                        "label": pred['class'],
                        "confidence": pred['confidence']
                    })
                image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                draw = ImageDraw.Draw(image_pil)
                for box in boxes:
                    x, y, w, h = box["bbox"]
                    left = int(x - w / 2)
                    top = int(y - h / 2)
                    right = int(x + w / 2)
                    bottom = int(y + h / 2)
                    label = f"{box['label']} ({box['confidence']:.2f})"
                    draw.rectangle([left, top, right, bottom], outline="lime", width=5)
                    img_w, img_h = image_pil.size
                    scale = img_h / 600

                    font_size = int(18 * scale)
                    font = ImageFont.truetype("arial.ttf", font_size)

                    border_width = int(3 * scale)
                    padding = int(8 * scale)

                    # คำนวณตำแหน่งกลาง
                    center_x = img_w // 2
                    center_y = img_h // 2

                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    text_x = center_x - text_width // 2
                    text_y = center_y - text_height // 2

                    # วาด
                    draw.rectangle(
                        [text_x - padding, text_y - padding,
                        text_x + text_width + padding, text_y + text_height + padding],
                        fill="lime"
                    )
                    draw.text((text_x, text_y), label, font=font, fill="black")
                st.image(image_pil, caption="ผลการวิเคราะห์", use_container_width=True)
                with st.container(border=True):
                    for pred in result:
                        if pred['confidence'] > 0.5:
                            name_breed = pred['class'].strip()
                            df["Suitable_For"] = df.apply(classify_owner, axis=1)
                            df_get = df.loc[df["Breed Name"] == name_breed]
                            mask = df["Breed Name"].str.lower().eq(name_breed.lower())
                            if mask.any():                 
                                df_breed = df.loc[mask].iloc[0].copy()
                                percent = round(pred["confidence"] * 100, 2)
                                st.write(f'ค่าความเเม่นยำ: {percent}%')
                                st.write(f'พันธุ์สุนัขชื่อ: {df_breed["Breed Name"]}')
                                st.write(f'ขนาด: {df_breed["Dog Size"]}')
                                st.write(f'น้ำหนักเฉลี่ย: {df_breed["Avg. Weight, kg"]} kg')
                                st.write(f'ความสูงเฉลี่ย: {df_breed["Avg. Height, cm"]} cm')
                                st.write(f'เป็นมิตรกับเด็ก: {df_breed["Kid-Friendly"]}')
                                st.write(f'เป็นมิตรกับหมา: {df_breed["Dog Friendly"]}')
                                st.write(f'ความฉลาด: {df_breed["Intelligence"]}')
                                st.write(f'อายุเฉลี่ย: {df_breed["Life Span"]}')
                            else:
                                st.write(f"ไม่พบข้อมูลพันธุ์สุนัขชื่อ '{name_breed}' ในชุดข้อมูล")
                            st.write(f'--------------------------------')

           
            


