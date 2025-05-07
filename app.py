import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

st.title("정서 기반 원격 간호 시스템 (표정 & 행동 분석)")

uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드한 사진', use_column_width=True)
    
    img_array = np.array(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    results = face_detection.process(img_rgb)
    
    if results.detections:
        st.success("얼굴이 감지되었습니다. 표정 분석 중...")
        st.info("가상의 분석 결과: 스트레스 지수 ↑")
    else:
        st.warning("얼굴을 감지하지 못했습니다. 다시 시도하세요.")
