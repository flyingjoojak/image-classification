import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import requests
from io import BytesIO
import pandas as pd

# 1. 페이지 설정
st.set_page_config(
    page_title="이미지 분류기",
    page_icon="🖼️",
    layout="centered"
)

# 2. 모델 로딩 함수 (캐싱 적용)
@st.cache_resource
def load_model():
    """
    Hugging Face의 'google/vit-base-patch16-224' 모델을 로드하여 이미지를 분류하는 파이프라인을 반환합니다.
    """
    model_pipeline = pipeline("image-classification", model="google/vit-base-patch16-224")
    return model_pipeline

def get_emoji(label):
    """
    라벨에 따라 적절한 이모지를 반환합니다.
    """
    label_lower = label.lower()
    if 'dog' in label_lower or 'retriever' in label_lower:
        return "🐶"
    elif 'cat' in label_lower or 'tabby' in label_lower:
        return "🐱"
    elif 'bird' in label_lower:
        return "🐦"
    elif 'car' in label_lower or 'vehicle' in label_lower:
        return "🚗"
    elif 'food' in label_lower or 'pizza' in label_lower or 'burger' in label_lower:
        return "🍕"
    elif 'flower' in label_lower or 'rose' in label_lower:
        return "🌸"
    else:
        return "🏷️"

def process_image(image, classifier):
    """
    이미지를 받아 분류하고 결과를 시각화합니다. (버튼 없음, 즉시 실행)
    """
    st.image(image, caption="입력된 이미지", width="stretch")

    with st.spinner("분류 중입니다..."):
        # 예측 수행
        results = classifier(image)
        
        # 결과 출력
        top_result = results[0]
        emoji = get_emoji(top_result['label'])
        
        st.success(f"분류 완료! {emoji}")
        st.markdown(f"### {emoji} {top_result['label']}")
        
        # Metric 표시
        st.metric(label="최고 확률", value=top_result['label'], delta=f"{top_result['score']:.2%}")

        st.markdown("---")
        st.markdown("#### 상세 결과 목록")
        # 상위 결과 텍스트 출력
        for result in results[:5]:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**{result['label']}**")
            with col2:
                st.progress(result['score'])
                st.caption(f"{result['score']:.2%}")
    st.markdown("---")


# 메인 UI
def main():
    st.title("🖼️ 이미지 분류 AI 서비스")
    st.markdown("이미지(파일, URL, 카메라)를 입력하면 AI가 무엇인지 분류해줍니다.")
    st.markdown("---")

    # 모델 로드 (최초 1회만 로딩됨)
    with st.spinner("AI 모델을 불러오는 중입니다..."):
        classifier = load_model()

    # 3. 이미지 입력 (Tabs 사용)
    tab1, tab2, tab3 = st.tabs(["📁 파일 업로드", "🔗 이미지 URL", "📸 카메라 촬영"])
    
    images_to_process = [] # image_object 리스트

    with tab1:
        uploaded_files = st.file_uploader("분류할 이미지를 업로드하세요 (여러 장 가능)", 
                                        type=["jpg", "jpeg", "png"], 
                                        accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                images_to_process.append(image)

    with tab2:
        url = st.text_input("이미지 URL을 입력하세요")
        if url:
            try:
                response = requests.get(url)
                response.raise_for_status() 
                image = Image.open(BytesIO(response.content))
                images_to_process.append(image)
            except Exception as e:
                st.error(f"이미지를 불러올 수 없습니다: {e}")

    with tab3:
        camera_image = st.camera_input("카메라로 사진을 찍으세요")
        if camera_image is not None:
            image = Image.open(camera_image)
            images_to_process.append(image)

    # 수집된 모든 이미지 일괄 처리
    if images_to_process:
        if st.button("🚀 이미지 분류 실행 (일괄 처리)", type="primary"):
            for i, image in enumerate(images_to_process):
                st.subheader(f"Image {i+1}")
                process_image(image, classifier)

if __name__ == "__main__":
    main()
