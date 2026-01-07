import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import requests
from io import BytesIO

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

# 메인 UI
def main():
    st.title("🖼️ 이미지 분류 AI 서비스")
    st.markdown("이미지를 업로드하거나 URL을 입력하면 AI가 해당 이미지가 무엇인지 분류해줍니다.")
    st.markdown("---")

    # 모델 로드 (최초 1회만 로딩됨)
    with st.spinner("AI 모델을 불러오는 중입니다..."):
        classifier = load_model()

    # 3. 이미지 입력 (Tabs 사용)
    tab1, tab2 = st.tabs(["📁 파일 업로드", "🔗 이미지 URL"])
    
    image = None

    with tab1:
        uploaded_file = st.file_uploader("분류할 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
             image = Image.open(uploaded_file)

    with tab2:
        url = st.text_input("이미지 URL을 입력하세요")
        if url:
            try:
                response = requests.get(url)
                response.raise_for_status() # HTTP 에러 발생 시 예외 처리
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                st.error(f"이미지를 불러올 수 없습니다: {e}")
                image = None

    if image is not None:
        # 4. 이미지 처리 및 표시
        st.image(image, caption="입력된 이미지", width="stretch")

        # 5. 분류 실행 버튼
        if st.button("이미지 분류 실행"):
            with st.spinner("분류 중입니다..."):
                # 예측 수행
                results = classifier(image)
                
                # 6. 결과 출력
                st.success("분류 완료!")
                st.markdown("### 분류 결과")

                # Top 1 결과 강조
                top_result = results[0]
                st.metric(label="가장 높은 확률", value=top_result['label'], delta=f"{top_result['score']:.2%}")

                st.markdown("---")
                st.markdown("#### 상세 결과")

                # 상위 결과 리스트 출력 및 Progress Bar 시각화
                for result in results:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**{result['label']}**")
                    with col2:
                        st.progress(result['score'])
                        st.caption(f"{result['score']:.2%}")

if __name__ == "__main__":
    main()
