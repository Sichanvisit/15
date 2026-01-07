import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd

# 1. 페이지 설정
st.set_page_config(page_title="AI 프리미엄 이미지 분류기", page_icon="🤖", layout="wide")

# 2. 모델 로딩 (캐싱 적용)
@st.cache_resource
def load_model():
    # Vision Transformer(ViT) 모델 로드
    return pipeline("image-classification", model="google/vit-base-patch16-224")

with st.spinner('AI 모델을 준비 중입니다...'):
    classifier = load_model()

# 3. UI 레이아웃
st.title("🤖 AI 인텔리전트 이미지 분석 서비스")
st.write("사진을 업로드하거나 직접 촬영하여 AI가 무엇인지 분석하는 과정을 경험해보세요.")

# 사이드바에 옵션 추가
st.sidebar.header("설정")
top_k = st.sidebar.slider("표시할 결과 개수", min_value=1, max_value=10, value=5)

# 4. 입력 방식 선택 (탭 활용)
input_tab1, input_tab2 = st.tabs(["📁 파일 업로드", "📸 카메라 촬영"])

source_image = None

with input_tab1:
    uploaded_file = st.file_uploader("이미지 파일을 선택하세요...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        source_image = Image.open(uploaded_file)

with input_tab2:
    camera_photo = st.camera_input("카메라로 사진을 찍어보세요!")
    if camera_photo:
        source_image = Image.open(camera_photo)

# 5. 분석 섹션
if source_image is not None:
    st.divider()
    col1, col2 = st.columns([1, 1]) # 화면 비율 설정

    with col1:
        st.subheader("📷 대상 이미지")
        st.image(source_image, use_container_width=True, caption="분석 중인 이미지")

    with col2:
        st.subheader("🔍 AI 분석 결과")
        if st.button("AI 분석 실행", type="primary", use_container_width=True):
            with st.spinner('데이터를 분석 중입니다...'):
                # 모델 추론
                results = classifier(source_image, top_k=top_k)
                
                # 가장 유력한 결과 표시
                top_res = results[0]
                st.metric(label="예측 1순위", value=top_res['label'], delta=f"{top_res['score']:.2%}")
                
                # 상세 결과 시각화
                st.write("---")
                df = pd.DataFrame(results)
                
                # 막대 차트 (Plotly 스타일)
                st.bar_chart(df.set_index('label'))

                # 프로그레스 바 형태의 상세 리스트
                for res in results:
                    col_label, col_score = st.columns([3, 1])
                    with col_label:
                        st.write(f"**{res['label']}**")
                    with col_score:
                        st.write(f"{res['score']:.2%}")
                    st.progress(float(res['score']))
else:
    st.info("좌측 상단에서 파일을 업로드하거나 카메라를 사용해 보세요.")