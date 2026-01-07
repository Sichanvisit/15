import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd

# 1. 페이지 설정
st.set_page_config(page_title="AI 이미지 분류기", page_icon="🖼️")

# 2. 모델 로딩 (캐싱 적용)
@st.cache_resource
def load_model():
    # Vision Transformer(ViT) 모델 로드
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

# 3. UI 레이아웃
st.title("🖼️ AI 이미지 분류 서비스")
st.write("이미지를 업로드하면 AI가 무엇인지 분석해 드립니다.")

# 파일 업로더
uploaded_file = st.file_uploader("이미지 파일을 선택하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 열기
    image = Image.open(uploaded_file)
    
    # 화면을 두 칼럼으로 나눔 (왼쪽: 이미지, 오른쪽: 결과)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="업로드된 이미지", use_container_width=True)
    
    with col2:
        if st.button("분류 시작"):
            with st.spinner('AI가 분석 중입니다...'):
                # 모델 추론 (Top 5 결과 요청)
                results = classifier(image, top_k=5)
                
                # 결과 표시
                st.subheader("분석 결과")
                
                # 가장 높은 확률의 결과 강조
                top_result = results[0]
                st.success(f"이 이미지는 **{top_result['label']}** 일 확률이 {top_result['score']:.2%} 입니다.")
                
                # 전체 결과를 데이터프레임으로 변환하여 차트 시각화
                df = pd.DataFrame(results)
                st.bar_chart(df.set_index('label'))
                
                # 리스트 형태로 상세 표시
                for res in results:
                    st.write(f"- {res['label']}: {res['score']:.2%}")