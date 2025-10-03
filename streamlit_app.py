import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# إعداد الصفحة
st.set_page_config(
    page_title="Cosmic Detective - كاشف الكواكب",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تنسيق CSS لمحاكاة الموقع الأصلي
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #764ba2;
        margin: 1rem 0;
        font-weight: 600;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stSlider > div > div > div {
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# الهيدر الرئيسي
st.markdown('<h1 class="main-header">🌌 Cosmic Detective - كاشف الكواكب</h1>', unsafe_allow_html=True)
st.markdown("### اكتشف الكواكب الخارجية باستخدام الذكاء الاصطناعي")

# المحاكاة - في الواقع الحقيقي ستقوم بتحميل النموذج
# @st.cache_resource
# def load_model():
#     return joblib.load('best_model.pkl')
# model = load_model()

# المحتوى الرئيسي
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="sub-header">🔭 معطيات الكوكب المرصود</div>', unsafe_allow_html=True)
    
    # مجموعة من sliders للمعايير الرئيسية
    with st.expander("📊 المعايير المدارية", expanded=True):
        koi_period = st.slider("الفترة المدارية (أيام)", 0.1, 1000.0, 365.0, key="period")
        koi_duration = st.slider("مدة العبور (ساعات)", 0.1, 24.0, 8.0, key="duration")
    
    with st.expander("🌍 خصائص الكوكب"):
        koi_prad = st.slider("نصف القطر (نصف قطر الأرض)", 0.1, 20.0, 1.0, key="prad")
        koi_depth = st.slider("عمق العبور (ppm)", 100, 100000, 5000, key="depth")
    
    with st.expander("⭐ خصائص النجم المضيف"):
        koi_teq = st.slider("درجة الحرارة التوازنية (كلفن)", 100, 3000, 288, key="teq")
        koi_insol = st.slider("الإشعاع النجمي (مضاعفات الأرض)", 0.1, 100.0, 1.0, key="insol")
    
    # زر التحليل
    if st.button("🔍 تحليل الكوكب", use_container_width=True):
        st.session_state.analyzed = True

with col2:
    if st.session_state.get('analyzed', False):
        st.markdown('<div class="sub-header">📈 نتائج التحليل</div>', unsafe_allow_html=True)
        
        # محاكاة نتائج النموذج
        planet_probability = np.random.uniform(0.7, 0.95)
        confidence = "عالية" if planet_probability > 0.8 else "متوسطة"
        
        # بطاقة النتيجة الرئيسية
        st.markdown(f"""
        <div class="result-card">
            <h3 style="color: white; margin: 0;">🎯 نتيجة التحليل</h3>
            <h1 style="color: white; font-size: 3rem; margin: 0.5rem 0;">{planet_probability:.1%}</h1>
            <p style="color: white; font-size: 1.2rem; margin: 0;">
                احتمالية أن يكون هذا الكوكب حقيقياً<br>
                <strong>ثقة {confidence}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # المقاييس التفصيلية
        col2a, col2b, col2c = st.columns(3)
        
        with col2a:
            st.markdown("""
            <div class="metric-card">
                <h4>📏 الحجم</h4>
                <h3>أرضي</h3>
                <p>مناسب للحياة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2b:
            st.markdown("""
            <div class="metric-card">
                <h4>🌡️ الحرارة</h4>
                <h3>معتدل</h3>
                <p>منطقة صالحة للسكن</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2c:
            st.markdown("""
            <div class="metric-card">
                <h4>⭐ النجم</h4>
                <h3>مستقر</h3>
                <p>قزم أصفر</p>
            </div>
            """, unsafe_allow_html=True)

# قسم التصورات البيانية
if st.session_state.get('analyzed', False):
    st.markdown("---")
    st.markdown('<div class="sub-header">📊 التصورات العلمية</div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # مخطط احتمالات
        labels = ['كوكب حقيقي', 'إيجابي كاذب']
        values = [planet_probability, 1 - planet_probability]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            hole=.3,
            marker_colors=['#667eea', '#764ba2']
        )])
        fig_pie.update_layout(title="توزيع الاحتمالات")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col4:
        # مخطط الخصائص
        categories = ['الحجم', 'الحرارة', 'المدار', 'النجم', 'الإشعاع']
        values_radar = [0.8, 0.9, 0.7, 0.85, 0.75]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values_radar + [values_radar[0]],
            theta=categories + [categories[0]],
            fill='toself',
            line=dict(color='#667eea')
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False,
            title="ملف الكوكب"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# قسم المعلومات
st.markdown("---")
st.markdown('<div class="sub-header">🔬 معلومات علمية</div>', unsafe_allow_html=True)

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("""
    **📡 كيف يعمل الكشف؟**
    - تحليل بيانات العبور الفلكي
    - نمذجة منحنى الضوء
    - كشف التغيرات الدورية
    - تحليل الإشارات الضوئية
    """)

with info_col2:
    st.markdown("""
    **🌍 منطقة صالحة للسكن**
    - مسافة مناسبة من النجم
    - درجة حرارة متوسطة
    - وجود ماء سائل
    - غلاف جوي مستقر
    """)

with info_col3:
    st.markdown("""
    **🤖 الذكاء الاصطناعي**
    - شبكات عصبية متقدمة
    - خوارزميات تعلم آلي
    - تحليل أنماط معقدة
    - تنبؤات عالية الدقة
    """)

# الفوتر
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌌 Cosmic Detective Tool - أداة محاكاة للكشف عن الكواكب الخارجية</p>
    <p>هذا تطبيق محاكاة لأغراض التعليم والت demonstration</p>
</div>
""", unsafe_allow_html=True)

# تهيئة حالة الجلسة
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
