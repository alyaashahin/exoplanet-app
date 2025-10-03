import streamlit as st
import joblib
import pandas as pd
import numpy as np

# عنوان التطبيق
st.title('🌌 مصنف الكواكب الخارجية')
st.write('تطبيق للتنبؤ باحتمالية وجود كواكب خارج المجموعة الشمسية')

# تحميل النموذج
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_model.pkl')
        return model
    except:
        st.error("لم يتم العثور على ملف النموذج. تأكد من وجود 'best_model.pkl' في نفس المجلد")
        return None

model = load_model()

if model is not None:
    # إنشاء واجهة الإدخال
    st.sidebar.header('🔧 إدخال معايير الكوكب')
    
    # تقسيم المعايير إلى مجموعات
    st.sidebar.subheader('المعايير الأساسية')
    koi_period = st.sidebar.slider('koi_period', -0.39, 3.88, 0.0)
    koi_depth = st.sidebar.slider('koi_depth', -0.57, 3.79, 0.0)
    koi_duration = st.sidebar.slider('koi_duration', -1.03, 3.61, 0.0)
    
    st.sidebar.subheader('معايير الحجم والكتلة')
    koi_prad = st.sidebar.slider('koi_prad', -1.14, 3.69, 0.0)
    koi_impact = st.sidebar.slider('koi_impact', -0.63, 3.63, 0.0)
    
    st.sidebar.subheader('معايير الطاقة والحرارة')
    koi_teq = st.sidebar.slider('koi_teq', -1.43, 3.57, 0.0)
    koi_insol = st.sidebar.slider('koi_insol', -0.31, 3.74, 0.0)
    log_insol = st.sidebar.slider('log_insol', -1.67, 0.95, 0.0)
    
    # إنشاء dataframe للإدخال
    input_data = pd.DataFrame([{
        'koi_period': koi_period,
        'koi_period_err1': 0.0,
        'koi_time0bk': 0.0,
        'koi_time0bk_err1': 0.0,
        'koi_impact': koi_impact,
        'koi_impact_err1': 0.0,
        'koi_impact_err2': 0.0,
        'koi_duration': koi_duration,
        'koi_duration_err1': 0.0,
        'koi_depth': koi_depth,
        'koi_depth_err1': 0.0,
        'koi_prad': koi_prad,
        'koi_prad_err1': 0.0,
        'koi_prad_err2': 0.0,
        'koi_teq': koi_teq,
        'koi_insol': koi_insol,
        'koi_insol_err1': 0.0,
        'koi_model_snr': 0.0,
        'koi_steff': 0.0,
        'koi_steff_err1': 0.0,
        'koi_steff_err2': 0.0,
        'koi_slogg': 0.0,
        'koi_slogg_err1': 0.0,
        'koi_slogg_err2': 0.0,
        'koi_srad_err1': 0.0,
        'koi_srad_err2': 0.0,
        'ra': 0.0,
        'dec': 0.0,
        'koi_kepmag': 0.0,
        'depth_to_srad': 0.0,
        'prad_to_srad_ratio': 0.0,
        'period_to_impact': 0.0,
        'log_insol': log_insol,
        'log_snr': 0.0
    }])
    
    # زر التنبؤ
    if st.sidebar.button('🔍 تنبؤ'):
        # التنبؤ
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # عرض النتائج
        st.subheader('📈 النتائج:')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.success('✅ الكوكب محتمل')
            else:
                st.error('❌ الكوكب غير محتمل')
        
        with col2:
            st.metric('احتمالية كونها كوكب', f'{probability[1]:.2%}')
        
        # مخطط الاحتمالات
        st.subheader('📊 توزيع الاحتمالات:')
        prob_df = pd.DataFrame({
            'الفئة': ['غير كوكب', 'كوكب'],
            'الاحتمالية': probability
        })
        st.bar_chart(prob_df.set_index('الفئة'))
    
    # معلومات عن النموذج
    st.sidebar.markdown('---')
    st.sidebar.subheader('ℹ️ معلومات النموذج')
    st.sidebar.write('**أفضل نموذج:** CatBoost')
    st.sidebar.write('**الدقة:** 83.3%')
    st.sidebar.write('**عينات التدريب:** 3,178')
    
else:
    st.warning('⚠️ يرجى التأكد من وجود ملف النموذج في المجلد الصحيح')

# قسم المعلومات
st.markdown('---')
st.subheader('📖 معلومات عن المشروع')
st.write('''
هذا التطبيق يستخدم نموذج CatBoost المدرب على بيانات الكواكب الخارجية من مهمة كيبلر.
- **34 ميزة** فلكية مختلفة
- **3,178** عينة تدريب
- **1,363** عينة اختبار
- دقة تصل إلى **83.3%**
''')
