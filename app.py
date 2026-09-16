import streamlit as st
import folium
from streamlit_folium import st_folium
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# إعدادات الصفحة
st.set_page_config(page_title="نظام الري الذكي", page_icon="🌿", layout="wide")
st.title("🌿 نظام الري الذكي التنبؤي (AI Irrigation System)")
st.write("حدد موقع المزرعة على الخريطة لجلب البيانات المناخية وتوقعات الذكاء الاصطناعي.")

# 1. الخريطة التفاعلية
m = folium.Map(location=[24.0, 45.0], zoom_start=5)
folium.TileLayer('OpenStreetMap').add_to(m)
map_data = st_folium(m, width=800, height=400)

# الإحداثيات الافتراضية (مشتل بوتانيكا)
lat, lon = 25.5324, 37.0592 
if map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

st.success(f"📍 الإحداثيات المحددة: خط العرض {lat:.4f} | خط الطول {lon:.4f}")

# 2. جلب البيانات وتدريب الذكاء الاصطناعي
st.subheader("☁️ جلب البيانات المناخية وتدريب الذكاء الاصطناعي...")
try:
    # إعداد الاتصال بالـ API
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "et0_fao_evapotranspiration", "wind_speed_10m_max", "shortwave_radiation_sum"],
        "wind_speed_unit": "ms",
    }
    responses = openmeteo.weather_api(url, params = params)
    daily = responses[0].Daily()

    # ترتيب البيانات في جدول
    daily_data = {
        "Date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        ),
        "Temp_Max": daily.Variables(0).ValuesAsNumpy(),
        "Temp_Min": daily.Variables(1).ValuesAsNumpy(),
        "Wind_Max": daily.Variables(3).ValuesAsNumpy(),
        "Solar_Rad": daily.Variables(4).ValuesAsNumpy(),
        "ET0_Actual": daily.Variables(2).ValuesAsNumpy()
    }
    df = pd.DataFrame(data = daily_data)

    # 3. عقل الذكاء الاصطناعي
    X = df[['Temp_Max', 'Temp_Min', 'Wind_Max', 'Solar_Rad']]
    y = df['ET0_Actual']
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    df['ET0_AI_Predicted'] = model.predict(X) # جعل الذكاء الاصطناعي يتوقع

    # 4. عرض الرسوم البيانية
    st.write("📊 **جدول البيانات المناخية (لـ 7 أيام قادمة):**")
    st.dataframe(df)

    st.write("📈 **مقارنة بين الحساب التقليدي وتوقع الذكاء الاصطناعي (ET0):**")
    st.line_chart(df.set_index('Date')[['ET0_Actual', 'ET0_AI_Predicted']])

    # حساب دقة الذكاء الاصطناعي
    mae = mean_absolute_error(df['ET0_Actual'], df['ET0_AI_Predicted'])
    r2 = r2_score(df['ET0_Actual'], df['ET0_AI_Predicted'])

    st.write("🎯 **تقييم دقة الذكاء الاصطناعي (Model Accuracy):**")
    st.info(f"✔️ نسبة ذكاء النموذج (R² Score): **{r2 * 100:.2f}%**")
    st.warning(f"⚠️ متوسط نسبة الخطأ (MAE): **{mae:.2f} ملم/يوم فقط!**")

    # 5. القرار الهندسي (حساب كمية الري)
    st.subheader("💧 قرار الري الذكي (Irrigation Decision)")
    col1, col2, col3 = st.columns(3)
    with col1:
        kc = st.slider("معامل المحصول (Kc):", 0.1, 2.0, 1.2)
    with col2:
        area = st.number_input("مساحة الحقل (متر مربع):", value=100)
    with col3:
        efficiency = st.selectbox("نظام الري المستخدم:", ["تنقيط (90%)", "رش (75%)", "غمر (60%)"])
    
    eff_value = 0.90 if "تنقيط" in efficiency else (0.75 if "رش" in efficiency else 0.60)
    
    # حسابات المهندس
    today_et0 = df['ET0_AI_Predicted'].iloc[0]
    etc = today_et0 * kc
    water_needed = (etc * area) / eff_value

    st.success(f"🌱 الاحتياج المائي الصافي للمحصول (ETc): {etc:.2f} ملم/يوم")
    st.info(f"🚰 كمية الضخ المطلوبة لتعويض الفواقد: **{water_needed:.2f} لتر**")

except Exception as e:
    st.error(f"حدث خطأ: {e}")