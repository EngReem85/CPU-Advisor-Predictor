import gradio as gr
import pandas as pd
import skops.io as sio
import joblib
import numpy as np
import os

print("Current directory contents:", os.listdir())

# تحميل أسماء الأعمدة
try:
    feature_names = joblib.load("feature_names.pkl")
    print(f"✅ Loaded feature names: {len(feature_names)} features")
except Exception as e:
    print(f"❌ Error loading feature_names.pkl: {e}")
    feature_names = ['tdp', 'cores', 'logicals', 'cpuCount', 'rank', 'samples', 
                     'extracted_ghz', 'speed_ghz', 'turbo_ghz', 'cost_per_rank_point', 'cost_per_core',
                     'brand_encoded', 'category_final_encoded', 'socket_final_encoded']

# تحميل النموذج
try:
    untrusted_types = sio.get_untrusted_types(file="cpu_price_model.skops")
    model = sio.load("cpu_price_model.skops", trusted=untrusted_types)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

#  اسماءالشركات مع عدد المعالجات من البيانات
BRAND_MAPPING = {
    "Intel": 0,        # 2593 معالج
    "AMD": 1,          # 1252 معالج
    "Other": 2,        # 319 معالج
    "Qualcomm": 3,     # 107 معالج
    "MediaTek": 4,     # 61 معالج
    "Samsung": 5,      # 25 معالج
    "VIA": 6,          # 23 معالج
    "Rockchip": 7,     # 19 معالج
    "Unisoc": 8,       # 16 معالج
    "Snapdragon": 9,   # 15 معالج
    "Apple": 10,       # 15 معالج
    "QCT": 11,         # 5 معالج
    "Spreadtrum": 12,  # 3 معالج
    "Nvidia": 13,      # 2 معالج
    "AArch64": 14,     # 2 معالج
    "Microsoft": 15,   # 2 معالج
}

# الفئات وعدد المعالجات في البيانات
CATEGORY_MAPPING = {
    "Server/Workstation": 0,   # 1313 معالج
    "Desktop": 1,              # 1301 معالج
    "Laptop": 2,               # 1195 معالج
    "Embedded/IoT": 3,         # 537 معالج
    "Mobile / Tablet": 4,      # 72 معالج
    "Unknown": 5,              # 41 معالج
}

#السوكت وعدد المعالجات في البيانات ذي اعلى الفئات لان كانت فوق 190 
SOCKET_MAPPING = {
    "Unknown": 0,        # 868 معالج
    "FCLGA3647": 1,      # 178 معالج
    "AM4": 2,            # 140 معالج
    "LGA1366": 3,        # 104 معالج
    "LGA1155": 4,        # 103 معالج
    "Other": 5           # باقي المقابس
}

def predict_price(tdp, cores, logicals, cpuCount, rank, samples, 
                  extracted_ghz, speed_ghz, turbo_ghz, 
                  cost_per_rank_point, cost_per_core,
                  brand_name, category_name, socket_name):
    """دالة التنبؤ مع مدخلات سهلة الفهم"""
    
    # تحويل الأسماء إلى أرقام مشفرة
    brand_encoded = BRAND_MAPPING.get(brand_name, 2)  # 2 = Other
    category_encoded = CATEGORY_MAPPING.get(category_name, 5)  # 5 = Unknown
    socket_encoded = SOCKET_MAPPING.get(socket_name, 5)  # 5 = Other
    
    # إنشاء مصفوفة المدخلات (14 عموداً)
    input_data = np.array([[
        tdp, cores, logicals, cpuCount, rank, samples, 
        extracted_ghz, speed_ghz, turbo_ghz, 
        cost_per_rank_point, cost_per_core,
        brand_encoded, category_encoded, socket_encoded
    ]])
    
    # تحويل إلى DataFrame
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # التنبؤ
    prediction = model.predict(input_df)[0]
    
    # تنسيق النتيجة
    return f"💰 السعر المتوقع: ${prediction:,.2f}"

# ========== واجهة المستخدم ==========

# قائمةالشركات 
brand_choices = ["Intel", "AMD", "Other", "Qualcomm", "MediaTek", 
                 "Samsung", "VIA", "Rockchip", "Unisoc", "Snapdragon", 
                 "Apple", "QCT", "Spreadtrum", "Nvidia", "AArch64", "Microsoft"]

# قائمة الفئات
category_choices = ["Server/Workstation", "Desktop", "Laptop", "Embedded/IoT", "Mobile / Tablet", "Unknown"]

# قائمة السوكت (حسب شيوعها في البيانات)
socket_choices = ["Unknown", "FCLGA3647", "AM4", "LGA1366", "LGA1155", "Other"]

inputs = [
    # الأعمدة الرقمية
    gr.Number(label="استهلاك الطاقة (TDP) - واط", value=65),
    gr.Number(label="عدد الأنوية (Cores)", value=6),
    gr.Number(label="عدد الخيوط (Threads)", value=12),
    gr.Number(label="عدد المقابس (Sockets)", value=1),
    gr.Number(label="ترتيب الأداء (Rank) - كلما قل الرقم كان أفضل", value=1000),
    gr.Number(label="عدد العينات (Samples)", value=50),
    gr.Number(label="السرعة المستخرجة من الاسم (GHz)", value=2.5),
    gr.Number(label="السرعة الأساسية (GHz)", value=2.4),
    gr.Number(label="السرعة القصوى (Turbo - GHz)", value=4.0),
    gr.Number(label="التكلفة لكل نقطة ترتيب", value=0.5),
    gr.Number(label="التكلفة لكل نواة", value=15.0),
    
    # الأعمدة الفئوية (قوائم منسدلة)
    gr.Dropdown(
        choices=brand_choices,
        label="العلامة التجارية (Brand)",
        value="Intel"
    ),
    gr.Dropdown(
        choices=category_choices,
        label="فئة المعالج (Category)",
        value="Desktop"
    ),
    gr.Dropdown(
        choices=socket_choices,
        label="نوع المقبس (Socket)",
        value="AM4"
    ),
]

outputs = gr.Textbox(label="نتيجة التنبؤ", lines=2)

title = "💰 تنبؤ أسعار المعالجات"
description = """
### 🖥️ أدخل مواصفات المعالج لتحصل على تقدير لسعره

**ملاحظة:** هذا التطبيق للأغراض التعليمية والبحثية فقط. الأسعار تقديرية وقد لا تعكس الواقع الحالي.

📊 **إحصائيات قاعدة البيانات:**
- **Intel:** 2,593 معالج | **AMD:** 1,252 معالج
- **أكثر الفئات شيوعاً:** Server/Workstation (1,313), Desktop (1,301), Laptop (1,195)
- **أكثر المقابس شيوعاً:** Unknown (868), FCLGA3647 (178), AM4 (140)
"""

examples = [
    # [TDP, cores, logicals, cpuCount, rank, samples, extracted_ghz, speed_ghz, turbo_ghz, cost_per_rank, cost_per_core, brand, category, socket]
    [65, 6, 12, 1, 1000, 50, 2.5, 2.4, 4.0, 0.5, 15.0, "Intel", "Desktop", "LGA1155"],
    [125, 8, 16, 1, 500, 100, 3.0, 3.0, 4.5, 0.8, 20.0, "AMD", "Desktop", "AM4"],
    [15, 4, 8, 1, 2000, 30, 2.0, 1.8, 3.2, 0.3, 8.0, "Qualcomm", "Mobile / Tablet", "Unknown"],
    [95, 6, 12, 1, 800, 60, 2.8, 2.6, 4.2, 0.6, 18.0, "Intel", "Server/Workstation", "FCLGA3647"],
    [10, 8, 8, 1, 1500, 40, 3.5, 3.2, 4.8, 0.4, 12.0, "Apple", "Laptop", "Unknown"],
    [65, 4, 8, 1, 1200, 45, 2.2, 2.0, 3.8, 0.4, 12.0, "AMD", "Desktop", "AM4"],
    [35, 2, 4, 1, 2500, 20, 1.8, 1.6, 2.8, 0.2, 5.0, "Intel", "Embedded/IoT", "Unknown"],
]

demo = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    examples=examples,
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()