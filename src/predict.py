import pandas as pd
import numpy as np
import skops.io as sio
import joblib
import os

# المسارات
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cpu_price_model.skops")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

def load_model(model_path=MODEL_PATH):
    
    try:
        untrusted_types = sio.get_untrusted_types(file=model_path)
        model = sio.load(model_path, trusted=untrusted_types)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

def load_feature_names(features_path=FEATURES_PATH):
    
    try:
        feature_names = joblib.load(features_path)
        print(f"✅ Loaded {len(feature_names)} feature names")
        return feature_names
    except Exception as e:
        print(f"❌ Error loading feature names: {e}")
        raise e

def predict_price(features, model=None, feature_names=None):
    """
    تنفيذ التنبؤ بناءً على قيم الميزات
    
    Args:
        features: قائمة بالقيم (ترتيبها حسب feature_names)
        model: النموذج المدرب (إذا كان None سيتم تحميله)
        feature_names: أسماء الأعمدة (إذا كان None سيتم تحميلها)
    
    Returns:
        float: السعر المتوقع
    """
    if model is None:
        model = load_model()
    
    if feature_names is None:
        feature_names = load_feature_names()
    
    # تحويل المدخلات إلى DataFrame
    input_data = np.array([features])
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # التنبؤ
    prediction = model.predict(input_df)[0]
    
    return prediction

# مثال للاستخدام
if __name__ == "__main__":
    # مثال: Intel Core i5-12600K
    sample_features = [65, 6, 12, 1, 1000, 50, 2.5, 2.4, 4.0, 0.5, 15.0, 1, 0, 0]
    
    model = load_model()
    feature_names = load_feature_names()
    
    price = predict_price(sample_features, model, feature_names)
    print(f"\n💰 Predicted Price: ${price:,.2f}")