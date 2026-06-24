import sys
import os

# إضافة مجلد src إلى مسار Python
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import load_model, load_feature_names, predict_price

def test_load_model():
    
    try:
        model = load_model()
        assert model is not None
        print("✅ test_load_model: PASSED")
    except Exception as e:
        print(f"❌ test_load_model: FAILED - {e}")

def test_load_features():
   
    try:
        features = load_feature_names()
        assert len(features) > 0
        print(f"✅ test_load_features: PASSED ({len(features)} features)")
    except Exception as e:
        print(f"❌ test_load_features: FAILED - {e}")

def test_prediction():
    """اختبار التنبؤ"""
    try:
        # Intel Core i5-12600K
        sample = [65, 6, 12, 1, 1000, 50, 2.5, 2.4, 4.0, 0.5, 15.0, 1, 0, 0]
        model = load_model()
        features = load_feature_names()
        price = predict_price(sample, model, features)
        
        assert isinstance(price, float)
        assert price > 0
        print(f"✅ test_prediction: PASSED (Predicted: ${price:,.2f})")
    except Exception as e:
        print(f"❌ test_prediction: FAILED - {e}")

if __name__ == "__main__":
    print("="*50)
    print(" Running Tests...")
    print("="*50)
    
    test_load_model()
    test_load_features()
    test_prediction()
    
    print("="*50)
    print("✅ All tests completed!")