# CPU Advisor — Performance Prediction & Analytics

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange.svg)](https://xgboost.readthedocs.io/)
[![Gradio](https://img.shields.io/badge/Gradio-4.12.0-green.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-ff69b4.svg)](رابط-التطبيق)

---

## Overview

**CPU Advisor** is an end-to-end machine learning platform that predicts CPU performance rankings based on technical specifications. It combines:

-  **Machine Learning Model** (XGBoost) with R² ~ 0.89
-  **Interactive Web Application** (Gradio) hosted on Hugging Face Spaces
-  **Power BI Dashboard** for interactive data visualization
- **Automated Data Pipeline** for cleaning and feature engineering

The project helps consumers, researchers, and tech enthusiasts make informed decisions when comparing processors.

---

##  The Challenge

With thousands of CPU models in the market, users face challenges in:

- Understanding true performance differences between processors
- Making informed purchasing decisions
- Comparing CPUs across different brands and generations
- Estimating the value of used processors

**Our Solution:** A machine learning-powered platform that predicts performance rankings and provides deep analytical insights.

---

##  Technologies Used

| Domain | Tools |
|--------|-------|
| **Programming** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Web Application** | Gradio |
| **Deployment** | Hugging Face Spaces |
| **Dashboard** | Power BI |
| **Model Serialization** | Skops, Joblib |
| **Version Control** | Git, GitHub |

---

##  Dataset Overview

| Feature | Value |
|---------|-------|
| **Total CPUs** | 4,000+ |
| **Original Columns** | 13 |
| **Engineered Columns** | 11 |
| **Total Columns** | 24 |
| **Target Variable** | Performance `rank` (0 = Best) |
| **Features Used** | 14 (TDP, cores, clock speed, brand, category, socket, etc.) |

### Top Brands
| Brand | Count |
|-------|-------|
| Intel | 2,593 |
| AMD | 1,252 |
| Other | 319 |
| Qualcomm | 107 |
| MediaTek | 61 |

### Top Categories
| Category | Count |
|----------|-------|
| Server/Workstation | 1,313 |
| Desktop | 1,301 |
| Laptop | 1,195 |

---

##  Features

### 1.  Machine Learning Model
- **Algorithm:** XGBoost
- **Target:** Performance Ranking (0 = Best)
- **R² Score:** ~0.95
- **Key Features:** TDP, core count, clock speed, brand, category, socket

### 2.  Web Application
- **Framework:** Gradio
- **Bilingual:** English & Arabic interface
- **Instant Predictions:** Get performance rankings in seconds
- **Pre-loaded Examples:** Quick testing with real CPU data

### 3. Power BI Dashboard
- **Interactive Visualizations:** Bar charts, scatter plots, heatmaps
- **Filters:** Filter by brand, category, TDP, release year
- **Trend Analysis:** CPU performance evolution over time
- **Brand Comparison:** Compare Intel vs AMD vs ARM

### 4. Data Pipeline
- **Automated Cleaning:** Handle missing values, outliers
- **Feature Engineering:** 11 new columns added
- **Data Validation:** Ensure data quality and consistency

---

## 📂 Project Structure

```
cpu-price-predictor/
│
├── README.md                      # Project documentation
├── LICENSE                        # MIT License
├── .gitignore                     # Files to ignore in Git
├── requirements.txt               # Python dependencies
│
├── data/                          # Data files
│   ├── raw_data.csv               # Original dataset (Kaggle)
│   └── Final_CPU_Data_Cleaned.csv           # Processed dataset (24 columns)
│
├── notebooks/                     # Jupyter notebooks
│   ├── CPU_data_cleaner.ipynb             # Data cleaning script
│   └── cpu_eda_training.ipynb     # EDA, cleaning, model training
│
├── src/                           # Source code
│   └── predict.py                 # Prediction functions
│
├── models/                        # Trained models
│   ├── cpu_price_model.skops      # XGBoost model
│   └── feature_names.pkl          # Feature names for inference
│
├── huggingface_space/             # Hugging Face Space files
│   ├── app.py                     # Gradio web application
│   └── requirements.txt           # Space-specific dependencies
│
├── powerbi_dashboard/             # Power BI files
│   └── cpu_dashboard.pbix         # Interactive dashboard
│
└── tests/                         # Unit tests
    └── test_predict.py            # Test prediction functions
```

---

##  Installation & Setup

### Prerequisites
- Python 3.10+
- Git
- (Optional) Power BI Desktop for viewing the dashboard

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/cpu-price-predictor.git
cd cpu-price-predictor
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

1. Download the original dataset from [Kaggle](https://www.kaggle.com/datasets/reem85/cpu-specs-dataset-for-ml)
2. Place it in `data/Final_CPU_Data_Cleaned.csv`


### Step 5: Train the Model (Optional)

Open `notebooks/cpu_eda_training.ipynb` in Jupyter and run all cells to retrain the model.

### Step 6: Run the Web Application

```bash
python huggingface_space/app.py
```

Open `http://localhost:7860` in your browser.

---

##  How to Use

### Via Web Application
1. Enter CPU specifications (TDP, cores, clock speed, brand, category, socket)
2. Click **Predict**
3. Get the predicted performance ranking instantly

### Via Power BI Dashboard
1. Open `powerbi_dashboard/cpu_dashboard.pbix`
2. Use interactive filters to explore data
3. Analyze performance trends and comparisons

### Via Python Code

```python
from src.predict import load_model, predict_price

# Load model
model = load_model()

# Sample features: [TDP, cores, logicals, cpuCount, rank, samples, extracted_ghz, speed_ghz, turbo_ghz, cost_per_rank, cost_per_core, brand_encoded, category_encoded, socket_encoded]
features = [65, 6, 12, 1, 1000, 50, 2.5, 2.4, 4.0, 0.5, 15.0, 1, 0, 0]

# Predict
price = predict_price(features, model)
print(f"💰 Predicted Price: ${price:,.2f}")
```

---

##  Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | XGBoost |
| **R² Score** | 0.9573 |
| **RMSE** | $408.77 |
| **MAE** | $100.42 |

### Feature Importance
| Feature | Importance |
|---------|------------|
| cost_per_rank_point | 0.68 |
| cost_per_core | 0.22 |
| cpuCount | 0.03 |
| cores | 0.01 |
| turbo_ghz | 0.01 |

---

## 🔗 Links

| Resource | Link |
|----------|------|
| **Live Application** | [Hugging Face Space](https://huggingface.co/spaces/EngReem85/cpu-price-predictor) |
| **GitHub Repository** | [GitHub Repository](https://github.com/EngReem85/CPU-Advisor-Predictor) |
| **Original Dataset** | [Kaggle](https://www.kaggle.com/datasets/lincolnzh/cpu-specifications-dataset) |
| **Power BI Dashboard** | [Download .pbix](رابط-اللوحة) |

---

##  Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. Create a feature branch: `git checkout -b feature/improvement`
3. Commit your changes: `git commit -m "Add improvement"`
4. Push: `git push origin feature/improvement`
5. Open a **Pull Request**

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

##  Author

**Reem Algethami**

- GitHub: [GitHub](https://github.com/EngReem85)
- LinkedIn: [LinkedIn](https://linkedin.com/in/www.linkedin.com/in/reem-algethami-245800316)
- Email: reem525979@gmail.com

---

## 📚 References

- [CPU Specifications Dataset – Kaggle](https://www.kaggle.com/datasets/lincolnzh/cpu-specifications-dataset)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Gradio Documentation](https://gradio.app/)
- [Power BI Documentation](https://powerbi.microsoft.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

##  Acknowledgments

- **LincolnZh** for the original CPU dataset on Kaggle
- **Open Source Community** for the amazing tools and libraries

---

## 📊 Sample Dashboard Screenshot


![Power BI Dashboard](./powerbi_dashboard.png)


---

## ⭐ Star the Project

If you found this project useful, please give it a ⭐ on GitHub!

---

**Built with ❤️ for the tech community**

---

**Tags:** `Machine Learning` `Data Science` `CPU` `Power BI` `Python` `XGBoost` `Gradio` `Hugging Face` `Data Visualization` `Predictive Analytics`


