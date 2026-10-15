# 🛒 Big Mart Sales Prediction

Predicting the sales of each product at each Big Mart outlet using an **XGBoost regression** model, built end to end in a Kaggle notebook: data cleaning, exploratory data analysis, preprocessing, model training and evaluation.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange) ![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-yellow) ![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF)

---

## 📌 Problem Statement

Big Mart has collected 2013 sales data for 1,559 products across 10 stores in different cities. The goal is to build a regression model that predicts **`Item_Outlet_Sales`** (sales of a product in a given store), so the business can understand which product and store attributes drive sales.

## 📂 Dataset

- **Source:** [BigMart Sales Data on Kaggle](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data) (`Train.csv`)
- **Size:** 8,523 rows × 12 columns
- **Target:** `Item_Outlet_Sales`

| Feature | Description |
|---|---|
| `Item_Identifier` | Unique product ID |
| `Item_Weight` | Weight of the product |
| `Item_Fat_Content` | Low Fat / Regular |
| `Item_Visibility` | % of store display area allocated to the product |
| `Item_Type` | Product category (Dairy, Snack Foods, etc.) |
| `Item_MRP` | Maximum retail price |
| `Outlet_Identifier` | Unique store ID |
| `Outlet_Establishment_Year` | Year the store opened |
| `Outlet_Size` | Small / Medium / High |
| `Outlet_Location_Type` | Tier 1 / Tier 2 / Tier 3 city |
| `Outlet_Type` | Grocery store or Supermarket type 1/2/3 |
| `Item_Outlet_Sales` | **Target** – sales of the product in the store |

## 🔄 Project Workflow

1. **Data loading & inspection** – `shape`, `info()`, `describe()`, missing-value check
2. **Handling missing values**
   - `Item_Weight` → filled with the column **mean**
   - `Outlet_Size` → filled with the **mode for each `Outlet_Type`** (via a pivot table)
3. **Exploratory Data Analysis** (seaborn / matplotlib)
   - Distributions of `Item_Weight`, `Item_Visibility`, `Item_MRP`, `Item_Outlet_Sales`
   - Count plots of `Outlet_Establishment_Year`, `Item_Fat_Content`, `Item_Type`, `Outlet_Size`
4. **Data cleaning** – standardised inconsistent `Item_Fat_Content` labels (`low fat`, `LF` → `Low Fat`; `reg` → `Regular`)
5. **Label encoding** of all 7 categorical columns with `LabelEncoder`
6. **Train/test split** – 80% / 20% (`random_state=2`)
7. **Model training** – `XGBRegressor` (default hyperparameters)
8. **Evaluation** – R² score on training and test data

## 📊 Results

| Dataset | R² Score |
|---|---|
| Training | **0.875** |
| Test | **0.516** |

**Observation:** The gap between training and test R² shows the model is **overfitting**. See the next steps below for planned improvements.

## 🔍 Key EDA Insights

- `Item_Visibility` is right-skewed, and many items have a visibility of **0**, which is unrealistic and likely a data issue.
- `Item_Outlet_Sales` is right-skewed: most products sell in low amounts, with a few high sellers.
- `Item_Fat_Content` had inconsistent labels for the same category, which were merged.
- The spike in the `Item_Weight` histogram comes from filling missing values with the mean.

## 🚀 Future Improvements

- [ ] Tune XGBoost hyperparameters (`max_depth`, `learning_rate`, `n_estimators`, regularisation) with cross-validation to reduce overfitting
- [ ] Treat zero `Item_Visibility` values as missing and impute them
- [ ] Feature engineering: outlet age (instead of establishment year), item category from `Item_Identifier` prefix (FD / DR / NC)
- [ ] One-hot encode nominal features instead of label encoding
- [ ] Compare with other models (Linear Regression, Random Forest, LightGBM) and use RMSE alongside R²
- [ ] Predict on `Test.csv` and build a simple prediction app

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost
- **Environment:** Kaggle Notebooks

## ▶️ How to Run

```bash
git clone https://github.com/GuninderSinghBhatia/big-mart-sales-prediction.git
cd big-mart-sales-prediction
pip install -r requirements.txt
jupyter notebook big-mart-sales-prediction.ipynb
```

> Download `Train.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data) and update the path in the `pd.read_csv(...)` cell if you run it outside Kaggle.

## 📁 Repository Structure

```
big-mart-sales-prediction/
├── big-mart-sales-prediction.ipynb   # Full notebook: EDA, preprocessing, model, evaluation
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## 👤 Author

**Guninder Singh Bhatia**
- GitHub: [@GuninderSinghBhatia](https://github.com/GuninderSinghBhatia)
- Kaggle: [guninder07](https://www.kaggle.com/guninder07)
