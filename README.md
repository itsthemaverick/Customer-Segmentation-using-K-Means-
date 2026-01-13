# Customer Segmentation using K-Means (Engineering-Grade ML Project)

## Overview
This project demonstrates a **clean, script-based implementation of unsupervised learning** using **K-Means clustering**.  
It is intentionally built **without notebooks** to reflect real-world ML engineering practices.

The goal is to segment customers based on **behavioral and financial attributes** and produce **interpretable visual insights**.

---

## Why This Project Matters
Most beginner ML projects:
- Rely on notebooks
- Mix preprocessing, modeling, and plotting
- Guess the number of clusters

This project:
- Separates concerns across modules
- Uses metrics to justify decisions
- Saves reproducible artifacts to disk
- Mirrors how ML code is written in production environments

---

## Dataset
**Mall Customer Segmentation Dataset**

Features used:
- Age  
- Annual Income (k$)  
- Spending Score (1–100)

Excluded:
- CustomerID (identifier, no learning value)
- Gender (categorical, distance distortion in K-Means)

---

## Project Structure
```
customer-segmentation/
│
├── data/
│   └── Mall_Customers.csv
│
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── clustering.py
│   ├── metrics.py
│   └── visualize.py
│
├── visualizations/
│   ├── elbow.png
│   └── clusters.png
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Data Loading
Raw CSV data is loaded in isolation to maintain modularity and reusability.

### 2. Preprocessing
- Identifier columns removed
- Feature selection done explicitly
- Features scaled using StandardScaler (mandatory for distance-based algorithms)

### 3. Clustering
- K-Means applied on scaled features
- Deterministic behavior ensured using fixed random state

### 4. Model Justification
- **Elbow Method** used to observe diminishing returns
- **Silhouette Score** used to validate cluster separation

### 5. Visualization
- Elbow curve saved to disk
- Clusters visualized in 2D using PCA (visualization only)

---

## Results
- Optimal number of clusters chosen based on elbow curve inspection
- Silhouette score printed for validation
- Visual artifacts saved in `visualizations/`

---

## How to Run

### Install dependencies
```
pip install -r requirements.txt
```

### Execute pipeline
```
python main.py
```

---

## Engineering Principles Applied
- Single responsibility per module
- No hidden state or magic
- Human-in-the-loop decision making
- Script-first ML workflow
- Reproducible outputs

---

## Limitations
- K-Means assumes spherical clusters
- Sensitive to outliers
- Categorical features intentionally excluded

---

## Future Extensions
- KNN-based recommendations within clusters
- DBSCAN comparison
- Isolation Forest before clustering
- CLI arguments for full automation

---

## Author : itsthemaverick
Built with an ML-engineering-first mindset.
