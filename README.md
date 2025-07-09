# Fintech Needs-Based Recommendation System

**Team 15 – Final Project**  
Mehdi Ghiasipour, Reza Ilzadi Jahromi, Hirdesh Kumar, Suraj Manickam Thirumalai Raja, Christos Partasidis  
**Date:** 25 June 2025

---

## 📌 Overview

This project builds a **needs-first recommendation system** for retail banking clients using interpretable machine learning. Instead of starting from financial products, we identify individual client needs — *Income vs Accumulation* — and match suitable products from a catalog using a transparent, explainable model.

---

## 🎯 Objectives

- Detect client investment needs (Income / Accumulation) based on demographic and financial data.
- Use interpretable decision trees to ensure explainability.
- Recommend a suitable product that adheres to client-specific risk tolerances.

---

## 🧩 Files

- **`Fintech_Final_Project_Estimating_Needs_V1_3.ipynb`**  
  Main implementation in Jupyter Notebook. Sections include:
  - Data ingestion and EDA
  - Feature engineering
  - Model training and validation (Decision Trees)
  - Explainability and Partial Dependence Plots (PDPs)
  - Next Best Action (NBA) logic for product recommendation

- **`Fintech_Final_Project_Team_15.pdf`**  
  Accompanying presentation deck summarizing the project:  
  - WHAT / WHY / HOW / RESULTS  
  - Visuals and metrics  
  - Regulatory context and business motivation

---

## 🧠 How It Works

### 1. Data

- **Rows:** 5000 anonymized retail bank clients  
- **Features:**  
  - Age, Gender, FamilyMembers, FinancialEducation, RiskPropensity, Income, Wealth  
- **Targets:**  
  - Binary flags: `IncomeInvestment`, `AccumulationInvestment`

### 2. Feature Engineering

- `Gender x Age` → life-stage modeling  
- `log(Income / Wealth)` → liquidity vs capital insight  
- Log-transform + MinMax scaling for skewed data  

### 3. Model

- Two separate **Decision Trees (depth=5)** for the two need targets  
- **5-fold cross-validation**  
- F1 score used as primary evaluation metric

### 4. Recommendation Logic (NBA)

- Predict each client’s needs
- Filter product catalog by need type
- Enforce risk suitability: `ProductRisk ≤ ClientRiskPropensity`
- Recommend the **highest-risk suitable product**

---

## 📊 Results

| Target                | Baseline F1 | F1 w/ Engineering | Δ F1  | Precision | Recall |
|-----------------------|-------------|-------------------|-------|-----------|--------|
| IncomeInvestment      | 0.689       | 0.729             | +0.04 | 0.77      | 0.79   |
| AccumulationInvestment| 0.680       | 0.818             | +0.14 | 0.70      | 0.72   |

- **Recommendation Coverage**  
  - IncomeInvestment: 94% (227/241 matched)  
  - AccumulationInvestment: 75% (357/475 matched)  
  - 0 risk cap violations (MiFID II compliant)

---

## 🛠️ How to Run

1. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
