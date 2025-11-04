# 🔬 Formation Energy Predictor

A web-based application for predicting the **formation energy per atom** of inorganic compounds using a trained deep learning model.  
Built with **Streamlit**, this tool allows users to upload composition and space group data or manually input chemical formulas to estimate the material's formation energy.

🌐 **Live App:** [Formation Energy Predictor](https://fenergypredictor-f9v7pnvx9rhzs6nistwsl8.streamlit.app/)

---

## 🚀 Features

- 🧠 Predicts **formation energy (eV/atom)** from:
  - Chemical formula (e.g., `FeO2`, `NiMgO`)
  - Space group number
- 📊 Accepts **CSV uploads** or **manual inputs**
- 🔍 Automatically applies normalization and feature extraction using **pymatgen** and **matminer**
- 📦 Deployed on **Streamlit Cloud**
- ⚙️ Built with **PyTorch**, **pandas**, and **NumPy**

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **ML Framework** | PyTorch |
| **Feature Engineering** | pymatgen, matminer |
| **Deployment** | Streamlit Community Cloud |

---

## 📁 Project Structure

