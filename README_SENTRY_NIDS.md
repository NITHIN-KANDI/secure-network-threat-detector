
# 🛡️ SENTRY: Secure Ensemble-Based Network Threat Recognition System

## 📘 Overview

**SENTRY** is an advanced network intrusion detection system designed to identify, classify, and securely communicate network threats. It employs an ensemble of powerful machine learning models, combined with cryptographic and steganographic techniques, to ensure reliable detection and privacy-preserving communication of threat predictions.

---

## 🚀 Key Features

- **🔍 Accurate Intrusion Detection** using:
  - Random Forest
  - XGBoost
  - LightGBM
  - Stacking Classifiers

- **🔐 Prediction Security** through:
  - Hybrid **RSA/AES encryption**
  - Emoji-based encoding
  - Fractal-based steganography using Mandelbrot pixel embedding

- **📊 Dimensionality Reduction**: PCA for better generalization
- **🔄 Data Integrity**: Row-level hashing & digital signatures
- **🖥️ User Interface**: Built with **Streamlit** for real-time prediction and secure output delivery

---

## 🧪 Dataset

- **UNSW-NB15**: A modern benchmark dataset used for training and evaluation.
- Includes both **binary** and **multiclass** classification of threats.

---

## 🧰 Technologies Used

- Python
- Scikit-learn
- LightGBM
- XGBoost
- Streamlit
- RSA/AES encryption (PyCryptodome)
- Custom steganography with Mandelbrot embedding
- Pandas, NumPy, Matplotlib, and Seaborn

---

## 📂 Project Structure

```
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for model experimentation
├── src/                   # Source code for model, pipeline, crypto, stego
├── attack_logs/            # Logs for threats, predictions, encryption
├── Requirements.txt        # List of required dependencies
├── README.md               # Project documentation
```

---

## 🖥️ How to Run

### Install Dependencies:
```bash
pip install -r Requirements.txt
```

### Launch Streamlit App:
```bash
streamlit run src/app.py
```

---

## 📊 Evaluation Results

- **Accuracy**: > 95% on UNSW-NB15 test set
- **Secure Output Handling**:
  - Emoji encoding via RSA-encrypted AES key
  - Mandelbrot-based fractal steganography for image embedding

---

## 🔐 Security Additions

- **Row-level Hashing + Digital Signatures**: Ensures data integrity
- **AES + RSA Encryption**: Hybrid approach for safe output storage
- **Emoji Encoder**: Makes encrypted output readable and shareable
- **Fractal Steganography**: Encodes output within Mandelbrot-generated images

---

## 🧑‍💻 Authors

- **Nithin Kandi**  
- **Amrita Vishwa Vidyapeetham**  
- 2025

---

## 📄 License

This project is released for research and educational use only.
