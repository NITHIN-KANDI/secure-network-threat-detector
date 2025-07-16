import os
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import random
import json
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Util.Padding import pad, unpad
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
import base64

# ─── CONFIG ──────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\kandi\Documents\sem_6\computer_security\Project_cs\IDS-binary-classification"
RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "UNSW_NB15_training-set.csv")
PREPROCESSOR = os.path.join(BASE_DIR, "data", "processed", "preprocessor.pkl")
PCA_MODEL = os.path.join(BASE_DIR, "data", "processed", "pca.pkl")
MC_MODELS_DIR = os.path.join(BASE_DIR, "src", "multiclass_models_dummy")
MODEL_FILES = {
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl",
    "LightGBM": "lgb_model.pkl",
    "CatBoost": "cb_model.pkl",
    "Stacking Ensemble": "stack_model.pkl"
}
LABEL_ENCODER = os.path.join(MC_MODELS_DIR, "attack_label_encoder.pkl")
ATTACK_LOGS_FILE = os.path.join(BASE_DIR, "attack_logs.xlsx")

RSA_PRIVATE_KEY_PATH = os.path.join(BASE_DIR, "src", "gui", "rsa_private.pem")
RSA_PUBLIC_KEY_PATH = os.path.join(BASE_DIR, "src", "gui", "rsa_public.pem")

# Load dataset schema and preprocess info
schema_df = pd.read_csv(RAW_CSV)
feature_cols = [c for c in schema_df.columns if c not in ("label", "attack_cat", "row_hash")]

numeric_ranges = {}
numeric_is_integer = {}
categorical_options = {}
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(schema_df[col]):
        numeric_ranges[col] = (schema_df[col].min(), schema_df[col].max())
        numeric_is_integer[col] = schema_df[col].dtype.kind in "iu"
    else:
        categorical_options[col] = schema_df[col].dropna().unique().tolist()

# Emoji encoding map
emoji_map = {
    'A': '😀', 'B': '😁', 'C': '😂', 'D': '🤣', 'E': '😃', 'F': '😄', 'G': '😅', 'H': '😆',
    'I': '😉', 'J': '😊', 'K': '😋', 'L': '😎', 'M': '😍', 'N': '😘', 'O': '😗', 'P': '😙',
    'Q': '😚', 'R': '🙂', 'S': '🤗', 'T': '🤩', 'U': '🤔', 'V': '🤨', 'W': '😐', 'X': '😑',
    'Y': '😶', 'Z': '🙄', 'a': '😏', 'b': '😣', 'c': '😥', 'd': '😮', 'e': '🤐', 'f': '😯',
    'g': '😪', 'h': '😫', 'i': '😴', 'j': '😌', 'k': '😛', 'l': '😜', 'm': '😝', 'n': '🤤',
    'o': '😒', 'p': '😓', 'q': '😔', 'r': '😕', 's': '🙃', 't': '🤑', 'u': '😲', 'v': '☹️',
    'w': '🙁', 'x': '😖', 'y': '😞', 'z': '😟', '0': '😤', '1': '😢', '2': '😭', '3': '😦',
    '4': '😧', '5': '😨', '6': '😩', '7': '🤯', '8': '😬', '9': '😰', '+': '😱', '/': '😳', '=': '🤪'
}
inv_emoji_map = {v: k for k, v in emoji_map.items()}

def emoji_encode(b64_str):
    return ''.join(emoji_map.get(c, c) for c in b64_str)

def emoji_decode(emoji_str):
    decoded = ""
    i = 0
    while i < len(emoji_str):
        if i + 1 < len(emoji_str) and emoji_str[i:i+2] in inv_emoji_map:
            decoded += inv_emoji_map[emoji_str[i:i+2]]
            i += 2
        elif emoji_str[i] in inv_emoji_map:
            decoded += inv_emoji_map[emoji_str[i]]
            i += 1
        else:
            i += 1
    return decoded

def aes_encrypt(plaintext, key_bytes):
    cipher = AES.new(key_bytes, AES.MODE_CBC)
    ct = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    return base64.b64encode(cipher.iv + ct).decode()

def aes_decrypt(enc_b64, key_bytes):
    raw = base64.b64decode(enc_b64)
    iv = raw[:16]
    ct = raw[16:]
    cipher = AES.new(key_bytes, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size)
    return pt.decode()

def rsa_encrypt_key(aes_key, rsa_pub_key_pem):
    rsa_pub = RSA.import_key(rsa_pub_key_pem)
    cipher_rsa = PKCS1_OAEP.new(rsa_pub)
    enc_key = cipher_rsa.encrypt(aes_key)
    return base64.b64encode(enc_key).decode()

def rsa_decrypt_key(enc_key_b64, rsa_priv_key_pem):
    rsa_priv = RSA.import_key(rsa_priv_key_pem)
    cipher_rsa = PKCS1_OAEP.new(rsa_priv)
    enc_key = base64.b64decode(enc_key_b64)
    aes_key = cipher_rsa.decrypt(enc_key)
    return aes_key

# Load RSA keys
try:
    with open(RSA_PRIVATE_KEY_PATH, "rb") as f:
        rsa_private_key_pem = f.read()
    with open(RSA_PUBLIC_KEY_PATH, "rb") as f:
        rsa_public_key_pem = f.read()
except Exception as e:
    st.error(f"Failed to load RSA keys: {e}")
    st.stop()

# Copy button helper with streamlit components and dark style
def copy_button(text, button_text="Copy"):
    escaped_text = text.replace('"', '\\"').replace('\n', '\\n')
    html_code = f"""
    <style>
        .copy-container {{
            display: flex;
            align-items: center;
            gap: 8px;
            background-color: #1e1e2f;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 20px;
            overflow-x: auto;
            white-space: nowrap;
            user-select: text;
            color: #fff;
        }}
        .copy-input {{
            flex-grow: 1;
            background-color: transparent;
            border: none;
            outline: none;
            color: #fff;
            font-size: 20px;
            font-family: 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji', sans-serif;
            cursor: text;
        }}
        .copy-button {{
            background-color: #4caf50;
            border: none;
            color: white;
            padding: 8px 16px;
            font-size: 14px;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }}
        .copy-button:hover {{
            background-color: #45a049;
        }}
    </style>
    <div class="copy-container">
        <input type="text" class="copy-input" value="{escaped_text}" id="copyText" readonly />
        <button class="copy-button" onclick="navigator.clipboard.writeText(document.getElementById('copyText').value)"> {button_text} </button>
    </div>
    """
    components.html(html_code, height=50)


# Streamlit UI setup
st.set_page_config(page_title="IDS Multiclass Predictor", layout="centered")
st.title("IDS Attack-Type Prediction with Hybrid RSA+AES Encryption")

model_choice = st.selectbox("Choose multiclass model", list(MODEL_FILES.keys()))

if st.button("Random Sample"):
    for col in feature_cols:
        if col in numeric_ranges:
            lo, hi = numeric_ranges[col]
            if numeric_is_integer[col]:
                st.session_state[col] = random.randint(int(lo), int(hi))
            else:
                st.session_state[col] = random.uniform(lo, hi)
        else:
            st.session_state[col] = random.choice(categorical_options[col])

st.write("#### Enter features:")
inputs = {}
for col in feature_cols:
    key_ui = f"input_{col}"
    if col in numeric_ranges:
        lo, hi = numeric_ranges[col]
        if numeric_is_integer[col]:
            default = st.session_state.get(col, int((lo + hi) // 2))
            val = st.number_input(col, min_value=int(lo), max_value=int(hi), value=default, key=key_ui)
        else:
            default = st.session_state.get(col, float((lo + hi) / 2))
            val = st.number_input(col, min_value=float(lo), max_value=float(hi), value=default, key=key_ui)
    else:
        opts = categorical_options[col]
        default = st.session_state.get(col, opts[0])
        val = st.selectbox(col, opts, index=opts.index(default), key=key_ui)
    inputs[col] = val

if st.button("Predict and Encrypt (Hybrid RSA+AES)"):
    sample_df = pd.DataFrame([inputs])

    try:
        preproc = joblib.load(PREPROCESSOR)
        Xp = preproc.transform(sample_df)
        pca = joblib.load(PCA_MODEL)
        Xp = pca.transform(Xp)
        le = joblib.load(LABEL_ENCODER)
        model = joblib.load(os.path.join(MC_MODELS_DIR, MODEL_FILES[model_choice]))
    except Exception as e:
        st.error(f"Loading error: {e}")
        st.stop()

    try:
        proba = model.predict_proba(Xp)[0]
        idx = int(np.argmax(proba))
        attack = le.inverse_transform([idx])[0]
        prob = proba[idx]
    except Exception as e:
        st.error(f"Inference error: {e}")
        st.stop()

    summary = {
        "input_features": inputs,
        "model_used": model_choice,
        "predicted_attack": attack,
        "confidence": prob
    }

    try:
        json_str = json.dumps(summary)

        # Generate fresh random AES key
        aes_key = get_random_bytes(16)

        # Encrypt summary with AES key
        encrypted_data_b64 = aes_encrypt(json_str, aes_key)
        encrypted_data_emoji = emoji_encode(encrypted_data_b64)

        # Encrypt AES key with RSA public key
        encrypted_aes_key_b64 = rsa_encrypt_key(aes_key, rsa_public_key_pem)
        encrypted_aes_key_emoji = emoji_encode(encrypted_aes_key_b64)

        # Save to Excel
        new_log = pd.DataFrame([{
            "encrypted_data_emoji": encrypted_data_emoji,
            "encrypted_aes_key_emoji": encrypted_aes_key_emoji,
            "model_used": model_choice
        }])
        if os.path.exists(ATTACK_LOGS_FILE):
            df_logs = pd.read_excel(ATTACK_LOGS_FILE)
            df_logs = pd.concat([df_logs, new_log], ignore_index=True)
        else:
            df_logs = new_log
        df_logs.to_excel(ATTACK_LOGS_FILE, index=False)

        st.success("Prediction encrypted and saved (hybrid RSA+AES) to attack_logs.xlsx")

        st.write("Encrypted AES Key (emoji):")
        copy_button(encrypted_aes_key_emoji)

        st.write("Encrypted Data (emoji):")
        copy_button(encrypted_data_emoji)

    except Exception as e:
        st.error(f"Encryption or saving failed: {e}")
        st.stop()

st.write("---")
st.subheader("Decrypt Hybrid RSA+AES Encrypted Data")

enc_key_emoji = st.text_area("Paste Encrypted AES Key (emoji):", height=100)
enc_data_emoji = st.text_area("Paste Encrypted Data (emoji):", height=200)

if st.button("Decrypt Data"):
    if not enc_key_emoji.strip() or not enc_data_emoji.strip():
        st.warning("Please paste both the encrypted AES key and encrypted data.")
    else:
        try:
            enc_key_b64 = emoji_decode(enc_key_emoji.strip())
            enc_data_b64 = emoji_decode(enc_data_emoji.strip())

            aes_key = rsa_decrypt_key(enc_key_b64, rsa_private_key_pem)
            decrypted_json = aes_decrypt(enc_data_b64, aes_key)
            obj = json.loads(decrypted_json)

            st.markdown("### Decrypted Input Features:")
            st.json(obj.get("input_features", {}))
            st.markdown("### Prediction Summary:")
            st.write(f"Model Used: {obj.get('model_used', 'N/A')}")
            st.write(f"Predicted Attack: {obj.get('predicted_attack', 'N/A')}")
            st.write(f"Confidence: {obj.get('confidence', 'N/A'):.4f}")
        except Exception as e:
            st.error(f"Decryption failed: {e}")
