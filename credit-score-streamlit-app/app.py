import os
import json
import zipfile
import tempfile
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
import pathlib
import joblib


# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="CreditScore AI | Juan Escobar",
    page_icon="💳",
    layout="wide"
)

# --- ESTILOS PERSONALIZADOS (CSS) ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border: none;
    }
    .card {
        padding: 1.5rem;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE SOPORTE ---
def get_categories(encoder):
    if hasattr(encoder, 'classes_'):
        return list(encoder.classes_)
    return []

def encode_value(encoder, value):
    try:
        return encoder.transform([value])[0]
    except:
        return 0

def _remove_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for k, v in obj.items():
            obj[k] = _remove_quantization_config(v)
        return obj
    if isinstance(obj, list):
        return [_remove_quantization_config(x) for x in obj]
    return obj

def load_model_compat(model_path: str):
    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixed_model_path = os.path.join(tmpdir, "modelo_fixed.keras")
            with zipfile.ZipFile(model_path, "r") as zin, zipfile.ZipFile(
                fixed_model_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == "config.json":
                        cfg = json.loads(data.decode("utf-8"))
                        cfg = _remove_quantization_config(cfg)
                        data = json.dumps(cfg, ensure_ascii=False).encode("utf-8")
                    zout.writestr(item, data)
            return load_model(fixed_model_path, compile=False)

@st.cache_resource
def load_artifacts():
    # Detect the directory where app.py is actually located
    base_path = pathlib.Path(__file__).parent.resolve()
    
    # 1. Load Model
    model_path = base_path / "models" / "modelo_riesgo_credito.keras"
    model = load_model_compat(str(model_path))
    
    # 2. Load Label Encoders
    encoders_path = base_path / "models" / "label_encoders.joblib"
    label_encoders = joblib.load(str(encoders_path))
    
    # 3. Load PCA (checking both possible names)
    pca = None
    pca_options = [
        base_path / "models" / "pca_components.joblib",
        base_path / "models" / "pca_8_componentes.joblib"
    ]
    
    for p in pca_options:
        if p.exists():
            pca = joblib.load(str(p))
            break
            
    return model, label_encoders, pca

# --- INICIALIZACIÓN ---
model, label_encoders, pca = load_artifacts()

# --- HEADER ---
with st.container():
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.title("🛡️ Credit Score Analyzer")
        st.markdown("### Sistema de Evaluación de Riesgo con Redes Neuronales (ANN)")
        st.info("Complete la información financiera del cliente para obtener una predicción instantánea.")
    with col_t2:
        st.image("https://images.unsplash.com/photo-1563013544-824ae1b704d3", use_container_width=True)

st.divider()

# --- FORMULARIO ---
FEATURES = ["Num_Cuentas_Bancarias", "Num_Tarjetas_Credito", "Tasa_Interes", "Num_Prestamos", 
            "Retraso_Desde_Vencimiento", "Num_Pagos_Retrasados", "Cambio_Limite_Credito", 
            "Num_Consultas_Credito", "Mezcla_Credito", "Deuda_Pendiente", 
            "Ratio_Utilizacion_Credito", "Antiguedad_Historial_Crediticio", 
            "Pago_Monto_Minimo", "Cuota_Mensual_Total", "Balance_Mensual"]

RANGOS = {
    "Num_Cuentas_Bancarias": (0.0, 10.5, 5.0),
    "Num_Tarjetas_Credito": (0.5, 10.8, 5.0),
    "Tasa_Interes": (1.0, 34.0, 13.0),
    "Num_Prestamos": (0.0, 9.0, 3.0),
    "Retraso_Desde_Vencimiento": (-2.0, 63.0, 17.0),
    "Num_Pagos_Retrasados": (0.0, 26.0, 13.0),
    "Cambio_Limite_Credito": (0.5, 31.0, 9.0),
    "Num_Consultas_Credito": (0.0, 16.0, 5.0),
    "Mezcla_Credito": (0.0, 2.0, 1.0),
    "Deuda_Pendiente": (0.2, 5000.0, 1100.0),
    "Ratio_Utilizacion_Credito": (3.0, 125.0, 45.0),
    "Antiguedad_Historial_Crediticio": (0.3, 33.0, 18.0),
    "Pago_Monto_Minimo": (0.0, 2.0, 2.0),
    "Cuota_Mensual_Total": (2.0, 130.0, 53.0),
    "Balance_Mensual": (92.0, 1349.0, 338.0),
}

input_features = {}

# Organización en 3 columnas para que no sea eterno el scroll
cols = st.columns(3)

for i, feature in enumerate(FEATURES):
    idx = i % 3
    with cols[idx]:
        # Estética de tarjeta simulada
        with st.expander(f"📌 {feature.replace('_', ' ')}", expanded=True):
            encoder = label_encoders.get(feature) if isinstance(label_encoders, dict) else None
            categories = get_categories(encoder) if encoder is not None else []

            if categories:
                input_features[feature] = st.selectbox("Seleccione:", categories, key=f"ui_{feature}")
            elif feature in RANGOS:
                mn, mx, dv = RANGOS[feature]
                input_features[feature] = st.slider("Ajuste:", float(mn), float(mx), float(dv), step=0.1, key=f"ui_{feature}")
            else:
                input_features[feature] = st.number_input("Valor:", value=0.0, key=f"ui_{feature}")

st.markdown("---")

# --- LÓGICA DE PREDICCIÓN ---
if st.button("📊 ANALIZAR PERFIL CREDITICIO"):
    with st.spinner('Procesando datos con la Red Neuronal...'):
        try:
            # Preprocesamiento
            row = {}
            for f in FEATURES:
                val = input_features[f]
                encoder = label_encoders.get(f) if isinstance(label_encoders, dict) else None
                row[f] = encode_value(encoder, val) if get_categories(encoder) else float(val)

            X = pd.DataFrame([row])
            expected_dim = int(model.input_shape[-1])

            if expected_dim == X.shape[1]:
                X_model = X.values
            elif expected_dim == 8 and pca is not None:
                X_model = pca.transform(X)
            else:
                st.error(f"Error de dimensiones: El modelo espera {expected_dim} variables.")
                st.stop()

            # Predicción
            pred = model.predict(X_model, verbose=0)
            score_idx = np.argmax(pred, axis=1)[0]
            confidence = np.max(pred) * 100

            # --- RESULTADOS VISUALES ---
            st.subheader("Resultado del Análisis")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.metric("Confianza del Modelo", f"{confidence:.2f}%")
                
            with res_col2:
                if score_idx == 2: # Bueno
                    st.markdown('<div class="result-card" style="background-color: #28a745;">✅ CRÉDITO BUENO (RIESGO BAJO)</div>', unsafe_allow_html=True)
                elif score_idx == 1: # Medio
                    st.markdown('<div class="result-card" style="background-color: #ffc107; color: black;">⚠️ CRÉDITO MEDIO (RIESGO MODERADO)</div>', unsafe_allow_html=True)
                else: # Malo
                    st.markdown('<div class="result-card" style="background-color: #dc3545;">🚨 CRÉDITO MALO (RIESGO ALTO)</div>', unsafe_allow_html=True)

            st.balloons() if score_idx == 2 else None

        except Exception as e:
            st.error(f"Ocurrió un error en el procesamiento: {e}")

# --- FOOTER ---
st.markdown("<br><hr><center>Desarrollado por <b>Juan Escobar</b> | 2026</center>", unsafe_allow_html=True)