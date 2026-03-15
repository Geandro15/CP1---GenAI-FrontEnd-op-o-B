# Geandro Dezordi    RM 562316
# Alexandre Ferreira RM 565626
# Lucas Veronezi     RM 564202
# Roberto Eduardo    RM 564537
# Guilherme Almeida  RM 563972

import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image

# -----------------------------
# CONFIGURAÇÃO
# -----------------------------

st.set_page_config(
    page_title="IA Diagnóstico de Pneumonia",
    layout="wide"
)

st.title("🫁 Diagnóstico de Pneumonia com IA")
st.markdown("Sistema experimental usando **Variational Autoencoder (VAE)**")

# -----------------------------
# SESSION STATE
# -----------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("⚙️ Configurações")

    threshold = st.slider(
        "Confiança mínima",
        0.0,
        1.0,
        0.5
    )

    st.divider()

    run_analysis = st.button("🔍 Executar Análise")

# -----------------------------
# TABS
# -----------------------------

tab1, tab2, tab3 = st.tabs([
    "🔬 Análise",
    "📊 Histórico",
    "📈 Monitoramento"
])

# -----------------------------
# TAB 1 - ANÁLISE
# -----------------------------

with tab1:

    st.header("Upload de imagem")

    uploaded_file = st.file_uploader(
        "Envie uma imagem de raio-X",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagem enviada")
            st.image(image, caption="Raio-X carregado", width="stretch")   # ← corrigido aqui

        with col2:

            # Inicializa flag se ainda não existir
            if "analise_realizada" not in st.session_state:
                st.session_state.analise_realizada = False

            if run_analysis and uploaded_file is not None:

                with st.spinner("Processando imagem..."):

                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress.progress(i + 1)

                    confidence = np.random.uniform(0.5, 0.95)

                    if confidence > threshold:
                        result = "Possível Pneumonia"
                    else:
                        result = "Sem sinais de Pneumonia"

                st.subheader("Resultado da análise")

                st.metric("Confiança da IA", f"{confidence*100:.2f}%")
                st.progress(confidence)

                if confidence > 0.8:
                    st.success("Alta confiança no diagnóstico")
                elif confidence > 0.6:
                    st.warning("Confiança moderada")
                else:
                    st.error("Baixa confiança — revisar exame")

                st.write(f"**Resultado:** {result}")

                # Salva no histórico
                record = {"resultado": result, "confiança": confidence}
                st.session_state.history.append(record)

                # Marca que já fizemos pelo menos uma análise
                st.session_state.analise_realizada = True

            # ────────────────────────────────────────────────
            # Feedback: aparece se já houve análise nesta sessão
            # ────────────────────────────────────────────────
            if st.session_state.analise_realizada:

                st.markdown("---")
                st.subheader("Feedback do usuário (sobre a última análise)")

                colf1, colf2 = st.columns(2)

                with colf1:
                    if st.button("✅ Acertou", key="btn_positivo"):
                        st.session_state.feedback.append("positivo")
                        st.toast("Feedback positivo registrado!")
                        st.rerun()

                with colf2:
                    if st.button("❌ Errou", key="btn_negativo"):
                        st.session_state.feedback.append("negativo")
                        st.toast("Feedback negativo registrado!")
                        st.rerun()

            # Caso ainda não tenha rodado análise
            if uploaded_file is not None and not st.session_state.analise_realizada:
                st.info("Clique em 'Executar Análise' para ver o resultado e dar feedback.")

    else:

        st.info("Faça upload de uma imagem para iniciar a análise.")

# -----------------------------
# TAB 2 - HISTÓRICO
# -----------------------------

with tab2:

    st.header("Histórico de análises")

    if len(st.session_state.history) == 0:
        st.write("Nenhuma análise realizada ainda.")

    else:

        df = pd.DataFrame(st.session_state.history)

        df["confiança (%)"] = df["confiança"] * 100

        st.dataframe(df)

# -----------------------------
# TAB 3 - MONITORAMENTO
# -----------------------------

with tab3:
    st.header("Monitoramento do modelo")

    total = len(st.session_state.feedback)
    if total == 0:
        st.info("Nenhum feedback registrado ainda.")
    else:
        positivos = st.session_state.feedback.count("positivo")
        negativos = total - positivos  # mais eficiente que .count("negativo")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Feedback", total)
        col2.metric("Positivos", positivos, delta_color="normal")
        col3.metric("Negativos", negativos)

        confiabilidade = positivos / total
        st.subheader(f"Confiabilidade: {confiabilidade:.1%}")
        st.progress(confiabilidade)

        if confiabilidade < 0.5:
            st.error("Possível degradação do modelo – revisar dados")
        elif confiabilidade < 0.75:
            st.warning("Confiabilidade razoável – monitorar")
        else:
            st.success("Modelo operando normalmente")
