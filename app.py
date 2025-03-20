import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuración de la página y estilos
st.set_page_config(page_title="Mentorship Matching", page_icon="🎯", layout="wide")

# Estilo personalizado
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #F7F9FC;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stDownloadButton>button {
            background-color: #007BFF;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stDataFrame {
            border: 2px solid #4CAF50;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Cargar modelo de embeddings
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Mostrar logo y título
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.image("nova.png", width=120)
with col2:
    st.title("🎯 Sistema de Recomendación de Mentoría")

st.markdown("### 🔹 Encuentra el mejor mentor basado en habilidades, experiencia y afinidad de industria.")

# Función para procesar datos
def process_data(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    df_mentors = xls.parse("Mentors data updated").fillna("")
    df_mentees = xls.parse("Mentee data updated").fillna("")
    
    df_mentors["years_of_experience"] = pd.to_numeric(df_mentors["years_of_experience"], errors="coerce").fillna(0.0)
    df_mentees["years_of_experience"] = pd.to_numeric(df_mentees["years_of_experience"], errors="coerce").fillna(0.0)
    
    return df_mentors, df_mentees

# Función para generar matches
def generate_matches(df_mentors, df_mentees):
    mentor_embeddings = model.encode(df_mentors["mentor_biography"].astype(str).tolist(), convert_to_tensor=True)
    mentee_embeddings = model.encode(df_mentees["mentee_biography"].astype(str).tolist(), convert_to_tensor=True)
    
    similarity_matrix = cosine_similarity(mentee_embeddings.cpu().numpy(), mentor_embeddings.cpu().numpy())

    best_matches = np.argmax(similarity_matrix, axis=1)
    best_scores = np.max(similarity_matrix, axis=1)

    df_mentees["Best Mentor"] = [df_mentors.iloc[idx]["Name"] for idx in best_matches]
    df_mentees["Best Mentor Email"] = [df_mentors.iloc[idx]["Email"] for idx in best_matches]
    df_mentees["Best Mentor LinkedIn"] = [df_mentors.iloc[idx]["linkedin"] for idx in best_matches]
    df_mentees["Matching Score (Embeddings)"] = best_scores

    def calculate_final_score(row):
        mentor_idx = best_matches[row.name]
        mentor = df_mentors.iloc[mentor_idx]

        final_score = row["Matching Score (Embeddings)"]

        if row["current_industry"] == mentor["current_industry"]:
            final_score += 0.2

        mentee_languages = set(row["languages"].split(";;"))
        mentor_languages = set(mentor["languages"].split(";;"))
        common_languages = mentee_languages.intersection(mentor_languages)
        fluent_or_native = any("Fluent" in lang or "Native" in lang for lang in common_languages)

        if fluent_or_native:
            final_score += 0.15

        if mentor["years_of_experience"] - row["years_of_experience"] >= 5:
            final_score += 0.2

        return final_score

    df_mentees["Final Matching Score"] = df_mentees.apply(calculate_final_score, axis=1)

    return df_mentees, best_matches

# Función para mostrar resultados
def display_results(df_mentees, df_mentors, best_matches):
    def generate_explanation(row):
        mentor_idx = best_matches[row.name]
        mentor = df_mentors.iloc[mentor_idx]
        
        explanation = f"Recomendado a {row['Best Mentor']} basado en:"
        if row["Matching Score (Embeddings)"] > 0.5:
            explanation += " ↪ Similitud en biografía."
        if row["current_industry"] == mentor["current_industry"]:
            explanation += " ↪ Industria compatible."
        if mentor["years_of_experience"] - row["years_of_experience"] >= 5:
            explanation += " ↪ Diferencia de experiencia adecuada."

        return explanation

    df_mentees["Match Explanation"] = df_mentees.apply(generate_explanation, axis=1)

    st.markdown("## 🔹 Resultados del Matching")
    st.dataframe(df_mentees[[
        "Name", "linkedin", "Best Mentor", "Best Mentor Email", "Best Mentor LinkedIn",
        "Matching Score (Embeddings)", "Final Matching Score", "Match Explanation"
    ]])

    output_excel_path = "mentoring_recommendations.xlsx"
    df_mentees.to_excel(output_excel_path, index=False)
    with open(output_excel_path, "rb") as file:
        st.download_button(label="📥 Descargar Resultados en Excel", data=file, file_name="mentoring_recommendations.xlsx")

# Cargar archivos
uploaded_file = st.file_uploader("📂 Sube un archivo Excel con datos de mentores y mentees", type=["xlsx"])

if uploaded_file:
    df_mentors, df_mentees = process_data(uploaded_file)
    df_mentees, best_matches = generate_matches(df_mentors, df_mentees)
    display_results(df_mentees, df_mentors, best_matches)
