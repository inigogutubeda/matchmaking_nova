import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Mentorship Matching", layout="wide")

# Cargar modelo de embeddings
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Función para procesar los datos
def process_data(uploaded_file):
    # Cargar el archivo Excel
    xls = pd.ExcelFile(uploaded_file)
    
    # Cargar hojas de mentores y mentees
    df_mentors = xls.parse("Mentors data updated").fillna("")
    df_mentees = xls.parse("Mentee data updated").fillna("")
    
    # Convertir experiencia laboral a float
    df_mentors["years_of_experience"] = pd.to_numeric(df_mentors["years_of_experience"], errors="coerce").fillna(0.0)
    df_mentees["years_of_experience"] = pd.to_numeric(df_mentees["years_of_experience"], errors="coerce").fillna(0.0)
    
    return df_mentors, df_mentees

# Función para generar matches
def generate_matches(df_mentors, df_mentees):
    # Generar embeddings de las biografías
    mentor_embeddings = model.encode(df_mentors["mentor_biography"].astype(str).tolist(), convert_to_tensor=True)
    mentee_embeddings = model.encode(df_mentees["mentee_biography"].astype(str).tolist(), convert_to_tensor=True)

    # Calcular similitud de coseno
    similarity_matrix = cosine_similarity(mentee_embeddings.cpu().numpy(), mentor_embeddings.cpu().numpy())

    # Obtener los mejores mentores para cada mentee
    best_matches = np.argmax(similarity_matrix, axis=1)
    best_scores = np.max(similarity_matrix, axis=1)

    # Asignar mentor al mentee
    df_mentees["Best Mentor"] = [df_mentors.iloc[idx]["Name"] for idx in best_matches]
    df_mentees["Best Mentor Email"] = [df_mentors.iloc[idx]["Email"] for idx in best_matches]
    df_mentees["Best Mentor LinkedIn"] = [df_mentors.iloc[idx]["linkedin"] for idx in best_matches]
    df_mentees["Matching Score (Embeddings)"] = best_scores

    # Ajustar score final considerando industria, idioma y experiencia
    def calculate_final_score(row):
        mentor_idx = best_matches[row.name]
        mentor = df_mentors.iloc[mentor_idx]

        final_score = row["Matching Score (Embeddings)"]

        # Ajuste por industria (0.2 si coinciden)
        if row["current_industry"] == mentor["current_industry"]:
            final_score += 0.2

        # Ajuste por idioma (0.15 si tienen un idioma fluido en común)
        mentee_languages = set(row["languages"].split(";;"))
        mentor_languages = set(mentor["languages"].split(";;"))

        common_languages = mentee_languages.intersection(mentor_languages)
        fluent_or_native = any("Fluent" in lang or "Native" in lang for lang in common_languages)

        if fluent_or_native:
            final_score += 0.15

        # Ajuste por experiencia (0.2 si el mentor tiene 5+ años más)
        if mentor["years_of_experience"] - row["years_of_experience"] >= 5:
            final_score += 0.2

        return final_score

    df_mentees["Final Matching Score"] = df_mentees.apply(calculate_final_score, axis=1)

    return df_mentees

# Función para mostrar resultados y permitir descarga
def display_results(df_mentees):
    # Explicación del match
    def generate_explanation(row):
        explanation = f"Recomendado a {row['Best Mentor']} basado en:"
        if row["Matching Score (Embeddings)"] > 0.5:
            explanation += " ↪ Similitud en biografía."
        if row["current_industry"] == df_mentors.iloc[best_matches[row.name]]["current_industry"]:
            explanation += " ↪ Industria compatible."
        if df_mentors.iloc[best_matches[row.name]]["years_of_experience"] - row["years_of_experience"] >= 5:
            explanation += " ↪ Diferencia de experiencia adecuada."
        return explanation

    df_mentees["Match Explanation"] = df_mentees.apply(generate_explanation, axis=1)

    # Mostrar tabla de resultados
    st.write("### Resultados de Matching")
    st.dataframe(df_mentees[[
        "Name", "Mentee LinkedIn", "Best Mentor", "Best Mentor Email", "Best Mentor LinkedIn",
        "Matching Score (Embeddings)", "Final Matching Score", "Match Explanation"
    ]])

    # Descargar resultados en Excel
    output_excel_path = "mentoring_recommendations.xlsx"
    df_mentees.to_excel(output_excel_path, index=False)
    with open(output_excel_path, "rb") as file:
        st.download_button(label="📥 Descargar Resultados en Excel", data=file, file_name="mentoring_recommendations.xlsx")

# Interfaz en Streamlit
st.title("📌 Sistema de Recomendación de Mentoría")

uploaded_file = st.file_uploader("📂 Sube un archivo Excel con datos de mentores y mentees", type=["xlsx"])

if uploaded_file:
    df_mentors, df_mentees = process_data(uploaded_file)
    df_mentees = generate_matches(df_mentors, df_mentees)
    display_results(df_mentees)
