import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    st.title("Clinical Decision Support System")
    
    st.image("imagen_cabezera.jpg", caption="Fuente: Unsplash", use_column_width=True)
    
    st.write("Upload your CSV file with metabolite concentrations.")
    
    # Cargar archivo CSV
    file = st.file_uploader("load CSV file", type=["csv"])

    if file is not None:
        st.write("File uploaded successfully. Showing first rows:")
        data = pd.read_csv(file, sep=';')
        st.write(data.head())
        
        # Preprocesar los datos cargados
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed 0'])
        metabolite_columns = [f'METABOLITE {i}' for i in range(15)]
        data[metabolite_columns] = data[metabolite_columns].replace(',', '.', regex=True)
        data[metabolite_columns] = data[metabolite_columns].apply(pd.to_numeric, errors='coerce')

        st.markdown("<h5 style='text-align: center; color: black;'>Data after preprocessing</h5>",unsafe_allow_html=True)
        st.write(data.head())
        
        # ESCALADO
        scaler = RobustScaler(quantile_range=(2.5, 97.5))
        data_scaled = scaler.fit_transform(data[metabolite_columns])
        
        # Cargar el modelo previamente entrenado
        # Debes haber entrenado y guardado el modelo previamente, y aquí lo cargamos
        mlp_final = joblib.load('modelo_mlp_final_1.pkl')
        
        # Predecir usando el modelo entrenado
        y_pred_prob = mlp_final.predict_proba(data_scaled)  # Obtener probabilidades
        
        # Asignar los nombres de las clases según los valores numéricos
        class_names = ['MENINGIOMA', 'ASTROCYTOMA', 'GLIOBLASTOMA']
        
        # Crear un DataFrame con las probabilidades
        prob_df = pd.DataFrame(y_pred_prob, columns=[f"Prob_{name}" for name in class_names])
        
        # Añadir la predicción con la mayor probabilidad (diagnóstico recomendado)
        prob_df['Diagnóstico Recomendado'] = prob_df.idxmax(axis=1).str.replace('Prob_', '')
        
        
        st.markdown("<h3 style='text-align: center; color: black;'>Results with probabilities and recommended diagnosis:</h3>", unsafe_allow_html=True)
        st.write("Click on the button below to download the probability file")
        st.write(prob_df.head())
        

        # Descargar archivo CSV con las probabilidades y el diagnóstico recomendado
        csv_data = prob_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV with probabilities and diagnosis",
            data=csv_data,
            file_name="predicciones_diagnostico.csv",
            mime="text/csv"
        )
        
        
        # Análisis Exploratorio de Datos (AED)
        # Cambiar el tamaño de la letra para el título del AED
        st.markdown("<h2 style='text-align: center; color: black;'>Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)
        
        # Mostrar estadísticas descriptivas
        st.write("Estadísticas descriptivas de los metabolitos:")
        st.write(data[metabolite_columns].describe())
        
        
        # Calcular y mostrar la matriz de correlación
        st.markdown("<h5 color: black;'>Some graphics:</h5>", unsafe_allow_html=True)
        corr_matrix = data[metabolite_columns].corr()
        
        # Graficar la matriz de correlación
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        plt.title('Metabolite Correlation Matrix')
        
        # Mostrar la matriz de correlación en Streamlit
        st.pyplot(fig)
        
        st.markdown("""
                    - **Strong correlations** (positive or negative) indicate that the metabolites might be involved in similar processes or be part of the same metabolic pathway. In some cases, a high correlation might suggest redundancy in the information provided by these metabolites.
  
                    - **Correlations close to zero** suggest that there is no clear relationship between these metabolites in your dataset.
                    """)
        
        # Mantener el gráfico original pero sin las etiquetas en el eje X
        fig, ax = plt.subplots()
        prob_df.drop(columns=['Diagnóstico Recomendado']).plot(kind='bar', stacked=True, ax=ax)
        plt.title('Distribution of Probabilities by Diagnosis')
        plt.ylabel('Probability')
        plt.xlabel('Samples')
        ax.set_xticks([])  # Quitar las etiquetas del eje X
        
        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
        
        st.markdown("""
                    - **Samples with bars completely filled with a single color**: These show high certainty in predicting that class. For example, if a bar is completely green, it means the model predicted a probability close to 100% for **GLIOBLASTOMA**.
  
                    - **Bars divided into multiple colors**: This indicates that the model is not fully confident and assigns probabilities to more than one diagnosis. For example, if a bar has significant portions of blue and orange, the model predicts that the sample is likely **MENINGIOMA** or **ASTROCYTOMA**, but it's not completely certain.
                    """)
                    
                    
        # Incluir una imagen antes o después del texto de autores y asignatura
        # Cambiar el tamaño de la imagen usando la opción de 'width' (en píxeles)
        st.image("logo.png", width=20)  # Ajusta el ancho a 200 píxeles

# Texto con autores y asignatura
        st.markdown("""
---
**Authors**:  
- Evgeny Grachev  
- David Blásco  
- Manuel Rocamora  

**Subject**:  
- Data Science and Clinical Decision Support Systems
""")
        st.image("logo.png", use_column_width=True)


if __name__ == '__main__':
    main()