import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Candidate AI Classifier - Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
<style>
    /* Fuente y colores generales */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* --- DARK MODE STYLES --- */
    
    /* Background color for the whole app */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: #FAFAFA !important;
    }
    
    .main-header {
        font-size: 3rem;
        color: #82B1FF !important; /* Azul claro neón */
        font-weight: 800;
        margin-bottom: 0;
        text-shadow: 0px 0px 10px rgba(130, 177, 255, 0.3);
    }
    
    .sub-text {
        font-size: 1.1rem;
        color: #B0BEC5 !important;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* KPI Cards - Dark */
    .kpi-card {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid #30333F;
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.5);
        border-color: #82B1FF;
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #B0BEC5 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 5px;
    }
    
    /* Tabs customisation - Dark */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #262730;
        color: #FAFAFA;
        border-radius: 8px;
        border: 1px solid #30333F;
        padding: 0 20px;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A; /* Azul oscuro intenso */
        border: 1px solid #82B1FF;
        color: #FFFFFF !important;
        font-weight: 700;
    }
    
    /* Candidate Card - Dark */
    .candidate-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #252526 100%);
        border: 1px solid #424242;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    /* Adjust text colors inside specific components if needed */
    strong {
        color: #FFFFFF !important;
    }
    hr {
        border-color: #424242;
    }
    
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown('<p class="main-header">🧠 Candidate AI Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Sistema Inteligente de Clasificación y Selección de Talento</p>', unsafe_allow_html=True)
with col_head2:
    # Placeholder para logo si existiera
    st.markdown("### 🏢 HHRR Analytics")

st.markdown("---")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('results/candidate_predictions.csv')
        return df
    except FileNotFoundError:
        return None

df = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3048/3048122.png", width=100) # Icono genérico
    st.header("⚙️ Configuración")
    
    if df is not None:
        st.subheader("Filtros Activos")
        
        # Filtros
        pred_filter = st.multiselect(
            "Estado Predicción:",
            options=df['Predicción'].unique(),
            default=df['Predicción'].unique()
        )
        
        conf_min = st.slider(
            "Confianza Mínima:",
            min_value=df['Confianza'].min(),
            max_value=100.0,
            value=50.0,
            step=1.0,
            format="%.1f%%"
        )
        
        # Filtrado
        df_filtered = df[
            (df['Predicción'].isin(pred_filter)) &
            (df['Confianza'] >= conf_min)
        ]
        
        st.info(f"Mostrando **{len(df_filtered)}** de **{len(df)}** candidatos")
        
        st.markdown("---")
        st.caption("v1.0.0 | Powered by Experto en Data Science")

# --- CONTENIDO PRINCIPAL ---

if df is not None:
    # 1. KPIs SECTION (Custom HTML Cards)
    k1, k2, k3, k4 = st.columns(4)
    
    total = len(df)
    current = len(df_filtered)
    
    # Calcular métricas
    if current > 0:
        pct_accepted = (len(df_filtered[df_filtered['Predicción']=='Aceptado']) / current) * 100
        avg_conf = df_filtered['Confianza'].mean()
    else:
        pct_accepted = 0
        avg_conf = 0
        
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{current}</p>
            <p class="kpi-label">Candidatos Filtrados</p>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        icon = "📈" if pct_accepted > 20 else "📉"
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #43A047;">
            <p class="kpi-value" style="color: #2E7D32;">{pct_accepted:.1f}%</p>
            <p class="kpi-label">Tasa de Aceptación {icon}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #E65100;">
            <p class="kpi-value">{avg_conf:.1f}%</p>
            <p class="kpi-label">Confianza Promedio</p>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #546E7A;">
            <p class="kpi-value">{len(df)}</p>
            <p class="kpi-label">Total Base de Datos</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###") # Spacer

    # 2. TABS SECTION
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analytics & Datos", 
        "📈 Performance Modelo", 
        "🔍 Interpretabilidad", 
        "👤 Ficha Candidato"
    ])

    # TAB 1: DATA EXPLORER
    with tab1:
        st.subheader("Explorador de Resultados")
        
        col_list, col_chart = st.columns([2, 1])
        
        with col_list:
            st.dataframe(
                df_filtered[['Nombre', 'Predicción', 'Confianza', 'Probabilidad_Aceptado', 'Probabilidad_Rechazado']],
                use_container_width=True,
                height=400,
                column_config={
                    "Probabilidad_Aceptado": st.column_config.ProgressColumn(
                        "Prob. Aceptado", format="%.2f", min_value=0, max_value=1,
                    ),
                    "Probabilidad_Rechazado": st.column_config.ProgressColumn(
                        "Prob. Rechazado", format="%.2f", min_value=0, max_value=1,
                    ),
                    "Confianza": st.column_config.NumberColumn(
                        "Confianza", format="%.1f %%"
                    ),
                    "Predicción": st.column_config.TextColumn(
                        "Estado"
                    )
                }
            )
            
        with col_chart:
            st.markdown("#### Distribución por Estado")
            if current > 0:
                counts = df_filtered['Predicción'].value_counts()
                fig, ax = plt.subplots(figsize=(5, 5))
                colors = ['#66BB6A', '#EF5350'] if 'Aceptado' in counts else ['#EF5350']
                # Asegurar colores consistentes
                color_map = {'Aceptado': '#66BB6A', 'Rechazado': '#EF5350'}
                plot_colors = [color_map.get(lbl, '#999') for lbl in counts.index]
                
                patches, texts, autotexts = ax.pie(
                    counts, labels=counts.index, autopct='%1.1f%%', 
                    startangle=90, colors=plot_colors,
                    textprops={'fontsize': 12}
                )
                ax.axis('equal')  
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning("No hay datos para mostrar.")

    # TAB 2: MODEL PERFORMANCE (NEW)
    with tab2:
        st.subheader("Diagnóstico del Modelo")
        st.markdown("Evaluación del rendimiento del clasificador en el conjunto de prueba.")
        
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.markdown("#### Matriz de Confusión")
            if os.path.exists("results/confusion_matrix.png"):
                st.image("results/confusion_matrix.png", use_column_width=True)
            else:
                st.warning("Matriz de confusión no encontrada.")
                
        with col_perf2:
            st.markdown("#### Curva ROC")
            if os.path.exists("results/roc_curve.png"):
                st.image("results/roc_curve.png", use_column_width=True)
            else:
                st.warning("Curva ROC no encontrada.")
        
        # Reporte de texto si existe
        if os.path.exists("results/classification_report.txt"):
            with st.expander("Ver Reporte de Clasificación Detallado (Texto)"):
                with open("results/classification_report.txt", "r") as f:
                    st.code(f.read())

    # TAB 3: INTERPRETABILITY
    with tab3:
        st.subheader("Entendiendo las Decisiones (Interpretabilidad)")
        
        st.markdown("#### 1. ¿Qué variables importan más?")
        if os.path.exists("results/interpretability_feature_importance.png"):
            st.image("results/interpretability_feature_importance.png", use_column_width=True)
        
        st.markdown("#### 2. ¿Cómo influyen las variables principales?")
        if os.path.exists("results/interpretability_partial_dependence.png"):
            st.image("results/interpretability_partial_dependence.png", use_column_width=True)

    # TAB 4: CANDIDATE PROFILE
    with tab4:
        st.subheader("Ficha Técnica del Candidato")
        
        all_candidates = df_filtered['Nombre'].unique()
        if len(all_candidates) > 0:
            selected = st.selectbox("Seleccione un candidato para ver detalle:", all_candidates)
            
            cand = df[df['Nombre'] == selected].iloc[0]
            
            # Tarjeta de Candidato
            st.markdown(f"""
            <div class="candidate-card">
                <h2 style="color: #1565C0; margin-bottom: 0;">{cand['Nombre']}</h2>
                <hr>
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <p><strong>Predicción Modelo:</strong> <span style="color: {'#2E7D32' if cand['Predicción']=='Aceptado' else '#C62828'}; font-weight: bold; font-size: 1.2rem;">{cand['Predicción']}</span></p>
                        <p><strong>Confianza:</strong> {cand['Confianza']:.1f}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("###")
            
            col_prob, col_analysis = st.columns([1, 2])
            
            with col_prob:
                st.markdown("#### Probabilidades")
                prob_data = pd.DataFrame({
                    'Clase': ['Aceptado', 'Rechazado'],
                    'Probabilidad': [cand['Probabilidad_Aceptado'], cand['Probabilidad_Rechazado']]
                })
                
                fig_bar, ax_bar = plt.subplots(figsize=(4, 3))
                sns.barplot(data=prob_data, x='Clase', y='Probabilidad', palette=['#66BB6A', '#EF5350'], ax=ax_bar)
                ax_bar.set_ylim(0, 1)
                st.pyplot(fig_bar)
                
            with col_analysis:
                st.markdown("#### Por qué esta decisión?")
                st.info("""
                💡 **Nota:** En una versión futura integrada en producción, aquí se mostraría el gráfico SHAP/LIME 
                específico para este candidato en tiempo real. 
                
                Actualmente, puedes consultar la pestaña 'Interpretabilidad' para ver las reglas generales del modelo.
                """)
                if os.path.exists("results/interpretability_individual_0.png"):
                    st.image("results/interpretability_individual_0.png", caption="Ejemplo de análisis individual (Estático)", width=400)
                    
        else:
            st.warning("No hay candidatos disponibles con los filtros actuales.")

else:
    st.error("⚠️ No se encontraron resultados. Por favor ejecuta el pipeline principal (`python main.py`) primero.")
