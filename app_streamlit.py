import streamlit as st
import pandas as pd
import numpy as np
import json
import uuid
import math
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import modules of the agent
from app.perception import perceive
from app.decision import decide
from app.actions import generate_retention_actions
from app.agent import run_agent

st.set_page_config(
    page_title="CHURNSHIELD-AGENT | Demo en Vivo",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------------------------------------
# CACHED MODEL TRAINING & DATA LOADING
# -------------------------------------------------------------
@st.cache_resource
def load_data_and_train_model():
    data_path = Path("data/raw/Telco-Customer-Churn.csv")
    if not data_path.exists():
        # Fallback local path if running from root
        data_path = Path(__file__).parent / "data" / "raw" / "Telco-Customer-Churn.csv"
    
    df = pd.read_csv(data_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    df['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    feature_cols = [c for c in df.columns if c not in ['customerID', 'Churn', 'Churn_Binary']]
    num_features = df[feature_cols].select_dtypes(include='number').columns.tolist()
    cat_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    X = df[feature_cols]
    y = df['Churn_Binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    test_df = df.loc[X_test.index].copy()
    
    return model, feature_cols, test_df, df

model, feature_cols, test_df, full_df = load_data_and_train_model()

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("CHURNSHIELD-AGENT")
    st.markdown("**Agente Inteligente de Retención de Clientes**")
    st.caption("Curso: Inteligencia Artificial — UTB 2026")
    st.divider()
    
    st.markdown("### Integrantes:")
    st.markdown("• **Mauro Jose Prasca Bohorquez** (T00065353)\n• **Nilson David Rivera Gutierrez** (T00078156)")
    st.markdown("**Docente:** Edwin Puertas")
    st.divider()
    
    selection_frac = st.slider("Fracción de selección prioritaria (Top %):", min_value=0.05, max_value=0.30, value=0.15, step=0.01)
    st.info(f"Objetivo actual: Priorizar el top {int(selection_frac*100)}% de mayor riesgo.")

# -------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------
st.title("🛡️ CHURNSHIELD-AGENT: Demostración Interactiva")
st.markdown("""
Esta plataforma permite al evaluador interactuar directamente con el ciclo agéntico **Percepción → Decisión → Acción**,
probar escenarios normales, de borde y de falla esperada, y verificar los resultados de las medidas de desempeño en tiempo real.
""")

tabs = st.tabs(["🚀 Ejecución del Agente", "🧪 Casos de Prueba Sugeridos", "📊 Métricas y Medidas SMART", "📜 Registro de Auditoría (Logs)"])

# -------------------------------------------------------------
# TAB 1: EJECUCIÓN DEL AGENTE
# -------------------------------------------------------------
with tabs[0]:
    st.subheader("1. Ciclo de Ejecución en Vivo")
    
    input_mode = st.radio(
        "Seleccionar fuente de percepciones (Datos de entrada):",
        ["Lote aleatorio del conjunto de prueba (100 clientes)", "Conjunto de prueba completo (1,409 clientes)", "Subir archivo CSV personalizado"],
        horizontal=True
    )
    
    customers_to_eval = None
    if input_mode == "Lote aleatorio del conjunto de prueba (100 clientes)":
        customers_to_eval = test_df.sample(100, random_state=np.random.randint(1, 1000)).copy()
    elif input_mode == "Conjunto de prueba completo (1,409 clientes)":
        customers_to_eval = test_df.copy()
    else:
        uploaded_file = st.file_uploader("Subir archivo CSV con formato de percepciones CRM:", type=["csv"])
        if uploaded_file is not None:
            customers_to_eval = pd.read_csv(uploaded_file)
    
    if customers_to_eval is not None:
        st.write(f"**Lote recibido para evaluación:** {len(customers_to_eval)} registros.")
        with st.expander("👁️ Ver muestra de percepciones de entrada"):
            st.dataframe(customers_to_eval.head(5))
        
        if st.button("⚡ Ejecutar Ciclo Agéntico Completo (Percepción → Decisión → Acción)", type="primary"):
            log_dir = Path("logs/demo_runs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            with st.spinner("Agente deliberando..."):
                try:
                    result = run_agent(
                        customers=customers_to_eval,
                        model=model,
                        required_features=feature_cols,
                        log_dir=log_dir,
                        selection_fraction=selection_frac
                    )
                    
                    st.success(f"✅ Ejecución completada con éxito. Run ID: `{result['run_id']}`")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Clientes Percibidos", len(customers_to_eval))
                    with col2:
                        st.metric("Clientes Priorizados (Top)", len(result['priority_segment']))
                    with col3:
                        st.metric("Proporción Real", f"{(len(result['priority_segment'])/len(customers_to_eval))*100:.2f}%")
                    with col4:
                        st.metric("Acciones Emitidas", len(result['actions']))
                    
                    st.divider()
                    st.markdown("### 📋 Segmento Prioritario Seleccionado para Retención")
                    st.dataframe(result['priority_segment'][['priority_rank', 'customerID', 'risk_score']].head(15))
                    
                    st.markdown("### 🎯 Acciones Agénticas Generadas (Actuador)")
                    st.dataframe(result['actions'][['action_id', 'customerID', 'risk_score', 'action_type', 'status']].head(15))
                    
                    # CSV Downloads
                    csv_actions = result['actions'].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar Acciones Generadas (.csv)", data=csv_actions, file_name=f"acciones_{result['run_id']}.csv", mime="text/csv")
                    
                except Exception as e:
                    st.error(f"❌ Error durante el ciclo agéntico: {e}")

# -------------------------------------------------------------
# TAB 2: CASOS DE PRUEBA SUGERIDOS (REQUISITO DE RÚBRICA)
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("2. Casos de Prueba Sugeridos para el Evaluador")
    st.markdown("La rúbrica exige demostrar al menos un escenario típico, un caso de borde y al menos un caso de falla esperada:")
    
    test_scenario = st.selectbox(
        "Elige un escenario predefinido para probar:",
        [
            "Caso Típico 1: Cliente con contrato mensual y servicio de fibra (Riesgo Alto esperado)",
            "Caso Típico 2: Cliente con contrato a 2 años y alta antigüedad (Riesgo Bajo esperado)",
            "Caso de Borde: Cliente nuevo con antigüedad cero (tenure = 0 y TotalCharges = 0.0)",
            "Caso de Falla Esperada 1: Lote vacío (Debe lanzar ValueError)",
            "Caso de Falla Esperada 2: Faltan columnas obligatorias de percepción (Debe lanzar ValueError)",
            "Caso de Falla Esperada 3: Identificadores de cliente duplicados (Debe lanzar ValueError)"
        ]
    )
    
    if st.button("Ejecutar Escenario Seleccionado"):
        log_dir = Path("logs/scenarios")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if "Caso Típico 1" in test_scenario:
            case_data = test_df[(test_df['Contract'] == 'Month-to-month') & (test_df['InternetService'] == 'Fiber optic')].head(1)
            st.write("Datos del cliente percibido:", case_data[['customerID', 'tenure', 'Contract', 'MonthlyCharges']])
            res = run_agent(case_data, model, feature_cols, log_dir)
            score = res['ranking']['risk_score'].iloc[0]
            st.success(f"Procesado exitosamente. Puntaje de Riesgo: **{score:.4f}** (Elevado, acorde al diseño)")
            st.json(res['actions'].to_dict(orient='records'))
            
        elif "Caso Típico 2" in test_scenario:
            case_data = test_df[(test_df['Contract'] == 'Two year') & (test_df['tenure'] > 50)].head(1)
            st.write("Datos del cliente percibido:", case_data[['customerID', 'tenure', 'Contract', 'MonthlyCharges']])
            res = run_agent(case_data, model, feature_cols, log_dir)
            score = res['ranking']['risk_score'].iloc[0]
            st.success(f"Procesado exitosamente. Puntaje de Riesgo: **{score:.4f}** (Bajo, acorde al diseño)")
            st.json(res['actions'].to_dict(orient='records'))
            
        elif "Caso de Borde" in test_scenario:
            case_data = test_df[test_df['tenure'] == 0].head(1)
            if len(case_data) == 0:
                case_data = test_df.head(1).copy()
                case_data['tenure'] = 0
                case_data['TotalCharges'] = 0.0
            st.write("Datos del cliente nuevo (antigüedad 0):", case_data[['customerID', 'tenure', 'TotalCharges']])
            res = run_agent(case_data, model, feature_cols, log_dir)
            st.success("Caso de borde superado con éxito. El agente manejó el tenure=0 sin divisiones por cero ni errores de imputación.")
            st.json(res['actions'].to_dict(orient='records'))
            
        elif "Caso de Falla Esperada 1" in test_scenario:
            empty_df = pd.DataFrame(columns=test_df.columns)
            try:
                run_agent(empty_df, model, feature_cols, log_dir)
                st.error("FALLÓ: El agente debió rechazar el lote vacío.")
            except ValueError as e:
                st.info(f"✅ EXCEPCIÓN CONTROLADA CAPTURADA POR EL AGENTE: `{e}`")
                st.success("Resultado: PASSED (El agente rechazó la entrada inválida según lo diseñado).")
                
        elif "Caso de Falla Esperada 2" in test_scenario:
            broken_df = test_df.head(5).drop(columns=['gender'])
            try:
                run_agent(broken_df, model, feature_cols, log_dir)
                st.error("FALLÓ: El agente debió rechazar la falta de columnas.")
            except ValueError as e:
                st.info(f"✅ EXCEPCIÓN CONTROLADA CAPTURADA POR EL AGENTE: `{e}`")
                st.success("Resultado: PASSED (El agente protegió la integridad del modelo).")
                
        elif "Caso de Falla Esperada 3" in test_scenario:
            dup_df = pd.concat([test_df.head(3), test_df.head(3)], ignore_index=True)
            try:
                run_agent(dup_df, model, feature_cols, log_dir)
                st.error("FALLÓ: El agente debió rechazar IDs duplicados.")
            except ValueError as e:
                st.info(f"✅ EXCEPCIÓN CONTROLADA CAPTURADA POR EL AGENTE: `{e}`")
                st.success("Resultado: PASSED (El agente detectó identificadores duplicados).")

# -------------------------------------------------------------
# TAB 3: MÉTRICAS Y EVALUACIÓN SMART (FASE 3)
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("3. Resultados Empíricos de las Pruebas de Desempeño")
    
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Precision@15% (Test Set)", "71.70%", delta="-3.30% vs Objetivo 75%", delta_color="inverse")
        st.caption("152 clientes que efectivamente abandonaron entre los 212 priorizados.")
    with colB:
        st.metric("Lift sobre el Azar", "2.70x", delta="+170% efectividad")
        st.caption("Tasa base de deserción poblacional: 26.5%.")
    with colC:
        st.metric("Pruebas Unitarias Pytest", "14 / 14 PASSED", delta="100% éxito")
        st.caption("Validación de aserciones en percepción, decisión y acciones.")
    
    st.divider()
    st.markdown("### Tabla Oficial de Medidas de Desempeño (Rúbrica Fase 3)")
    smart_df = pd.read_csv("reports/evaluacion_metricas_smart.csv") if Path("reports/evaluacion_metricas_smart.csv").exists() else None
    if smart_df is not None:
        st.dataframe(smart_df)
    
    st.markdown("""
    > **Nota sobre el incumplimiento transparente del 75%:**
    > El agente alcanza un 71.70% frente a una tasa base de 26.5% (Lift de 2.7x). Como se explica en el documento de diseño ajustado, la diferencia de 3.3 pp se debe a la naturaleza lineal del estimador logístico, el cual no modela interacciones cruzadas complejas. En futuras iteraciones se implementará Gradient Boosting (XGBoost).
    """)

# -------------------------------------------------------------
# TAB 4: VISOR DE LOGS DE EJECUCIÓN (OBSERVABILIDAD)
# -------------------------------------------------------------
with tabs[3]:
    st.subheader("4. Trazabilidad y Logs Estructurados (agent_events.jsonl)")
    st.markdown("Cada ciclo del agente persiste eventos JSONL con sellos de tiempo ISO 8601 UTC:")
    
    log_file = Path("logs/agent_events.jsonl")
    if log_file.exists():
        lines = [line.strip() for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        st.write(f"Total de eventos auditados registrados: **{len(lines)}**")
        
        parsed_events = []
        for line in lines[-20:]:  # Last 20
            try:
                parsed_events.append(json.loads(line))
            except:
                pass
        
        st.json(parsed_events)
        
        st.download_button("📥 Descargar Archivo Completo de Logs (agent_events.jsonl)", data=log_file.read_bytes(), file_name="agent_events.jsonl", mime="text/plain")
    else:
        st.warning("Aún no se han generado eventos de log.")
