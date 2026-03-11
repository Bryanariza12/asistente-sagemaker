import streamlit as st
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Carga de configuraciones desde el archivo .env
load_dotenv()

# 2. Configuración del modelo: Definimos qué IA usaremos y cómo debe comportarse
# MODEL_ID apunta al modelo en AWS Bedrock.
# SYSTEM_PROMPT es el "manual de instrucciones" que define la personalidad y estructura del asistente.
load_dotenv()

# Si MODEL_ID no está en el .env, el programa se detiene para avisarte
MODEL_ID = os.getenv("MODEL_ID")
if not MODEL_ID:
    st.error("Error: La variable de entorno MODEL_ID no está configurada en el archivo .env")
    st.stop()
SYSTEM_PROMPT = """
Eres un mentor experto en Amazon SageMaker y en la certificación AWS Certified AI Practitioner.
Tu misión es transformar conceptos complejos de Machine Learning en conocimiento sencillo y accionable.

Sigue ESTRICTAMENTE este protocolo de respuesta para cada consulta del estudiante:

1. ANALOGÍA: Empieza con una analogía creativa y cercana que conecte el concepto técnico con algo cotidiano.
2. EXPLICACIÓN TÉCNICA: Define el concepto con precisión de ingeniero de ML, mencionando el rol en el ciclo de vida.
3. PASO A PASO: Proporciona un flujo de trabajo lógico en formato de lista numerada.
4. CLAVE PARA EL EXAMEN: Identifica qué aspecto de este tema es un "gancho" común en la certificación.
5. DIAGRAMA TEXTUAL: Crea un esquema visual simple (ASCII).
6. EJEMPLO PRÁCTICO: Proporciona un fragmento de código o acción crítica.

REGLAS DE ORO: Mantén un tono didáctico, profesional y no inventes información.
"""

# 3. Inicialización del cliente de AWS Bedrock
try:
    """Configura la conexión con el servicio de IA de AWS."""
    llm = ChatBedrock(
        model_id=MODEL_ID,
        provider="anthropic",
        model_kwargs={
            "temperature": 0.7,  # Nivel de creatividad: 0 es muy preciso, 1 es muy creativo
            "max_tokens": 2000
        }
    )
except Exception as e:
    st.error(f"Error crítico de conexión: {e}")
    st.stop()

# 4. Interfaz de Usuario y Memoria de la sesión
st.set_page_config(page_title="Experto SageMaker", page_icon="🤖")
st.title("🤖 Asistente experto en Amazon SageMaker")

# Inicializamos el historial de mensajes si es la primera vez que entra el usuario
if "messages" not in st.session_state or st.session_state.get("prompt_version") != "v2":
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.prompt_version = "v2"
    st.rerun()

# 5. Lógica del Chat
def obtener_respuesta(historial):
    """
    Envía todo el historial de la conversación al modelo para generar la respuesta.
    
    Args:
        historial (list): Lista de mensajes (Usuario, IA y Sistema).
    Returns:
        str: El contenido de la respuesta generada por la IA.
    """
    try:
        respuesta = llm.invoke(historial)
        return respuesta.content
    except Exception as e:
        return f"Error al generar respuesta: {e}"

# Renderizado del historial en pantalla: Muestra la charla actual al usuario
for msg in st.session_state.messages:
    if isinstance(msg, (HumanMessage, AIMessage)):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

# Captura de entrada del usuario
if entrada := st.chat_input("Pregunta sobre SageMaker (ej. ¿Cómo crear un endpoint?)"):
    # Guardamos y mostramos la pregunta del usuario
    st.session_state.messages.append(HumanMessage(content=entrada))
    with st.chat_message("user"):
        st.markdown(entrada)

    # Generamos la respuesta del asistente usando el historial acumulado
    with st.chat_message("assistant"):
        with st.spinner("Consultando al experto..."):
            respuesta = obtener_respuesta(st.session_state.messages)
            st.markdown(respuesta)
            # Guardamos la respuesta en el historial para mantener el contexto
            st.session_state.messages.append(AIMessage(content=respuesta))