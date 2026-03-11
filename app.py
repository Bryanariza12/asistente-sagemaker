import streamlit as st
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Carga variables de entorno
load_dotenv()

# 2. Configuración (Usa variables de entorno para mayor seguridad)
MODEL_ID = os.getenv("MODEL_ID", "arn:aws:bedrock:us-east-1:385208337656:inference-profile/us.anthropic.claude-sonnet-4-6")
SYSTEM_PROMPT = """
Eres un mentor experto en Amazon SageMaker y en la certificación AWS Certified AI Practitioner.
Tu misión es transformar conceptos complejos de Machine Learning en conocimiento sencillo y accionable.

Sigue ESTRICTAMENTE este protocolo de respuesta para cada consulta del estudiante:

1. ANALOGÍA: Empieza con una analogía creativa y cercana que conecte el concepto técnico con algo cotidiano.
2. EXPLICACIÓN TÉCNICA: Define el concepto con precisión de ingeniero de ML, mencionando el rol en el ciclo de vida (preparación, entrenamiento, despliegue o monitoreo).
3. PASO A PASO: Proporciona un flujo de trabajo lógico en formato de lista numerada.
4. CLAVE PARA EL EXAMEN: Identifica qué aspecto de este tema es un "gancho" común en la certificación AI Practitioner.
5. DIAGRAMA TEXTUAL: Crea un esquema visual simple (ASCII) del flujo de datos o componentes.
6. EJEMPLO PRÁCTICO: Proporciona un fragmento de código (SDK Python) o una acción crítica en la consola.

REGLAS DE ORO:
- Si no sabes algo, no inventes: sugiere consultar la documentación oficial de AWS.
- Mantén un tono didáctico, empático, profesional y ligeramente ingenioso (estilo tutor entusiasta).
- ¡IMPORTANTE!: No omitas ninguna de las 6 secciones. Si el usuario hace una pregunta rápida, responde de forma concisa pero manteniendo la estructura obligatoria.
- Usa terminología técnica correcta.
"""

# 3. Inicialización definitiva (sin trucos, usando el parámetro correcto)
try:
    llm = ChatBedrock(
        model_id=MODEL_ID,
        provider="anthropic", # <--- Aquí le damos el proveedor directamente
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 2000
        }
    )
except Exception as e:
    st.error(f"Error crítico de conexión: {e}")
    st.stop()

    

# 4. Interfaz y Persistencia
st.set_page_config(page_title="Experto SageMaker", page_icon="🤖")
st.title("🤖 Asistente experto en Amazon SageMaker")

# Bloque de inicialización forzada
if "messages" not in st.session_state or st.session_state.get("prompt_version") != "v2":
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.prompt_version = "v2" # Esto obliga a recrear el chat
    st.rerun() # Esto recarga la app automáticamente

# 5. Lógica del Chat
def obtener_respuesta(historial):
    try:
        respuesta = llm.invoke(historial)
        return respuesta.content
    except Exception as e:
        return f"Error al generar respuesta: {e}"

# Renderizar mensajes previos
for msg in st.session_state.messages:
    if isinstance(msg, (HumanMessage, AIMessage)):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

# Input del usuario
if entrada := st.chat_input("Pregunta sobre SageMaker (ej. ¿Cómo crear un endpoint?)"):
    # Agregar mensaje usuario
    st.session_state.messages.append(HumanMessage(content=entrada))
    with st.chat_message("user"):
        st.markdown(entrada)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Consultando al experto..."):
            respuesta = obtener_respuesta(st.session_state.messages)
            st.markdown(respuesta)
            # Guardar respuesta
            st.session_state.messages.append(AIMessage(content=respuesta))