# Asistente Experto: Amazon SageMaker & AWS Bedrock

Este proyecto es una app web interactiva que fue desarollada con **Streamlit**
permite a los usuarios conversar con un asistente de IA especializado en conceptos de
Amazon SageMaker. la aplicacion utiliza **AWS Bedrock** -> (Anthropic Claude) para ofrecer dichas respuestas.

## Características Principales:

* **IA Especializada:** Prompt engineering diseñado para explicar conceptos de SageMaker desde lo básico hasta lo avanzado.
* **Seguridad:** Implementación del principio de minimo privilegio ejecutando la aplicación bajo un usuario sin permisos (`appuser`).
* **Contenerización:** Imagen Docker optimizada y ligera mediante el uso de `--no-cache-dir`.
* **Configuración Segura:** Gestión de credenciales mediante variables de entorno (evitando hardcoding).

## 🛠️ Tecnologías Utilizadas

* [Streamlit](https://streamlit.io/): Framework para la interfaz web.
* [LangChain](https://www.langchain.com/): Orquestación de modelos de lenguaje.
* [AWS Bedrock](https://aws.amazon.com/bedrock/): Acceso a modelos fundacionales.
* [Docker](https://www.docker.com/): Contenerización del entorno.



##  Requisitos Previos

1. Tener Docker instalado.
2. Contar con credenciales de AWS configuradas (IAM User con acceso a Bedrock).

##  Cómo Ejecutar el Proyecto

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DE_EL_REPOSITORIO>
   cd LLM
