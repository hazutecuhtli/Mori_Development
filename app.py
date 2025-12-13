# -*- coding: utf-8 -*-
#***************************************************************************
# Importing Libraries
#***************************************************************************
import os, warnings, json, random, uuid, csv, sys
import numpy as np
from Scripts.config import LLM1_DIR, LLM2_DIR

#=====================================================================================
# Importing Libraries  ===============================================================
#=====================================================================================
import streamlit as st
import datetime as dt
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from Scripts.Mori_TechnicalPrompts import answer_with_mori_rag, answer_with_mori_plain, answer_with_qwen_base
import torch

#***************************************************************************
# Defining default paths for the model to work
#***************************************************************************
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#***************************************************************************
# Setting up variables
#***************************************************************************
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
model_name = LLM1_DIR

#***************************************************************************
# Sidebar controls for generation params
#***************************************************************************

def sidebar_params():
    with st.sidebar:
        st.title("🎮 Configuración de Mori")

        ss = st.session_state

        # -------------------------
        # Estado inicial
        # -------------------------
        ss.setdefault("persona", "exacto")   # "exacto" | "creativo"
        ss.setdefault("mode", "beam")
        ss.setdefault("max_new", 128)
        ss.setdefault("min_tok", 16)
        ss.setdefault("no_repeat", 3)
        ss.setdefault("repetition_penalty", 1.0)

        # NUEVO: backend
        ss.setdefault("backend", "🍮 FLAN-T5 (fine-tuned)")
        ss.setdefault("use_rag", True)

        # -------------------------
        # Personalidades
        # -------------------------
        st.header("🧠 Personalidades")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Exacto 🧐", use_container_width=True):
                ss.persona = "exacto"
                st.rerun()

        with c2:
            if st.button("Creativo 😃", use_container_width=True):
                ss.persona = "creativo"
                st.rerun()

        st.caption(f"Personalidad actual: **{ss.persona}**")

        st.markdown(
            """
            🔗 Cómo controlar la generación de texto:
            - https://huggingface.co/blog/how-to-generate
            """
        )
        st.markdown("---")

        # -------------------------
        # Selección de modelo
        # -------------------------
        st.title("📙 Modelo")
        ss.backend = st.radio(
            "Elige el modelo de respuesta:",
            options=[
                "🍮 FLAN-T5 (fine-tuned)",
                "👸 Qwen",
            ],
            index=0 if ss.backend == "🍮 FLAN-T5 (fine-tuned)" else 1,
            help="Documentación:\n- FLAN-T5: https://huggingface.co/docs/transformers/model_doc/flan-t5\n- Qwen: https://huggingface.co/Qwen"
        )

        # -------------------------
        # RAG solo para FLAN-T5
        # -------------------------

        st.header("👀 RAG:")
        
        if ss.backend == "🍮 FLAN-T5 (fine-tuned)":
            ss.use_rag = st.checkbox(
                "👷🏽 Usar RAG (FAISS + One-Shot)",
                value=ss.use_rag,
                help=(
                    "Documentación útil:\n"
                    "- RAG: https://huggingface.co/docs/transformers/en/model_doc/rag\n"
                    "- FAISS: https://faiss.ai/\n"
                    "- One-Shot Prompting: https://huggingface.co/docs/transformers/en/tasks/prompting"
                ),
            )
        else:
            ss.use_rag = False
            st.caption("RAG no aplica en modo Qwen (usa solo el modelo base).")

        st.markdown("---")
        st.title("🧾 Vista previa del Prompt")

        if "last_prompt" in ss and ss["last_prompt"]:
            with st.expander("Mostrar prompt generado"):
                st.text_area(
                    "Prompt actual:",
                    ss["last_prompt"],
                    height=200,
                    disabled=True,
                )
        else:
            st.caption("🔍 Aún no se ha generado ningún prompt.")

        # -------------------------
        # Construir diccionario de parámetros
        # -------------------------
        params = {
            "persona": ss.persona,
            "mode": ss.mode,
            "max_new_tokens": int(ss.max_new),
            "min_tokens": int(ss.min_tok),
            "no_repeat_ngram_size": int(ss.no_repeat),
            "repetition_penalty": float(ss.repetition_penalty),
            "backend": ss.backend,
            "use_rag": ss.use_rag,
        }

        # Si ya tienes parámetros específicos para Qwen (como max_new_qwen),
        # los puedes añadir aquí, por ejemplo:
        # params["qwen_max_new"] = int(ss.qwen_max_new)

        return params


#***************************************************************************
# Functions
#***************************************************************************

# Function to clean the question field (por si luego lo quieres usar en un botón)
def limpiar_input():
    st.session_state["entrada"] = ""

# ✅ Corrige la ruta correctamente desde Scripts hacia Models
def get_model_path(folder_name):
    APP_DIR = Path(__file__).resolve().parent
    TEC_DIR = (APP_DIR / "Models" / folder_name).resolve()
    return TEC_DIR

# Function to save user interaction
def saving_interaction(question, response, user_id, use_of_rag, bot_personality, modelo):
    """
    Guarda la interacción en CSV y JSONL para análisis posterior.
    """
    timestamp = dt.datetime.now().isoformat()
    stats_dir = Path("Statistics")
    stats_dir.mkdir(parents=True, exist_ok=True)

    archivo_csv = stats_dir / "conversaciones_log.csv"
    existe_csv = archivo_csv.exists()

    # CSV
    with open(archivo_csv, mode="a", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        if not existe_csv:
            writer.writerow(["timestamp", "user_id", "pregunta", "respuesta", "modelo", "rag", "personality"])
        writer.writerow([timestamp, user_id, question, response, modelo, use_of_rag, bot_personality])

    # JSONL
    archivo_jsonl = stats_dir / "conversaciones_log.jsonl"
    with open(archivo_jsonl, mode="a", encoding="utf-8") as f_jsonl:
        registro = {
            "timestamp": timestamp,
            "user_id": user_id,
            "pregunta": question,
            "respuesta": response,
            "modelo": modelo,
            "uso_rag": use_of_rag,
            "personality": bot_personality
        }
        f_jsonl.write(json.dumps(registro, ensure_ascii=False) + "\n")

# Function to load Mori technical model
@st.cache_resource
def load_mori_model():
    tec_dir = get_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tec_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(tec_dir, local_files_only=True).to(device).eval()
    return model, tokenizer

#-------------------------------------------------------------------------
# Qwen base model (for comparison)
#-------------------------------------------------------------------------

QWEN_MODEL_NAME = LLM2_DIR  # Asegúrate que exista localmente en HF cache
  
@st.cache_resource
def load_qwen_model():
    """
    Carga el modelo base de Qwen sin fine-tuning.
    """
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        local_files_only=True
    ).to(device).eval()

    return model, tokenizer


#-------------------------------------------------------------------------
# Seeds
#-------------------------------------------------------------------------

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#***************************************************************************
# MAIN
#***************************************************************************

if __name__ == "__main__":
    # Semilla para reproducibilidad
    set_seeds(42)

    # --- Estado que debe persistir en todos los reruns ---
    ss = st.session_state
    ss.setdefault("historial", [])
    ss.setdefault("last_prompt", "")
    ss.setdefault("last_response", "")
    ss.setdefault("just_generated", False)

    # Sidebar (control total)
    GEN_PARAMS = sidebar_params()
    GEN_PARAMS["persona"] = ss.persona  # por si acaso

    # Assigning a new ID to the current user
    if "user_id" not in ss:
        ss["user_id"] = str(uuid.uuid4())[:8]

    # Cargar modelo técnico (cacheado)
    model, tokenizer = load_mori_model()

    # Presentación de Mori
    st.title("🤖 Mori - Tu Asistente Personal ⌨️")

    st.caption("🙋🏽‍ Puedes preguntarme conceptos sobre machine learning, estadística, visualización, BI, limpieza de datos y más.")
    st.caption("🙇🏽‍ Por el momento FLAN-T5, solo puedo contestar preguntas simples como:")

    st.caption("  🔹 **Definiciones** — Ejemplo: *¿Qué es machine learning?*")
    st.caption("  🔹 **Procedimientos** — Ejemplo: *¿Cómo limpiar datos?*")
    st.caption("  🔹 **Funcionalidad** — Ejemplo: *¿Para qué sirve un autoencoder?*")

    st.caption("🔥 Qwen 1.5 corre con todas sus capacidades completas.")
    st.caption("  🔹 **Consejo** — Sé paciente y específico. Usar signos correctos ayuda a obtener mejores respuestas.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption("🦾 Aún estoy aprendiendo. Puedes ver mi desarrollo aquí:")
    st.caption("[hazutecuhtli.github.io](https://github.com/hazutecuhtli/Mori_Development)")

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption("✏️ Escribe tu pregunta abajo.")

    # 🔁 Limpieza segura antes del formulario
    if ss.pop("_clear_entrada", False):
        if "entrada" in ss:
            del ss["entrada"]

    # 🧠 Flash de respuesta (la guardamos, pero la mostraremos después del form)
    _flash = ss.pop("_flash_response", None)

    # Formulario principal
    with st.form("formulario_mori"):
        user_question = st.text_area("📝 Escribe tu pregunta aquí", key="entrada", height=100)
        submitted = st.form_submit_button("Responder")

    if submitted:
        if not user_question:
            st.info("Mori: ¿Podrías repetir eso? No entendí bien 😅")
        else:
            backend = GEN_PARAMS.get("backend", "Mori (FT + RAG)")
            persona = GEN_PARAMS.get("persona", ss.persona)

            # -----------------------------------------
            # Backend Qwen base (sin RAG, sin FT)
            # -----------------------------------------
            if backend.startswith("👸 Qwen"):
                modelito = 'Qwen'
                qwen_model, qwen_tokenizer = load_qwen_model()
                response, prompt = answer_with_qwen_base(
                    qwen_tokenizer,
                    qwen_model,
                    user_question,
                    persona,
                    max_new_tokens=GEN_PARAMS.get("qwen_max_new", 64),
                )
                use_of_rag = "sin RAG"

            # -----------------------------------------
            # Backend Mori Técnico (FT + RAG / sin RAG)
            # -----------------------------------------
            else:
                modelito = 'FLAN-T5'
                use_rag = st.session_state.get("use_rag", False)

                if use_rag:
                    use_of_rag = 'Con RAG'
                    response, prompt = answer_with_mori_rag(
                        tokenizer, model, user_question,
                        modo=persona,
                        score_threshold=0.84,
                        verbose=False
                    )
                else:
                    use_of_rag = 'Sin RAG'
                    response, prompt = answer_with_mori_plain(
                        tokenizer, model, user_question,
                        modo=persona
                    )

            ss["last_prompt"] = prompt
            ss["just_generated"] = True

            # 🧠 Guarda historial
            hora_actual = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ss.historial.append(("Tú", user_question, hora_actual))

            hora_actual = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ss.historial.append(("Mori", response, hora_actual, modelito, use_of_rag, persona))

            # 💾 Guarda conversación
            saving_interaction(user_question, response, ss["user_id"], modelito, use_of_rag, persona)

            # 🟩 Guarda respuesta para mostrar después del rerun
            ss["_flash_response"] = response

            # 🧼 Limpieza del textarea en el próximo ciclo
            ss["_clear_entrada"] = True

            # ♻️ Forzar refresh (sidebar verá el nuevo prompt)
            st.rerun()

    # -----------------------------------------------------------
    # 💬 Mostrar la respuesta actual (flash) justo aquí ↓↓↓
    # -----------------------------------------------------------
    if _flash:
        st.success(_flash)

    # 🔁 Historial con estilo chat y contenedor con scroll
    if ss.historial:
        st.markdown("---")

        # 💾 Botón de descarga arriba del historial
        lineas = []
        for msg in reversed(ss.historial):
            if len(msg) == 6:
                autor, texto, hora, model, rag, bot_per = msg
                lineas.append(f"[{hora}], {autor}: {texto}, Model:{model}, RAG:{rag}, Persoality:{bot_per}")
            else:
                autor, texto, hora = msg
                lineas.append(f"[{hora}], {autor}: {texto}")
        texto_chat = "\n\n".join(lineas)

        st.download_button(
            label="💾 Descargar conversación como .txt",
            data=texto_chat,
            file_name="conversacion_mori.txt",
            mime="text/plain",
            use_container_width=True
        )

        # 🪟 Contenedor con scroll y burbujas
        st.markdown(
            """
            <div id="chat-container" style="
                max-height: 400px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #333;
                border-radius: 10px;
                background: linear-gradient(180deg, #0e0e0e 0%, #1b1b1b 100%);
                margin-top: 10px;
            ">
            """,
            unsafe_allow_html=True
        )

        for msg in reversed(ss.historial):
            if len(msg) == 6:
                autor, texto, hora, model, rag, bot_per = msg
            else:
                autor, texto, hora = msg

            if autor == "Tú":
                st.markdown(
                    f"""
                    <div style="
                        text-align: right;
                        background-color: #2d2d2d;
                        color: #e6e6e6;
                        padding: 10px 14px;
                        border-radius: 12px;
                        margin: 6px 0;
                        border: 1px solid #3a3a3a;
                        display: inline-block;
                        max-width: 80%;
                        float: right;
                        clear: both;
                    ">
                        🧍‍♂️ <b>{autor}:</b> {texto}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        text-align: left;
                        background-color: #162b1f;
                        color: #d9ead3;
                        padding: 10px 14px;
                        border-radius: 12px;
                        margin: 6px 0;
                        border: 1px solid #264d36;
                        display: inline-block;
                        max-width: 80%;
                        float: left;
                        clear: both;
                    ">
                        🤖 <b>{autor}:</b> {texto}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("</div>", unsafe_allow_html=True)
