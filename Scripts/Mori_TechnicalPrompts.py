# -*- coding: utf-8 -*-
"""Script to create prompt to interact with LLMs for text generation"""
#=====================================================================================
# Importing Libraries  ===============================================================
#=====================================================================================
import unicodedata
import re
from Scripts.Mori_Chatbot_SpanishCorrections import polish_spanish
from Scripts.Mori_Technical_RAGwithFAISS import retrieve_docs
import os, torch
import warnings
# ************************************************************************
# Defining default paths for the model to work
# ************************************************************************
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#=====================================================================================
# Functions  =========================================================================
#=====================================================================================

def recortar_ultima_oracion(texto):

    """Remove incomplete generated text"""
    texto = texto.strip()
    if not texto:
        return texto

    # signos válidos de cierre
    signos = ".?!…"

    # encontrar la última posición
    posiciones = [texto.rfind(s) for s in signos]
    posiciones = [p for p in posiciones if p != -1]

    if not posiciones:
        return texto  # no hay signos → lo regresamos

    final = max(posiciones)

    # aseguramos que no sea demasiado pronto
    if final < len(texto) * 0.3:
        return texto

    return texto[:final + 1].strip()


def normalize_text(text: str) -> str:

    """Normalize text for correct and similar processing"""
    t = text.lower().strip()
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.replace("¿", "").replace("?", "")
    t = re.sub(r"\s+", " ", t)
    return t

def classify_question_type_from_text(text: str) -> str:

    """Determine the type of question"""
    
    t = normalize_text(text)

    if "para que sirve" in t or "para que se usa" in t:
        return "funcionalidad"
    if t.startswith("como ") or "pasos para" in t or "como puedo" in t:
        return "procedimiento"
    if t.startswith("que es ") or "definicion de" in t:
        return "definicion"
    return "definicion"


def build_prompt(qtype: str, question: str) -> str:
    """Generates a base prompt"""
    return (
        f"Tipo: {qtype}\n"
        f"Pregunta: {question}\n"
        "Respuesta:"
    )

def build_prompt_inference(question: str):
    """Generates an inference prompt"""
    qtype = classify_question_type_from_text(question)
    return build_prompt(qtype, question)

def build_prompt_training(row):
    """Generates a prompt for training"""
    qtype = row["question_type"]       # definicion / procedimiento / funcionalidad
    question = row["input"]
    return build_prompt(qtype, question)


def build_prompt_for_mori(user_question: str, question_type: str, top_doc: dict) -> str:
    """
    Prompt one-shot for RAG Mori, relying on question_type (definicion, procedimiento, funcionalidad).
    """
    ejemplo_q = (top_doc.get("input") or "").strip()
    ejemplo_a = (top_doc.get("output") or "").strip()
    contexto  = (top_doc.get("context") or "").strip()
    term      = (top_doc.get("canonical_term") or "").strip()

    prompt = (
        "Eres un asistente técnico llamado Mori. Respondes en español, de forma clara y concisa.\n\n"
        f"Contexto del concepto:\n"
        f"- Término: {term}\n"
        f"- Área: {contexto}\n"
        f"- Tipo de pregunta: {question_type}\n\n"
        f"A continuación tienes un ejemplo de pregunta y respuesta del mismo tipo \"{question_type}\":\n"
        f"Pregunta de ejemplo:\n{ejemplo_q}\n\n"
        f"Respuesta de ejemplo:\n{ejemplo_a}\n\n"
        "Usa este estilo y nivel de detalle como guía.\n\n"
        f"Ahora responde la siguiente pregunta del usuario manteniendo el tipo \"{question_type}\" "
        "(sin inventar información que no aparezca en el contexto recuperado, o que contradiga el ejemplo):\n\n"
        f"Pregunta del usuario:\n{user_question}\n\n"
        "Respuesta:"
    )

    return prompt


def answer_with_mori_rag(tokenizer, model, question: str, modo: str = "exacto", k: int = 5, score_threshold: float = 0.88, verbose=True) -> str:
    """
    Mori RAG answer:
      - Detects question_type
      - Rcover docs
      - Filter by question_type
      - Use threshold to determine the answer to return
      - If threshold is surpass → asnwer from FAISS
      - Otherwise → Generative answer from fine tuned Mori
      - Use polish_spanish to return the best possible gramatically corrected asnwer
    """

    # 1) Detectar tipo de pregunta
    qtype = classify_question_type_from_text(question)
    print(f"[Tipo detectado] {qtype}")

    # 2) Recuperar documentos desde FAISS
    docs = retrieve_docs(question, k=k, verbose=False)

    if not docs:
        print("[RAG] No se encontraron documentos, usando prompt simple.")
        prompt = build_prompt_inference(question)
    else:
        # 3) Filtrar por question_type primero
        same_type = [d for d in docs if d.get("question_type") == qtype]

        if same_type:
            top_doc = same_type[0]
        else:
            print("[RAG] No hay docs del mismo question_type, usando top-1 general.")
            top_doc = docs[0]

        if verbose:
            # Debug bonito
            print("\n[RAG] Documento usado como ejemplo:")
            print(" score:", top_doc["score"])
            print(" term :", top_doc.get("canonical_term", ""))
            print(" ctx  :", top_doc.get("context", ""))
            print(" qtype:", top_doc.get("question_type", ""))
            print(" Qej  :", top_doc.get("input", ""))
            print(" Aej  :", top_doc.get("output", ""))

        # 4) Threshold SOLO sobre ese top_doc (idealmente del mismo tipo)
        if top_doc.get("question_type") == qtype and top_doc["score"] >= score_threshold:
            if verbose:
                print(f"[RAG] Coincidencia fuerte (>={score_threshold}) para tipo '{qtype}'. "
                      "Usando output directo del dataset.")
            return polish_spanish(top_doc["output"]), build_prompt_for_mori(question, qtype, top_doc)

        # 5) Si no pasa el threshold → usamos prompt generativo con RAG
        prompt = build_prompt_for_mori(question, qtype, top_doc)

    # 6) Generar con Mori usando el prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(model.device)

    gen_kwargs = get_gen_kwargs(modo)

    output_ids = model.generate(
        **inputs,
        **gen_kwargs
    )
    raw_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 7) Pulir la salida

    return polish_spanish(raw_answer), prompt



def answer_with_mori_plain(tokenizer, model, question: str, modo: str = "exacto") -> str:
    """
    Mori answer without RAG: jsut suing inference prompt with fine tuned model
      - Use polish_spanish to return the best possible gramatically corrected asnwer
    """


    
    prompt = build_prompt_inference(question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=64
    ).to(model.device)

    gen_kwargs = get_gen_kwargs(modo)

    output_ids = model.generate(
        **inputs,
        **gen_kwargs
    )

    raw_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return polish_spanish(raw_answer), prompt


def build_qwen_system_prompt(persona: str) -> str:

    """Generates prompts based on the model personality"""
    
    p = (persona or "").lower()

    base = (
        "Eres Mori Técnico, un asistente de ciencia de datos. "
        "Respondes siempre en español de México, con explicaciones claras y amables. "
    )

    if "exacto" in p:
        return (
            base +
            "Respondes de forma muy breve, directa y precisa, "
            "en un solo párrafo de máximo 64 palabras, sin listas ni numeración."
        )
    elif "creativo" in p:
        return (
            base +
            "Respondes de forma creativa y entusiasta, con un tono cálido y motivador, "
            "en un solo párrafo de máximo 92 palabras, evitando listas y numeración."
        )
    else:
        return (
            base +
            "Respondes de forma breve, clara y natural, "
            "en un solo párrafo y evitando listas y numeración."
        )

def answer_with_qwen_base(
    tokenizer,
    model,
    user_question: str,
    persona: str = "Mori Técnico",
    max_new_tokens: int = 64,
) -> str:
    """
    Genera una respuesta usando Qwen base, sin RAG ni fine-tuning.
    - Ajusta el estilo según la personalidad (exacto / creativo).
    - Usa max_new_tokens para controlar el largo de la respuesta.
    """
    if not user_question.strip():
        return "Necesito que me cuentes algo para poder ayudarte 🙂."

    system_prompt = build_qwen_system_prompt(persona)
    used_chat_template = False

    # 1) Construimos el prompt de texto
    if hasattr(tokenizer, "apply_chat_template"):
        used_chat_template = True
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question.strip()},
        ]
        # devolvemos string, no tensores
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (
            f"system {system_prompt}\n"
            f"user {user_question.strip()}\n"
            f"assistant "
        )

    # 2) Tokenizar el prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    gen_kwargs = get_gen_kwargs(persona)
    # 3) Generar (aquí usamos max_new_tokens que viene de la UI)
    with torch.no_grad():
        if persona == 'exacto':
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.2,
                num_beams=1, 
                top_p=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )

        elif persona =='creativo':
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.9,
                num_beams=1, 
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 4) Recortar el prompt de la salida
    cleaned = text

    if used_chat_template:
        if cleaned.startswith(prompt):
            cleaned = cleaned[len(prompt):].strip()
        else:
            lower = cleaned.lower()
            marker = "assistant"
            idx = lower.rfind(marker)
            if idx != -1:
                cleaned = cleaned[idx + len(marker):].strip()
    else:
        if cleaned.startswith(prompt):
            cleaned = cleaned[len(prompt):].strip()
        else:
            lower = cleaned.lower()
            marker = "assistant"
            idx = lower.rfind(marker)
            if idx != -1:
                cleaned = cleaned[idx + len(marker):].strip()

    cleaned = recortar_ultima_oracion(cleaned)

    return cleaned.strip(), prompt




def get_gen_kwargs(modo="exacto"):

    """Selecting the Mori personaliuty by using different hyperparameters settigns"""
    
    modo = modo.lower().strip()

    presets = {
        "exacto": dict(
            max_new_tokens=64,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            early_stopping=True,
        ),

        "superexacto": dict(   # más estricto, menor creatividad
            max_new_tokens=48,
            num_beams=6,
            do_sample=False,
            no_repeat_ngram_size=4,
            repetition_penalty=1.2,
            early_stopping=True,
        ),

        "creativo": dict(
            max_new_tokens=64,
            num_beams=1,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            early_stopping=True,
        ),

        "suave": dict(      # sampling más libre
            max_new_tokens=80,
            num_beams=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            no_repeat_ngram_size=2,
            repetition_penalty=1.0,
            early_stopping=True,
        ),

        "agresivo": dict(  # máximo sampling creativo
            max_new_tokens=120,
            num_beams=1,
            do_sample=True,
            temperature=1.1,
            top_p=0.95,
            no_repeat_ngram_size=1,
            repetition_penalty=0.9,
            early_stopping=False,
        ),

        "beams_altos": dict( # modo generativo más estable
            max_new_tokens=80,
            num_beams=8,
            do_sample=False,
            no_repeat_ngram_size=4,
            repetition_penalty=1.1,
            early_stopping=True,
        ),
    }

    return presets.get(modo, presets["exacto"])


#=====================================================================================
# FIN  ===============================================================================
#=====================================================================================
