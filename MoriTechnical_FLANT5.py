# -*- coding: utf-8 -*-
"""Mori – Inferencia Técnica (estable, UTF-8, con opción RAG ON/OFF)"""
#=====================================================================================
# Importing Libraries  ===============================================================
#=====================================================================================
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from Scripts.Mori_TechnicalPrompts import answer_with_mori_rag, answer_with_mori_plain
import warnings
import torch
from Scripts.config import LLM1_DIR
# ************************************************************************
# Defining default paths for the model to work
# ************************************************************************
model_name = LLM1_DIR

# ************************************************************************
# Setting up variables
# ************************************************************************
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running Device:", device)

#=====================================================================================
# Functions  =========================================================================
#=====================================================================================

def load_mori(model_path=model_name):
    """
    Carga tokenizer y modelo de Mori Técnico.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    return tokenizer, model

def ask_bool(prompt="¿Usar RAG? (s/n): "):
    """
    Pequeña utilidad para leer un booleano desde consola.
    """
    val = input(prompt).strip().lower()
    return val in ["s", "si", "sí", "y", "yes", "1", "true", "t"]


#=====================================================================================
# MAIN  ==============================================================================
#=====================================================================================

if __name__ == "__main__":

    # Presentación de Mori
    print("\n👋 Hola, soy Mori, tu asistente de ciencia de datos 🤖")
    print("💬 Puedo ayudarte en temas técnicos (visualización, limpieza, BI, etc.).")
    print("💬 Por el momento solo puedo contestar los siguientes tipos de preguntas:")
    print("   - Definiciones:   ejemplo -> ¿Qué es machine learning?")
    print("   - Procedimientos: ejemplo -> ¿Cómo limpiar datos?")
    print("   - Funcionalidad:  ejemplo -> ¿Para qué sirve un autoencoder?")
    print("✏️ Escribe 'salir' para terminar.\n")

    # Elegir si usar RAG o no
    USE_RAG = ask_bool("¿Te gustaría usar RAG (búsqueda con FAISS)? (s/n): ")
    print(f"[Config] RAG activado: {USE_RAG}")

    tokenizer, model = load_mori()

    while True:
        try:
            q = input("\nTu pregunta para Mori: ")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 ¡Hasta luego! Fue un placer ayudarte.")
            break

        if q.lower().strip() in ["salir", "exit", "quit"]:
            print("\n👋 ¡Hasta luego! Gracias por usar a Mori.")
            break

        if not q.strip():
            print("Mori: ¿Podrías repetir eso? No entendí bien 😅")
            continue

        # Seleccionar modo de respuesta
        if USE_RAG:
            print("\n[Modo] Mori con RAG")
            answer = answer_with_mori_rag(tokenizer, model, q, modo="exacto",score_threshold=0.84, verbose=False)
        else:
            print("\n[Modo] Mori sin RAG")
            answer = answer_with_mori_plain(tokenizer, model, q, modo="exacto")

        print("\n🔹 Respuesta de Mori:\n", answer[0])

#=====================================================================================
# FIN  ===============================================================================
#=====================================================================================
