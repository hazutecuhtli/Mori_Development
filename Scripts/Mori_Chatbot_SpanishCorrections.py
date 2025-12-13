# -*- coding: utf-8 -*-
"""Correct Spanish sentences by fixing punctuation and restoring missing characters."""
#=====================================================================================
# Importing Libraries  ===============================================================
#=====================================================================================
import os, sys, warnings, json, joblib, random, re, unicodedata
import numpy as np

#=====================================================================================
# Custom functions
#=====================================================================================

# Normalize and correct common Spanish spelling, accents, and punctuation in Mori outputs

def polish_spanish(s: str) -> str:
    """
    Normalize and correct common Spanish punctuation and missing characters.

    This function applies deterministic, rule-based post-processing intended to
    improve the readability of Mori outputs. It normalizes Unicode characters,
    removes filtered model tags, fixes frequent spelling/accent issues, and
    enforces basic Spanish punctuation rules (opening question/exclamation marks
    and a final sentence terminator).

    Args:
        s: Input text to be normalized.

    Returns:
        A cleaned and normalized Spanish string.
    """
    if not s:
        return ""

    # Normalize Unicode (accents and ñ) and remove leading/trailing whitespace
    s = unicodedata.normalize("NFC", s).strip()

    # Remove model tags (e.g., "[Mori Técnico]" or "(Mori Social)")
    s = re.sub(
        r"\s*[\[\(]\s*Mori\s+(?:Social|T[eé]nico|T[eé]cnico)\s*[\]\)]\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )

    # Common spelling/diacritics fixes (rule-based)
    fixes = [
        (r"(?i)(^|\W)T\s+puedes(?P<p>[^\w]|$)", r"\1Tú puedes\g<p>"),
        (r"(?i)(^|\W)T\s+(ya|eres|estas|estás|tienes|puedes)\b", r"\1Tú \2"),
        (r"(?i)\bclaro que s(?:i|í)?\b(?P<p>[,.\!?…])?", r"Claro que sí\g<p>"),
        (r"(?i)(^|\s)si,", r"\1Sí,"),
        (r"(?i)(\beso\s+)s(\s+est[áa]\b)", r"\1sí\2"),
        (r"(?i)(^|[\s,;:])s(\s+es\b)", r"\1sí\2"),
        (r"(?i)\btiles\b", "útiles"),
        (r"(?i)\butiles\b", "útiles"),
        (r"(?i)\butil\b", "útil"),
        (r"(?i)\btil\b", "útil"),
        (r"(?i)\batpico\b", "atípico"),
        (r"(?i)\baqui\b", "aquí"),
        (r"(?i)\balgn\b", "algún"),
        (r"(?i)\banomala\b", "anomalía"),
        (r"(?i)\banomalas\b", "anomalías"),
        (r"(?i)\balgun\b", "algún"),
        (r"(?i)\bAnimo\b", "Ánimo"),
        (r"(?i)\bcario\b", "cariño"),
        (r"(?i)\baprendisaje\b", "aprendizaje"),
        (r"(?i)\bmanana\b", "mañana"),
        (r"(?i)\bmaana\b", "mañana"),
        (r"(?i)\benergia\b", "energía"),
        (r"(?i)\benerga\b", "energía"),
        (r"(?i)\bextrano\b", "extraño"),
        (r"(?i)\bextrana\b", "extraña"),
        (r"(?i)\bextranar\b", "extrañar"),
        (r"(?i)\bextranarte\b", "extrañarte"),
        (r"(?i)\bextranas\b", "extrañas"),
        (r"(?i)\bextranos\b", "extraños"),
        (r"(?i)\baqu\b", "aquí"),
        (r"(?i)\bestare\b", "estaré"),
        (r"(?i)\bclarn\b", "clarín"),
        (r"(?i)\bclarin\b", "clarín"),
        (r"(?i)\bclar[íi]n\s+cornetas\b", "clarín cornetas"),
        (r"(?i)(^|\s)s([,.;:!?])", r"\1Sí\2"),
        (r"(?i)\bfutbol\b", "fútbol"),
        (r"(?i)(^|\s)as(\s+se\b)", r"\1Así\2"),
        (r"(?i)(^|\s)s(\s+orientarte\b)", r"\1sí\2"),
        (r"(?i)\bbuen dia\b", "buen día"),
        (r"(?i)\bgran dia\b", "gran día"),
        (r"(?i)\bdias\b", "días"),
        (r"(?i)\bdia\b", "día"),
        (r"(?i)\bgran da\b", "gran día"),
        (r"(?i)\bacompa?a(r|rte|do|da|dos|das)?\b", r"acompaña\1"),
        (r"(?i)(^|\s)as([,.;:!?]|\s|$)", r"\1así\2"),
        (r"(?i)(^|\s)S lo se\b", r"\1Sí lo sé"),
        (r"(?i)(^|\s)S lo sé\b", r"\1Sí lo sé"),
        (r"(?i)\bcuidate\b", "cuídate"),
        (r"(?i)\bcuidese\b", "cuídese"),
        (r"(?i)\bcuidense\b", "cuídense"),
        (r"(?i)\bpequeo\b", "pequeño"),
        (r"(?i)\bpequea\b", "pequeña"),
        (r"(?i)\bpequeos\b", "pequeños"),
        (r"(?i)\bpequeas\b", "pequeñas"),
        (r"(?i)\bunico\b", "único"),
        (r"(?i)\bunica\b", "única"),
        (r"(?i)\bunicos\b", "únicos"),
        (r"(?i)\bunicas\b", "únicas"),
        (r"(?i)\bnico\b", "único"),
        (r"(?i)\bnica\b", "única"),
        (r"(?i)\bnicos\b", "únicos"),
        (r"(?i)\bnicas\b", "únicas"),
        (r"(?i)\bestadstico\b", "estadístico"),
        (r"(?i)\bestadstica\b", "estadística"),
        (r"(?i)\bestadsticos\b", "estadísticos"),
        (r"(?i)\bestadsticas\b", "estadísticas"),
        (r"(?i)\bgracias por confiar en m\b", "gracias por confiar en mí"),
        (r"(?i)\bcada dia\b", "cada día"),
        (r"(?i)\bcada da\b", "cada día"),
        (r"(?i)\bsegun\b", "según"),
        (r"(?i)\bcaracteristica(s)?\b", r"característica\1"),
        (r"(?i)\bcaracterstica(s)?\b", r"característica\1"),
        (r"(?i)\b([a-záéíóúñ]+)cion\b", r"\1ción"),
        (r"(?i)\bdeterminacio\b", "determinación"),
    ]

    for pat, rep in fixes:
        s = re.sub(pat, rep, s)

    # Normalize a common closing phrase
    s = re.sub(r"(?i)^eso es todo!(?P<r>(\s|$).*)", r"¡Eso es todo!\g<r>", s)

    # Add opening question mark when a question ends with "?" but lacks "¿"
    def _add_opening_question(m: re.Match) -> str:
        body = m.group("qbody")
        if "¿" in body:
            return m.group(0)
        return f"{m.group('pre')}¿{body}"

    s = re.sub(
        r"(?P<pre>(^|[.!…]\s+))(?P<qbody>[^?]*\?)",
        _add_opening_question,
        s,
    )

    # Add opening exclamation mark for common expressions ending with "!"
    def _open_exclamation(m: re.Match) -> str:
        word = m.group("w")
        rest = m.group("r") or ""
        return f"¡{word}!{rest}"

    s = re.sub(
        r"(?i)^\s*(?P<w>(hola|gracias|genial|perfecto|claro|por supuesto|con gusto|listo|vaya|wow|tu puedes|tú puedes|clarín|clarin|clarín cornetas))!(?P<r>(\s|$).*)",
        _open_exclamation,
        s,
    )

    # Final cleanup: collapse whitespace and ensure a sentence terminator
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".!?…":
        s += "."

    return s

#=====================================================================================
# Fin
#=====================================================================================
