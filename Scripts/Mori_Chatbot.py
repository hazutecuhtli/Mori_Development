'''************************************************************************
Importing Libraries
************************************************************************'''
import  os, sys,warnings,joblib,json,torch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, Dataset
from transformers import pipeline
from unidecode import unidecode
from pathlib import Path
'''************************************************************************
Defining default paths for the model to work
************************************************************************'''
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

'''************************************************************************
Setting up variables
************************************************************************'''
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)
'''************************************************************************
Functions
************************************************************************'''

# Function to look for the folder containing the trained models
def get_model_path(folder_name):

    '''
    inputs:
    
    folder_name --> folder name to look for trained models

    outsputs:

    path --> found folder containing the trained models

    '''
    
    return Path(__file__).resolve().parent.parent / "Models" / folder_name


# Funcion para clasificar las preguntas del usuario definiendo el contexto de las mismas
def classify_context(question, label_classes, model, tokenizer, device):

    '''
    inputs:
    
    question --> Pregunta formulada por el usuario
    label_classes --> Clases del label encoder para decodificar inferencias
    model --> Clasificador para determinar el contexto de las pregutnas
    tokenizer --> Tokenizer usada para clasificar contextos    
    device --> Usar el GPU o el CPU dependiendo de su disponibilidad    

    outsputs:

    predicted_label --> Clasificacion de la pregunta en diversos contextos (clases)

    '''

    # Moviendo el modelo al device disponible
    model = model.to(device)
    
    # Procesando la entrada del usuario
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Clasificacion de la pregunta del usuario en contextos
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Inferencia del clasificador
    pred_intent = torch.argmax(logits, dim=1).item()
    predicted_label = label_classes[pred_intent]
    
    return predicted_label


# Funcion para generar respuestas tecnicas de Mori
def technical_asnwer(question, context, model, tokenizer, device):

    '''
    inputs:
    
    question --> Pregunta formulada por el usuario
    context --> Contexto de la preguntadel usario definido por el clasificador
    model --> Modelo de Mori para responder preguntas tecnicas
    tokenizer --> Tokenizer usado para procesar entradas y decoodificar respuestas
    device --> Usar el GPU o el CPU dependiendo de su disponibilidad

    outsputs:

    response --> Respues de Mori tecnico (Modelo tecnico)

    '''
    
    # Moviendo el modelo al device disponible
    model = model.to(device)    
    
    # Promp Engineering para ayudar a Mori a encontrar la mejor respuesta
    input_text = f"Context: {context} [SEP] Question: {question}"
    
    # Tokenizando el texto de entrada
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generando la respuesta
    summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True)
    
    # Decodificando la respuesta
    response = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return "🧠 [Mori Técnico] " + response.strip()


# Funcion para generar respuestas sociales de Mori
def social_asnwer(question, model, tokenizer, device):

    '''
    inputs:
    
    question --> Pregunta formulada por el usuario
    model --> Modelo de Mori para responder preguntas sociales
    tokenizer --> Tokenizer usado para procesar entradas y decoodificar respuestas    
    device --> Usar el GPU o el CPU dependiendo de su disponibilidad    

    outsputs:

    response --> Respues de Mori social (Modelo social)

    '''

    # Moviendo el modelo al device disponible
    model = model.to(device)

    # Tokenizando la entrada del usuario sin agregar <eos> explícitamente    
    inputs = tokenizer(
        question,  # ✅ sin agregar eos_token
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128  # ✅ especificado para evitar warning
    ).to(device)

    # Generando respuesta usando muestreo
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # ✅ FIX agregado
        max_length=50,
        pad_token_id= tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        top_k=50)

    # Decodificando y limpiando la respuesta
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return "🤝 [Mori Social] " + response.strip()


# Funcion para generar respuesta general de Mori
def contextual_asnwer(question, label_classes, context_model, cont_tok, tec_model, tec_tok, soc_model, soc_tok, device):

    '''
    inputs:
    
    question --> Pregunta formulada por el usuario
    label_classes --> Clases del label encoder para decodificar inferencias
    context_model --> Clasificador para determinar el contexto de las pregutnas
    cont_tok --> Tokenizer usada para clasificar contextos
    tec_model --> Modelo de Mori para responder preguntas tecnicas
    tec_tok -->  Tokenizer usado por Mori Tenico
    soc_model --> Modelo de Mori para responder preguntas sociales
    soc_tok --> Tokenizer usado por Mori Social
    device --> Usar el GPU o el CPU dependiendo de su disponibilidad    

    outsputs:

    response --> Respues de Mori General (Respues con Prompt Engineering)

    '''
    
    # Detectar contexto usando el clasificador
    context = classify_context(question, label_classes, context_model, cont_tok, device)

    context_icons = {"social": "💬",
                     "modelos": "🔧",
                     "evaluación": "📏",
                     "optimización": "⚙️",
                     "visualización": "📈",
                     "aprendizaje": "🧠",
                     "vida digital" : "🧑‍💻",
                     "estadística": "📊",
                     "infraestructura": "🖥",
                     "datos": "📂",
                     "transformación digital": "🌀"}
        
    icon = context_icons.get(context, "🧠")
    print(f"{icon} Contexto detectado: {context}") # (opcional para debug)

    if context == 'social':
        
        # Generar respuesta contextual usando el modelo social
        response = social_asnwer(question, soc_model,soc_tok, device)

    else:      

        # Generar respuesta contextual usando el modelo tecnico
        response = technical_asnwer(question, context, tec_model, tec_tok, device)
    
    return response


'''************************************************************************
MAIN
************************************************************************'''

if __name__ == '__main__':

    #Dispositivo disponible para usar el modelo (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargando las clases usadas prara el Encoder del Clasificador de Contextos
    ModelContextualLabels_Path = os.path.join(sys.path[0], 'Models', "mori-context-model","context_labels.pkl")
    label_classes = joblib.load(ModelContextualLabels_Path)
    
    # Definiendo los paths pata los Modelos de Mori a usar (Clasificador, Tecnico y Social)
    path_classifier = get_model_path("mori-context-model")
    path_social = get_model_path("mori-social-model")
    path_model = get_model_path("mori-tecnico-model")

    # Cargando el Modelo Entrenado Clasificador
    context_model = AutoModelForSequenceClassification.from_pretrained(path_classifier)
    cont_tok = AutoTokenizer.from_pretrained(path_classifier)

    # Cargando el Modelo Mori Social
    soc_model = T5ForConditionalGeneration.from_pretrained(path_social)
    soc_tok = T5Tokenizer.from_pretrained(path_social)

    # Cargando el Modelo Mori Tecnico
    tec_model = T5ForConditionalGeneration.from_pretrained(path_model)
    tec_tok = T5Tokenizer.from_pretrained(path_model)

    # Probando a Mori
    print("\n👋 Hola, soy *Mori*, tu asistente personal de ciencia de datos 🤖")
    print("💬 Puedes preguntarme sobre conceptos técnicos como visualización, limpieza, BI, etc.")
    print("😅 Por el momento, solo puedo contestar preguntas como: ")
    print("🤓  ¿Como estas? ¿Que son?, Explícame algo, Define algo, ¿Para que sirven?")
    print("✏️ Escribe 'salir' para terminar.\n")

    while True:
        entrada = input("\nTú: ")
        if not entrada:
            print("Mori: ¿Podrías repetir eso? No entendí bien 😅")
            continue        
        if entrada.lower() in ["salir", "exit", "quit"]:
            print("👋 ¡Hasta luego! Fue un placer ayudarte.")
            break

        respuesta = contextual_asnwer(entrada, label_classes, context_model, cont_tok, tec_model, tec_tok, soc_model, soc_tok, device)
        print("Mori:", respuesta)


'''************************************************************************
FIN
************************************************************************'''
