'''************************************************************************
Importing Libraries
************************************************************************'''
import json, torch, os, sys, warnings, joblib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
from transformers import pipeline
from unidecode import unidecode
from datetime import datetime
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore", category=FutureWarning)
# Guardar las clases del LabelEncoder
'''************************************************************************
Setting up variables
************************************************************************'''
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

path_Mori_Classifier = os.path.join(sys.path[0], 'Data', 'Dataset_Social_ParafrasisTecnico_Unificado.json')
'''************************************************************************
Functions
************************************************************************'''

class FriendlyProgressCallback(TrainerCallback):
    def __init__(self):
        self.last_eval_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):

        if not logs:
            return

        epoch = state.epoch
        train_loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        lr = logs.get("learning_rate")

        # Compute eval loss change
        if eval_loss is not None and self.last_eval_loss is not None:
            delta = eval_loss - self.last_eval_loss
        else:
            delta = None

        self.last_eval_loss = eval_loss if eval_loss is not None else self.last_eval_loss

        # Pretty print
        msg = f"📘 Epoch {epoch:.1f} | "
        if train_loss:
            msg += f"Train Loss: {train_loss:.4f} | "
        if eval_loss:
            msg += f"Eval Loss: {eval_loss:.4f} "
            if delta:
                msg += f"({'↓' if delta < 0 else '↑'} {abs(delta):.4f})"
        if lr:
            msg += f" | LR: {lr:.2e}"

        print(msg)


# Cargar el dataset técnico
with open(path_Mori_Classifier, "r", encoding="utf-8") as f:
    data = json.load(f)
   
# Convertir el archivo JSON en un DataFrame
df = pd.DataFrame(data)

# Cargar el tokenizador BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenizar las preguntas
def tokenize_function(examples):
    return tokenizer(examples['input'], padding=True, truncation=True, return_tensors="pt")

# Aplicar tokenización
tokenized_inputs = df['input'].apply(lambda x: tokenize_function({"input": x}))

# Codificar las etiquetas de contexto en números
label_encoder = LabelEncoder()
df['context_label'] = label_encoder.fit_transform(df['context'])

# Dividir el dataset en entrenamiento y prueba
train_df, test_df = train_test_split(df, test_size=0.2)

class ContextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Tokenizar y crear datasets de entrenamiento y prueba
train_encodings = tokenizer(list(train_df['input']), truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(list(test_df['input']), truncation=True, padding=True, return_tensors='pt')

train_labels = list(train_df['context_label'])
test_labels = list(test_df['context_label'])

# Crear los datasets
train_dataset = ContextDataset(train_encodings, train_labels)
test_dataset = ContextDataset(test_encodings, test_labels)

# Cargar el modelo BERT preentrenado para clasificación de texto
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Mover el modelo a la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Definir los argumentos para el entrenamiento
training_args = TrainingArguments(
    output_dir='./mori_context_model_tmp',
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="epoch",         # <--- aquí va
    evaluation_strategy="epoch",      # <--- opcional pero útil
)

# Definir el entrenador
trainer = Trainer(
    model=model,                         # el modelo a entrenar
    args=training_args,                  # los parámetros de entrenamiento
    train_dataset=train_dataset,         # conjunto de datos de entrenamiento
    eval_dataset=test_dataset            # conjunto de datos de prueba
)

# Entrenar el modelo
trainer.train()

# Guardar modelo
ModelContextual_Path = os.path.join(sys.path[0], 'Models', "mori-context-model")
ModelContextualLabels_Path = os.path.join(sys.path[0], 'Models', "mori-context-model","context_labels.pkl")
os.makedirs(ModelContextual_Path, exist_ok=True)

model.save_pretrained(ModelContextual_Path)
tokenizer.save_pretrained(ModelContextual_Path)
joblib.dump(label_encoder.classes_, ModelContextualLabels_Path)

# Evaluar el modelo
results = trainer.evaluate()

# Función para predecir el contexto de una nueva pregunta
def predict_context(question):
    inputs = tokenizer(question, return_tensors='pt', truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1)
    predicted_label = label_encoder.inverse_transform(predicted_class.cpu().numpy())
    return predicted_label[0]

# Probar con una nueva pregunta
new_question = "¿Qué es una violin plot?"
predicted_context = predict_context(new_question)
print(f"La pregunta pertenece al contexto: {predicted_context}")



'''************************************************************************
FIN
************************************************************************'''