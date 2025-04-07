'''************************************************************************
Importing Libraries
************************************************************************'''
import json, torch, os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import TrainerCallback
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
from datasets import load_dataset, Dataset
from transformers import pipeline
from unidecode import unidecode
from datetime import datetime
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
'''************************************************************************
Setting up variables
************************************************************************'''
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
'''************************************************************************
Functions
************************************************************************'''

# ✅ Corrige la ruta correctamente desde Scripts hacia Models
def get_model_path(folder_name):
    return Path(__file__).resolve().parent.parent / "Models" / folder_name

path2classifier = get_model_path("mori-context-model")
path_Mori_TrainingData = os.path.join(sys.path[0], 'Data', 'Parafrasis_Conceptos_Curados_Manualmente_Tecnicos.json')


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
with open(path_Mori_TrainingData, "r", encoding="utf-8") as f:
    data = json.load(f)

# Preparar dataset Hugging Face
dataset = Dataset.from_list([
    {
        "input_text": f"Context: {item['context']} [SEP] Question: {item['input']}",  # Usar un separador explícito
        "output_text": item['output']
    }
    for item in data
])

print(dataset.column_names)

dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Ver el primer ejemplo para asegurarnos de que la estructura sea correcta
print(dataset['train'][0])

# Tokenizer y modelo
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, ignore_mismatched_sizes=True)

# Preprocesar con el contexto
# Función de preprocesamiento para tokenizar los datos
def preprocess(examples):
    model_inputs = tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(examples['output_text'], padding="max_length", truncation=True, max_length=128)
    
    model_inputs['labels'] = labels.input_ids
    return model_inputs

# Preprocesar el dataset
#tokenized_datasets = dataset.map(preprocess_function, batched=True)

tokenized = dataset.map(preprocess, remove_columns=["input_text", "output_text"])

# Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir="./mori_tecnico_model_tmp",  # Carpeta de salida para el modelo
    eval_strategy="epoch",         # Evalúa cada época
    save_strategy="epoch",               # Guarda checkpoints por época
    num_train_epochs=20,                 # Número máximo de épocas
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    learning_rate=3e-4,                  # Excelente valor para fine-tuning
    weight_decay=0.15,                    # Regularización para evitar overfitting
    logging_strategy="epoch",           # Registra por época
    save_total_limit=2,                 # Solo guarda los 2 mejores checkpoints
    fp16=torch.cuda.is_available(),     # Usa precision mixta si hay GPU
    report_to="none",                   # No conecta con WandB o TensorBoard
    load_best_model_at_end=True,       # ✅ ¡Crucial! Para usar el mejor modelo
    metric_for_best_model="eval_loss",  # Evalúa con base en pérdida de validación
    disable_tqdm=False,
    greater_is_better=False             # ✅ Porque menos pérdida es mejor
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
    callbacks=[
        FriendlyProgressCallback(),
        EarlyStoppingCallback(early_stopping_patience=3)]
)

# Entrenamiento
trainer.train()

# Guardar modelo
ModelTecnico_Path = os.path.join(sys.path[0], 'Models', "mori-tecnico-model")
os.makedirs(ModelTecnico_Path, exist_ok=True)

trainer.model.save_pretrained(ModelTecnico_Path)
timestamp = datetime.now()
trainer.model.save_pretrained(ModelTecnico_Path)
tokenizer.save_pretrained(ModelTecnico_Path)

# 📄 Crear README con info del entrenamiento
readme_path = os.path.join(ModelTecnico_Path, "README.txt")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(f"Mori Técnico - Modelo entrenado\n")
    f.write(f"Fecha de entrenamiento: {timestamp}\n")
    f.write(f"\nHiperparámetros principales:\n")
    f.write(f"- Epochs: {training_args.num_train_epochs}\n")
    f.write(f"- Learning Rate: {training_args.learning_rate}\n")
    f.write(f"- Batch Size: {training_args.per_device_train_batch_size}\n")
    f.write(f"- Weight Decay: {training_args.weight_decay}\n")
    f.write(f"- Mejora seleccionada por: {training_args.metric_for_best_model}\n")
    f.write(f"- Estrategia de guardado: {training_args.save_strategy}\n")
    f.write(f"\n¡Entrenamiento realizado con ayuda de GPT y un gran crack humano!\n")

print(f"✅ Mori Técnico guardado con README en: {ModelTecnico_Path}")


'''************************************************************************
FIN
************************************************************************'''