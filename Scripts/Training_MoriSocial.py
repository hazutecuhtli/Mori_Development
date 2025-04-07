'''************************************************************************
Importing Libraries
************************************************************************'''
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import TrainerCallback
from transformers import EarlyStoppingCallback
from datasets import load_dataset, Dataset
from transformers import pipeline
from unidecode import unidecode
from datetime import datetime
import json, torch, os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
'''************************************************************************
Setting up variables
************************************************************************'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

path_Mori_Social = os.path.join(sys.path[0], 'Data', 'Conceptos_Curados_Manualmente_Social.json')
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


# Cargar modelo y tokenizer base
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Cargar dataset JSON
dataset = load_dataset("json", data_files=path_Mori_Social)["train"]

# Preprocesamiento
def preprocess(example):
    # Agregar un prefijo de tarea para que mT5 entienda que debe generar una respuesta social
    prefijo = "respuesta social: "
    input_text = prefijo + example["input"]
    target_text = example["output"]

    # Tokenizar entrada
    model_inputs = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=64
    )

    # Tokenizar salida y reemplazar padding por -100 (para ignorarlo en el cálculo de la pérdida)
    with tokenizer.as_target_tokenizer():  # no es obligatorio con mT5, pero ayuda a claridad
        labels = tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=64
        )

    labels["input_ids"] = [
        token if token != tokenizer.pad_token_id else -100
        for token in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Preparar el data collator correcto para tareas de secuencia
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir="./mori_social_model_tmp",
    num_train_epochs=140,
    per_device_train_batch_size=12,
    learning_rate=1e-4,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=150,
    save_steps=200,
    logging_strategy="epoch",         # <--- aquí va
    fp16=False,  # importante: evitar pérdida silenciosa
)

# Crear el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    callbacks=[
        FriendlyProgressCallback()]
)

# Entrenar
trainer.train()

# Guardar modelo y tokenizer
ModelSocial_Path = os.path.join(sys.path[0], 'Models', "mori-social-model")
os.makedirs(ModelSocial_Path, exist_ok=True)
timestamp = datetime.now()
trainer.model.save_pretrained(ModelSocial_Path)
tokenizer.save_pretrained(ModelSocial_Path)

# 📄 Crear README con info del entrenamiento
readme_path = os.path.join(ModelSocial_Path, "README.txt")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(f"Mori Social - Modelo entrenado\n")
    f.write(f"Fecha de entrenamiento: {timestamp}\n")
    f.write(f"\nHiperparámetros principales:\n")
    f.write(f"- Epochs: {training_args.num_train_epochs}\n")
    f.write(f"- Learning Rate: {training_args.learning_rate}\n")
    f.write(f"- Batch Size: {training_args.per_device_train_batch_size}\n")
    f.write(f"- Weight Decay: {training_args.weight_decay}\n")
    f.write(f"- Mejora seleccionada por: {training_args.metric_for_best_model}\n")
    f.write(f"- Estrategia de guardado: {training_args.save_strategy}\n")
    f.write(f"\n¡Entrenamiento realizado con ayuda de GPT y un gran crack humano!\n")

print(f"✅ Mori Social guardado con README en: {ModelSocial_Path}")



'''************************************************************************
FIN
************************************************************************'''