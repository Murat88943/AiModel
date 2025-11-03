from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer
import torch
import os
from google.colab import drive

# Отключаем WandB
os.environ["WANDB_DISABLED"] = "true"

# Подключаем Google Drive
drive.mount("/content/drive")

# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
print("Загрузка датасета...")
dataset = load_dataset("json", data_files="/content/dataset.jsonl")

# Разделение на train/validation/test
train_test = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_val = train_test["train"].train_test_split(test_size=0.1, seed=42)

final_dataset = {
    "train": train_val["train"],
    "validation": train_val["test"], 
    "test": train_test["test"]
}

print("Размеры выборок:")
print(f"Train: {len(final_dataset['train'])}")
print(f"Validation: {len(final_dataset['validation'])}")
print(f"Test: {len(final_dataset['test'])}")

# 2. ИНИЦИАЛИЗАЦИЯ ТОКЕНИЗАТОРА
print("Инициализация токенизатора...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 3. ТОКЕНИЗАЦИЯ ДАННЫХ
def tokenize_function(examples):
    # Формируем текст: вопрос + ответ
    inputs = [f"Вопрос: {q}\nОтвет: {a}" for q, a in zip(examples["question"], examples["answer"])]
    
    # Токенизируем
    tokenized = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors=None
    )
    
    # Для языкового моделирования метки = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("Токенизация данных...")
tokenized_datasets = {
    split: final_dataset[split].map(tokenize_function, batched=True, remove_columns=final_dataset[split].column_names)
    for split in ["train", "validation", "test"]
}

# 4. СОЗДАНИЕ АРХИТЕКТУРЫ МОДЕЛИ
print("Создание модели с нуля...")

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=256,
    n_embd=384,
    n_layer=6,
    n_head=6,
    n_inner=1536,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# Создаем модель с нуля
model = GPT2LMHeadModel(config)

print(f"Модель создана! Параметров: {model.num_parameters():,}")

# 5. ОБУЧЕНИЕ МОДЕЛИ
print("Настройка обучения...")

training_args = TrainingArguments(
    output_dir="./my_ai_model",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    prediction_loss_only=True,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
    report_to="none",  # Отключаем все репорты
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Обучаем модель
print("Запуск обучения...")
train_results = trainer.train()

# Сохраняем модель
trainer.save_model()
tokenizer.save_pretrained("./my_ai_model")
print("Модель сохранена в './my_ai_model'")

# 6. ТЕСТИРОВАНИЕ МОДЕЛИ
print("\nТестирование модели...")

from transformers import pipeline

# Перезагружаем модель для тестирования
model = GPT2LMHeadModel.from_pretrained("./my_ai_model")
tokenizer = GPT2Tokenizer.from_pretrained("./my_ai_model")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Тестовые промпты
test_prompts = [
    "Вопрос: Как найти смысл жизни?\nОтвет:",
    "Вопрос: Что такое ООП?\nОтвет:",
    "Вопрос: Ты не видел мои ключи?\nОтвет:",
    "Вопрос: Как работает искусственный интеллект?\nОтвет:"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{i}. Промпт: {prompt}")
    try:
        outputs = generator(
            prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        for output in outputs:
            generated_text = output['generated_text']
            # Показываем только ответ (после "Ответ:")
            if "Ответ:" in generated_text:
                answer = generated_text.split("Ответ:")[-1].strip()
                print(f"Ответ: {answer}")
            else:
                print(f"Полный текст: {generated_text}")
            print("-" * 80)
    except Exception as e:
        print(f"Ошибка генерации: {e}")

# 7. СОХРАНЕНИЕ В GOOGLE DRIVE
import shutil

drive_path = "/content/drive/MyDrive/my_ai_model"
shutil.copytree("./my_ai_model", drive_path, dirs_exist_ok=True)
print(f"Модель сохранена в Google Drive: {drive_path}")

print("\n🎉 Обучение завершено! Ваша ИИ готова к использованию!")
