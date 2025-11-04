from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
import torch
import os
import shutil
from google.colab import drive

os.environ["WANDB_DISABLED"] = "true"

drive.mount("/content/drive")

# Путь к вашей сохраненной модели
model_path = "/content/drive/MyDrive/my_ai_model"

print("Загрузка датасета с отзывами для дообучения...")
dataset = load_dataset("json", data_files="product_reviews_sentiment_v1.jsonl")

# Посмотрим на структуру новых данных
print("Структура датасета с отзывами:")
print(dataset["train"][0])

# Разделяем данные
train_test = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_val = train_test["train"].train_test_split(test_size=0.1, seed=42)

final_dataset = {
    "train": train_val["train"],
    "validation": train_val["test"], 
    "test": train_test["test"]
}

print(f"Размеры выборок для дообучения: Train: {len(final_dataset['train'])}, Val: {len(final_dataset['validation'])}, Test: {len(final_dataset['test'])}")

print("Загрузка модели и токенизатора...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

def tokenize_function(examples):
    # АДАПТИРУЕМ ПОД ФОРМАТ ОТЗЫВОВ, НО СОХРАНЯЯ СТРУКТУРУ
    if 'review' in examples and 'sentiment' in examples:
        # Преобразуем в формат "Вопрос-Ответ" для совместимости
        inputs = [f"Вопрос: Проанализируй тональность отзыва: {review}\nОтвет: Тональность отзыва - {sentiment}{tokenizer.eos_token}" 
                 for review, sentiment in zip(examples["review"], examples["sentiment"])]
    
    elif 'review' in examples:
        # Если есть только отзыв
        inputs = [f"Вопрос: Проанализируй этот отзыв: {review}\nОтвет: Это отзыв о продукте.{tokenizer.eos_token}" 
                 for review in examples["review"]]
    
    else:
        # Базовый вариант
        text_column = list(examples.keys())[0]
        inputs = [f"Вопрос: Проанализируй текст: {text}\nОтвет: {text}{tokenizer.eos_token}" 
                 for text in examples[text_column]]
    
    tokenized = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors=None
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("Токенизация данных...")
tokenized_datasets = {
    split: final_dataset[split].map(
        tokenize_function, 
        batched=True, 
        remove_columns=final_dataset[split].column_names
    )
    for split in ["train", "validation", "test"]
}

# Настройка обучения
training_args = TrainingArguments(
    output_dir="./my_ai_model_reviews",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=50,
    eval_steps=50,
    save_steps=50,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    prediction_loss_only=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

print("Запуск дообучения на отзывах...")
train_results = trainer.train()

# Сохраняем модель
trainer.save_model()
tokenizer.save_pretrained("./my_ai_model_reviews")

print("="*60)
print("ТЕСТИРОВАНИЕ ДООБУЧЕННОЙ МОДЕЛИ")
print("="*60)

model_reviews = GPT2LMHeadModel.from_pretrained("./my_ai_model_reviews")
model_reviews.eval()

# ТЕСТИРУЕМ НА ОТЗЫВАХ (новые возможности)
print("=== ТЕСТ НА ОТЗЫВАХ ===")
review_prompts = [
    "Вопрос: Проанализируй тональность отзыва: Этот продукт просто потрясающий! Качество на высоте.\nОтвет:",
    "Вопрос: Проанализируй тональность отзыва: Разочарован покупкой, не оправдал ожидания.\nОтвет:",
    "Вопрос: Проанализируй этот отзыв: Нормальный товар за свои деньги.\nОтвет:",
    "Вопрос: Проанализируй тональность: Отличное качество, быстрая доставка, рекомендую!\nОтвет:",
    "Вопрос: Проанализируй отзыв: Ужасное качество, деньги на ветер.\nОтвет:"
]

for i, prompt in enumerate(review_prompts, 1):
    print(f"{i}. Промпт: {prompt}")
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model_reviews.generate(
            inputs,
            max_length=inputs.shape[1] + 50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлекаем только ответ
    if "Ответ:" in generated_text:
        answer = generated_text.split("Ответ:")[-1].strip()
        print(f"Ответ: {answer}")
    else:
        print(f"Полный текст: {generated_text}")
    print("-" * 50)

# ТЕСТИРУЕМ НА ПРОГРАММИРОВАНИИ (старые возможности)
print("\n=== ТЕСТ НА ПРОГРАММИРОВАНИИ ===")
programming_prompts = [
    "Вопрос: Что такое Питон?\nОтвет:",
    "Вопрос: Как объявить переменную в Питон?\nОтвет:",
    "Вопрос: Что такое ООП?\nОтвет:",
    "Вопрос: Как создать функцию в Питон?\nОтвет:"
]

for i, prompt in enumerate(programming_prompts, 1):
    print(f"{i}. Промпт: {prompt}")
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model_reviews.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Ответ:" in generated_text:
        answer = generated_text.split("Ответ:")[-1].strip()
        print(f"Ответ: {answer}")
    else:
        print(f"Полный текст: {generated_text}")
    print("-" * 50)

print("Сохранение в Google Drive...")
drive_reviews_path = "/content/drive/MyDrive/my_ai_model_reviews"
shutil.copytree("./my_ai_model_reviews", drive_reviews_path, dirs_exist_ok=True)
print(f"Модель сохранена в Google Drive: {drive_reviews_path}")

print("="*60)
print("ДООБУЧЕНИЕ ЗАВЕРШЕНО!")
print("="*60)
