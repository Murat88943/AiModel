from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer
import torch
import os
import shutil
from google.colab import drive

os.environ["WANDB_DISABLED"] = "true"

drive.mount("/content/drive")

print("Загрузка датасета...")
dataset = load_dataset("json", data_files="/content/dataset.jsonl")

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

print("Инициализация токенизатора...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    inputs = [f"Вопрос: {q}\nОтвет: {a}{tokenizer.eos_token}" for q, a in zip(examples["question"], examples["answer"])]
    
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

print("Создание модели GPT-2...")

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=256,
    n_embd=512,
    n_layer=8,
    n_head=8,
    n_inner=2048,
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

model = GPT2LMHeadModel(config)
print(f"Модель создана! Параметров: {model.num_parameters():,}")

print("Настройка обучения...")

training_args = TrainingArguments(
    output_dir="./my_ai_model",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_steps=200,
    save_steps=400,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    prediction_loss_only=True,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
    report_to="none",
    save_total_limit=3,
    max_grad_norm=1.0,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

print("Запуск обучения...")
train_results = trainer.train()

trainer.save_model()
tokenizer.save_pretrained("./my_ai_model")
print("Модель сохранена в './my_ai_model'")

print("="*60)
print("ТЕСТИРОВАНИЕ МОДЕЛИ")
print("="*60)

model = GPT2LMHeadModel.from_pretrained("./my_ai_model")
tokenizer = GPT2Tokenizer.from_pretrained("./my_ai_model")
model.eval()

test_prompts = [
    "Вопрос: Как найти смысл жизни?",
    "Вопрос: Что такое ООП?",
    "Вопрос: Ты не видел мои ключи?",
    "Вопрос: Как работает искусственный интеллект?",
    "Вопрос: Какие языки программирования самые популярные?",
    "Вопрос: Как твои выходные?",
    "Вопрос: Что посоветуешь почитать?",
    "Вопрос: Как научиться программировать?"
]

print("Генерация ответов...")

for i, prompt in enumerate(test_prompts, 1):
    print(f"{i}. {prompt}")
    
    formatted_prompt = f"{prompt}\nОтвет:"
    
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Ответ:" in generated_text:
        answer = generated_text.split("Ответ:")[-1].strip()
        print(f"Ответ: {answer}")
    else:
        print(f"Текст: {generated_text}")
    
    print("-" * 80)

print("Сохранение в Google Drive...")
drive_path = "/content/drive/MyDrive/my_ai_model"
shutil.copytree("./my_ai_model", drive_path, dirs_exist_ok=True)
print(f"Модель сохранена в Google Drive: {drive_path}")

print("="*60)
print("ФИНАЛЬНАЯ ОЦЕНКА")
print("="*60)

eval_results = trainer.evaluate(tokenized_datasets["test"])
print(f"Потери на тестовой выборке: {eval_results['eval_loss']:.4f}")

perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
print(f"Перплексия на тестовой выборке: {perplexity:.4f}")

print("="*60)
print("СТАТИСТИКА ОБУЧЕНИЯ")
print("="*60)

print(f"Размер датасета: {len(final_dataset['train'])} примеров")
print(f"Количество эпох: 10")
print(f"Общее количество шагов: {trainer.state.max_steps}")

if trainer.state.log_history:
    train_losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    
    if train_losses:
        print(f"Начальные потери: {train_losses[0]:.4f}")
        print(f"Финальные потери: {train_losses[-1]:.4f}")
    
    if eval_losses:
        print(f"Лучшие потери валидации: {min(eval_losses):.4f}")

print("="*60)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО! ВАША ИИ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
print("="*60)

print("Советы для дальнейшего улучшения:")
print("1. Увеличьте размер датасета")
print("2. Экспериментируйте с параметрами модели")
print("3. Увеличьте количество эпох при необходимости")
print("4. Добавьте больше разнообразных примеров")
print("5. Настройте параметры генерации под ваши задачи")
