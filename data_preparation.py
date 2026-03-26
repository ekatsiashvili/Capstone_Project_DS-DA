import json
import re
import os

def clean_text(text):
    """
    Очищення юридичного тексту від зайвих пробілів, html-тегів та перенесень рядків.
    """
    if not isinstance(text, str):
        return ""
    # Видаляємо множинні пробіли та специфічні символи розмітки
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\\n', ' ', text)
    return text.strip()

def prepare_training_dataset(input_jsonl_path, output_json_path):
    """
    Зчитує сирий датасет (legal_dataset_raw.jsonl), очищає його і перетворює 
    у формат Alpaca/Mistral Instruct (train.json) для подальшого Fine-Tuning.
    """
    print(f"⏳ Зчитування сирого датасету з {input_jsonl_path}...")
    
    if not os.path.exists(input_jsonl_path):
        print(f"❌ Помилка: Файл {input_jsonl_path} не знайдено.")
        return

    formatted_data = []
    # Системний промпт, який використовуємо для навчання
    instruction_prompt = "Створи стисле юридичне резюме (summary) для наступного документу, виділивши ключові зміни або рішення."
    
    processed_count = 0
    
    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                raw_record = json.loads(line)
                
                # Витягуємо повний текст закону
                doc_input = clean_text(raw_record.get('full_text', ''))
                
                doc_output = clean_text(raw_record.get('title', '')) 
                
                # Формуємо структуру для Unsloth / Mistral
                if doc_input and len(doc_input) > 50 and doc_output:
                    formatted_data.append({
                        "instruction": instruction_prompt,
                        "input": doc_input,
                        "output": doc_output
                    })
                    processed_count += 1

        # Зберігаємо у файл train.json
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
           
            json.dump(formatted_data, f_out, ensure_ascii=False, indent=2)
            
        print(f"📊 Оброблено документів: {processed_count}")
        print(f"✅ Датасет успішно очищено та збережено у файл: {output_json_path}")
        print("Готово до завантаження в Google Colab для Fine-Tuning!")
        
    except Exception as e:
        print(f"❌ Сталася помилка під час обробки даних: {e}")

if __name__ == "__main__":
    RAW_DATA_FILE = "data/legal_dataset_raw.jsonl"
    FINAL_TRAINING_FILE = "data/train.json"
    
    prepare_training_dataset(RAW_DATA_FILE, FINAL_TRAINING_FILE)