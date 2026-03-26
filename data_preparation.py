import json
import pandas as pd
import re

def clean_text(text):
    """
    Базове очищення юридичного тексту від зайвих пробілів та спецсимволів.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_instruction_format(document, summary):
    """
    Формує словник у форматі інструкцій для донавчання моделі (Alpaca/Mistral format).
    """
    system_prompt = "Виділи головну думку з цього тексту одним коротким реченням. Не додавай нічого від себе."
    
    # Формування єдиного рядка для навчання (формат Mistral Instruct)
    formatted_text = f"<s>[INST] {system_prompt}\n\nТекст:\n{document} [/INST] {summary} </s>"
    
    return {"text": formatted_text}

def prepare_dataset(output_jsonl_path):
    """
    Імітує завантаження сирих даних та їх перетворення у JSONL датасет для Fine-Tuning.
    У реальному проєкті тут може бути pd.read_csv('raw_legal_data.csv').
    """
    print("Запуск пайплайну підготовки даних...")
    
    # Приклад наших еталонних даних (Gold Standard)
    raw_data = [
        {
            "doc": "КМУ скасував часові обмеження щодо увімкнення денних ходових вогнів або ближнього світла фар поза населеними пунктами. Відтепер ця вимога діє цілий рік для всіх механічних транспортних засобів.",
            "sum": "КМУ скасував часові обмеження щодо увімкнення денних ходових вогнів... Відтепер ця вимога діє цілий рік для всіх транспортних засобів."
        },
        {
            "doc": "Договір оренди будівлі завжди укладається письмово. Обов'язковому нотаріальному посвідченню підлягають договори зі строком оренди від 3 років (для державного чи комунального майна на аукціоні — зі строком понад 5 років).",
            "sum": "Договір найму будівлі або іншої капітальної споруди укладається письмово та підлягає нотаріальному посвідченню, якщо строк дії договору більше трьох років."
        }
        # У реальному скрипті тут будуть сотні рядків з CSV або бази даних
    ]
    
    df = pd.DataFrame(raw_data)
    print(f"Знайдено записів для обробки: {len(df)}")
    
    # Збереження у формат JSONL (кожен рядок - це окремий валідний JSON)
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            cleaned_doc = clean_text(row['doc'])
            cleaned_sum = clean_text(row['sum'])
            
            instruction_data = create_instruction_format(cleaned_doc, cleaned_sum)
            f.write(json.dumps(instruction_data, ensure_ascii=False) + '\n')
            
    print(f"Датасет успішно збережено у {output_jsonl_path}")

if __name__ == "__main__":
    # Запуск скрипта
    prepare_dataset("legal_dataset_finetuning.jsonl")