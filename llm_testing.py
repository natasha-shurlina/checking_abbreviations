#python3 llm_testing.py -i ex1.txt -r rules.yaml -e extra_allowed.txt -o result_llm.json
import json
import argparse
from pathlib import Path
import yaml

PART_SIZE = 1500
PART_OVERLAP = 100

def load_allowed(rules_path: str, extra_path: str = None) -> list:
    # Собирает список правильных сокращений из rules.yaml и из extra_allowed.txt.
    allowed = []

    with open(rules_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for entry in data:
        raw = entry.get("allowed", []) or []
        if isinstance(raw, str):
            raw = [raw]
        allowed.extend(raw)

    if extra_path and Path(extra_path).exists():
        for line in Path(extra_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                allowed.append(line)

    return sorted(set(allowed))

def split_parts(text: str, size: int, overlap: int) -> list:
    parts = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        parts.append((start, text[start:end]))
        if end == len(text):
            break
        start += size - overlap
    return parts

def build_prompt(part: str, allowed_str: str) -> str:
    return (
        "Ты проверяешь технический текст на русском языке на ошибки в сокращениях.\n\n"
        "Вот некоторые уже правильные сокращения: " + allowed_str + " + , но это не все возможные допустимые сокращения\n\n"
        "Твоя задача: найти сокращения слов у которых:\n"
        "- пропущена точка (например разраб вместо разраб.)\n"
        "- неверный регистр (например гип вместо ГИП)\n"
        "- опечатка (например ГИБ вместо ГИП)\n"
        "- неверная форма сокращения (например кол-во вместо колич. или раз. вместо разраб.)\n\n"
        "Строго НЕ включай в ответ:\n"
        "- числа и единицы измерения (например м, кг, мм, 1.5, W10, F1150, КС-3)\n"
        "- названия стандартов (например ГОСТ, СП, ФЗ, СНиП, ТУ с любыми номерами)\n"
        "- названия компаний и организаций (например Газпромнефть, Росатом)\n"
        "- географические названия (например Балтийская, Московская)\n"
        "- инициалы (например А.С., И.И., В.Г.)\n"
        "- слова у которых fragment идентичен с suggestion\n\n"
        "Отвечай ТОЛЬКО валидным JSON-массивом, без markdown, без пояснений:\n"
        '[{"fragment": "ошибочное сокращение", "suggestion": "правильный вариант"}]\n'
        "Если ошибок нет — []\n\n"
        "Текст:\n" + part
    )

def call_llm(client, model: str, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "Ты полезный ассистент. Отвечай только JSON без пояснений."},
        {"role": "user", "content": prompt},
    ]
    response = client.chat_completion(
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

def parse_llm_response(raw: str, debug: bool) -> list:
    if debug:
        print(f"\n--- RAW LLM RESPONSE ---\n{raw}\n--- END ---\n")
    cleaned = raw.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                cleaned = part
                break

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except Exception as e:
        if debug:
            print(f"JSON parse error: {e}")
    return []


def deduplicate(findings: list) -> list:
    seen = set()
    out = []
    for f in findings:
        key = f.get("fragment", "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(f)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Проверка сокращений через LLM"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Путь к входному файлу")
    parser.add_argument("--rules", "-r", required=True,
                        help="Путь к rules.yaml")
    parser.add_argument("--extra", "-e", default=None,
                        help="Путь к файлу с дополнительными допустимыми сокращениями")
    parser.add_argument("--output", "-o", default=None,
                        help="Путь к выходному JSON (по умолчанию stdout)")
    parser.add_argument("--part-size", type=int, default=PART_SIZE,
                        help=f"Размер куска в символах (по умолчанию {PART_SIZE})")
    parser.add_argument("--debug", action="store_true",
                        help="Сырые ответы LLM")
    args = parser.parse_args()

    import os
    from huggingface_hub import InferenceClient

    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise ValueError("HF_TOKEN не задан")
    model = "deepseek-ai/DeepSeek-V3-0324"
    client = InferenceClient(token=api_token, model=model)

    allowed_list = load_allowed(args.rules, args.extra)
    allowed_str = ", ".join(allowed_list)

    text = Path(args.input).read_text(encoding="utf-8")
    parts = split_parts(text, args.part_size, PART_OVERLAP)

    all_findings = []
    for i, (offset, part) in enumerate(parts, start=1):
        prompt = build_prompt(part, allowed_str)
        try:
            raw = call_llm(client, model, prompt)
        except Exception as e:
            print(f"Ошибка запроса: {e}")
            continue
        findings = parse_llm_response(raw, args.debug)
        for f in findings:
            f["part_offset"] = offset
        all_findings.extend(findings)

    all_findings = deduplicate(all_findings)
    output = {
        "mode": "llm",
        "input": args.input,
        "parts_total": len(parts),
        "errors": all_findings,
    }

    serialized = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
        print(f"Сохранено: {args.output}")
    else:
        print(serialized)

if __name__ == "__main__":
    main()


