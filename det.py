import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import yaml

# Левенштейн
def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + (ca != cb))
        prev = cur
    return prev[-1]

# Нормализация
_WS_RE = re.compile(r"\s+")

def normalize(s):
    # Нижний регистр + удаление лишних пробелов
    return _WS_RE.sub(" ", s.strip().lower())

# Правила rules.yaml
@dataclass
class Rules:
    full_forms: set = field(default_factory=set)
    allowed_forms: set = field(default_factory=set)
    full_to_allowed: dict = field(default_factory=dict)
    allowed_to_full: dict = field(default_factory=dict)
    full_original: dict = field(default_factory=dict)
    allowed_original: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        r = cls()
        for entry in data:
            full_raw = entry["full"]
            allowed_raw = entry.get("allowed", []) or []
            if isinstance(allowed_raw, str):
                allowed_raw = [allowed_raw]
            full_norm = normalize(full_raw)
            r.full_forms.add(full_norm)
            r.full_original[full_norm] = full_raw
            for a_raw in allowed_raw:
                a_norm = normalize(a_raw)
                r.allowed_forms.add(a_norm)
                r.allowed_original[a_norm] = a_raw
                r.allowed_to_full[a_norm] = full_norm
                r.full_to_allowed.setdefault(full_norm, []).append(a_norm)
        return r

# Вердикты и Finding
class Verdict(str, Enum):
    OK = "OK"
    ERROR = "ОШИБКА"
    SKIP = "ПРОПУСК"
    UNKNOWN = "НЕРАСПОЗНАНО"

class Source(str, Enum):
    DETERMINISTIC = "det"

@dataclass
class Finding:
    fragment: str
    position: int
    expected: str | None
    verdict: Verdict
    source: Source
    note: str = ""

    def to_dict(self):
        return {
            "fragment": self.fragment,
            "position": self.position,
            "expected": self.expected,
            "verdict": self.verdict.value,
            "source": self.source.value,
            "note": self.note,
        }

# Очистка Markdown
def strip_markdown(text):
    def _link_repl(m):
        original_len = len(m.group(0))
        inner = m.group(1)
        return inner + " " * (original_len - len(inner))

    text = re.sub(r"\[(.+?)\]\((.+?)\)", _link_repl, text)
    text = re.sub(r"!\[.*?\]\(.*?\)", lambda m: " " * len(m.group(0)), text)

    for pat in [
        re.compile(r"^[#]{1,6}\s", re.MULTILINE),
        re.compile(r"`{1,3}"),
        re.compile(r"\*{1,3}|_{1,3}"),
        re.compile(r"^\s*[-*+]\s", re.MULTILINE),
        re.compile(r"\|"),
    ]:
        text = pat.sub(lambda m: " " * len(m.group(0)), text)
    return text

@dataclass(frozen=True)
class Candidate:
    text: str
    start: int
    has_dot: bool

_LETTER = r"[A-Za-zА-Яа-яЁё]"
_DOTTED_TOKEN = rf"{_LETTER}{{1,8}}\."
_DOTTED_GROUP_RE = re.compile(
    rf"\b{_DOTTED_TOKEN}(?:\s[а-яёa-z]{{1,8}}\.|-{_LETTER}{{1,8}}\.)?"
)
_ACRONYM_RE = re.compile(r"\b[А-ЯЁA-Z]{2,3}\b")
_SHORT_WORD_RE = re.compile(rf"\b{_LETTER}{{2,12}}\b(?!\.)")
_HYPHEN_RE = re.compile(r"\b[А-Яа-яЁёA-Za-z]{2,8}-[А-Яа-яЁёA-Za-z]{2,8}\b")

# Список слов, которые всегда пропускаются внутри составных сокращений
STOP_WORDS = {"во", "из", "за", "до", "по", "на", "под", "над", "без", "со", "что"}

# Поиск сокращений
def find_candidates(text):
    occupied = []
    cands = []

    def overlaps(a, b):
        return any(not (b <= s or a >= e) for s, e in occupied)

    for m in _DOTTED_GROUP_RE.finditer(text):
        s, e = m.span()
        cands.append(Candidate(text=m.group(0), start=s, has_dot=True))
        occupied.append((s, e))

    for m in _ACRONYM_RE.finditer(text):
        s, e = m.span()
        if overlaps(s, e):
            continue
        cands.append(Candidate(text=m.group(0), start=s, has_dot=False))
        occupied.append((s, e))

    for m in _HYPHEN_RE.finditer(text):
        s, e = m.span()
        if overlaps(s, e):
            continue
        cands.append(Candidate(text=m.group(0), start=s, has_dot=False))
        occupied.append((s, e))

    for m in _SHORT_WORD_RE.finditer(text):
        s, e = m.span()
        if overlaps(s, e):
            continue
        cands.append(Candidate(text=m.group(0), start=s, has_dot=False))
        occupied.append((s, e))

    cands.sort(key=lambda c: c.start)
    return cands

def context_around(text, start, length, window=100):
    left = max(0, start - window)
    right = min(len(text), start + length + window)
    return text[left:right]

# Конфиг детерминированного проверщика
@dataclass
class DeterministicConfig:
    threshold: int = 2
    short_word_min_len: int = 3

class DeterministicChecker:
    def __init__(self, rules, config=None):
        self.rules = rules
        self.config = config or DeterministicConfig()

    def check_text(self, text):
        findings = []
        for cand in find_candidates(text):
            findings.append(self.classify(cand))
        return findings

    def classify(self, cand, inside_compound=False):
        norm = normalize(cand.text)

        if norm in self.rules.allowed_forms:
            expected = self.rules.allowed_original.get(norm)
            if expected is not None:
                expected_without_dot = expected.rstrip('.')
                if expected_without_dot.isupper() and len(expected_without_dot) > 1:
                    if cand.text != expected:
                        full = self.rules.allowed_to_full.get(norm)
                        full_word = self.rules.full_original.get(full, "") if full else ""
                        note = f"Неверный регистр. Ожидается: {expected}"
                        if full_word:
                            note += f" (полное слово: {full_word})"
                        return Finding(
                            fragment=cand.text,
                            position=cand.start,
                            expected=expected,
                            verdict=Verdict.ERROR,
                            source=Source.DETERMINISTIC,
                            note=note,
                        )
            return Finding(
                fragment=cand.text,
                position=cand.start,
                expected=expected,
                verdict=Verdict.OK,
                source=Source.DETERMINISTIC,
            )

        if norm in self.rules.full_forms:
            return Finding(
                fragment=cand.text,
                position=cand.start,
                expected=None,
                verdict=Verdict.SKIP,
                source=Source.DETERMINISTIC,
            )

        if "-" in norm and not norm.endswith("-"):
            return self.classify_compound(cand)

        if " " in norm:
            return self.classify_spaced(cand)

        return self.classify_fuzzy(cand, norm, inside_compound)

    # Составное сокращение через дефис
    def classify_compound(self, cand):
        parts = [p for p in cand.text.split("-") if p]
        sub_findings = []
        offset = cand.start
        for part in parts:
            sub_cand = Candidate(text=part, start=offset, has_dot="." in part)
            sub_findings.append(self.classify(sub_cand, inside_compound=True))
            offset += len(part) + 1

        if all(f.verdict in (Verdict.OK, Verdict.SKIP) for f in sub_findings):
            return Finding(
                fragment=cand.text,
                position=cand.start,
                expected=None,
                verdict=Verdict.OK,
                source=Source.DETERMINISTIC,
            )

        error_parts = [f for f in sub_findings if f.verdict == Verdict.ERROR]
        unknown_parts = [f for f in sub_findings if f.verdict == Verdict.UNKNOWN]
        if len(error_parts) == 1 and not unknown_parts:
            other_parts = [f for f in sub_findings if f != error_parts[0]]
            if all(f.verdict == Verdict.SKIP for f in other_parts):
                expected = error_parts[0].expected
                details = f"«{error_parts[0].fragment}» → «{error_parts[0].expected}»"
                return Finding(
                    fragment=cand.text,
                    position=cand.start,
                    expected=expected,
                    verdict=Verdict.ERROR,
                    source=Source.DETERMINISTIC,
                    note=details,
                )
        expected_parts = []
        details = []
        for f in sub_findings:
            if f.verdict == Verdict.OK:
                expected_parts.append(f.expected or f.fragment)
            elif f.verdict == Verdict.ERROR:
                expected_parts.append(f.expected or f.fragment)
                details.append(
                    f"«{f.fragment}» → «{f.expected}»" if f.expected
                    else f"«{f.fragment}» (не распознано)"
                )
            else:
                expected_parts.append(f.fragment)
                if f.verdict == Verdict.UNKNOWN:
                    details.append(f"«{f.fragment}» не найдено в правилах")

        full_expected = "-".join(expected_parts)
        return Finding(
            fragment=cand.text,
            position=cand.start,
            expected=full_expected,
            verdict=Verdict.ERROR,
            source=Source.DETERMINISTIC,
            note="; ".join(details),
        )

    def classify_spaced(self, cand):
        parts = [p for p in cand.text.split(" ") if p]
        sub_findings = []
        offset = cand.start
        for part in parts:
            sub_cand = Candidate(text=part, start=offset, has_dot="." in part)
            sub_findings.append(self.classify(sub_cand))
            offset += len(part) + 1

        if all(f.verdict in (Verdict.OK, Verdict.SKIP) for f in sub_findings):
            return Finding(
                fragment=cand.text,
                position=cand.start,
                expected=None,
                verdict=Verdict.OK,
                source=Source.DETERMINISTIC,
            )

        bad = [f for f in sub_findings if f.verdict == Verdict.ERROR]
        if bad:
            expected_parts = []
            for f in sub_findings:
                if f.verdict == Verdict.ERROR and f.expected:
                    expected_parts.append(f.expected)
                else:
                    expected_parts.append(f.fragment)
            full_expected = " ".join(expected_parts)

            details = "; ".join(
                f"«{f.fragment}» → возможно «{f.expected}»" if f.expected
                else f"«{f.fragment}» (не распознано)"
                for f in bad
            )
            return Finding(
                fragment=cand.text,
                position=cand.start,
                expected=full_expected,
                verdict=Verdict.ERROR,
                source=Source.DETERMINISTIC,
                note=details,
            )
        return Finding(
            fragment=cand.text,
            position=cand.start,
            expected=None,
            verdict=Verdict.UNKNOWN,
            source=Source.DETERMINISTIC,
        )

    def classify_fuzzy(self, cand, norm, inside_compound=False):
        # инициалы
        if cand.has_dot and len(cand.text.rstrip(".")) == 1:
            return Finding(
                fragment=cand.text, position=cand.start,
                expected=None, verdict=Verdict.SKIP,
                source=Source.DETERMINISTIC,
            )
        # аббревиатура больше трёх букв — пропускаем
        if cand.text.isupper() and len(cand.text) > 3:
            return Finding(
                fragment=cand.text,
                position=cand.start,
                expected=None,
                verdict=Verdict.SKIP,
                source=Source.DETERMINISTIC,
            )
        # внутри составного: стоп-слова пропускаем
        if inside_compound and not cand.has_dot and not cand.text.isupper():
            if norm in STOP_WORDS:
                return Finding(
                    fragment=cand.text,
                    position=cand.start,
                    expected=None,
                    verdict=Verdict.SKIP,
                    source=Source.DETERMINISTIC,
                )
        # обычное слово без точки — пропускаем если не внутри составного
        if not cand.has_dot and not cand.text.isupper() and not inside_compound:
            return Finding(
                fragment=cand.text,
                position=cand.start,
                expected=None,
                verdict=Verdict.SKIP,
                source=Source.DETERMINISTIC,
            )

        is_short_acronym = (not cand.has_dot and cand.text.isupper() and len(norm) <= 3)
        local_threshold = 1 if is_short_acronym else self.config.threshold

        best_target, best_dist = self.nearest_with_prefix_priority(norm, self.rules.allowed_forms)

        if best_target is None:
            return self.unrecognized(cand)
        if best_dist <= local_threshold:
            expected_original = self.original_of(best_target, "allowed")
            full_word = ""
            if best_target in self.rules.allowed_to_full:
                full_norm = self.rules.allowed_to_full[best_target]
                full_word = self.rules.full_original.get(full_norm, "")
            if full_word:
                note = f"Возможно, имелось в виду: {expected_original} (полное слово: {full_word})"
            else:
                note = f"Возможно, имелось в виду: {expected_original}"
            return Finding(
                fragment=cand.text,
                position=cand.start,
                expected=expected_original,
                verdict=Verdict.ERROR,
                source=Source.DETERMINISTIC,
                note=note,
            )
        return self.unrecognized(cand)

    # Поиск ближайшего allowed
    def nearest_with_prefix_priority(self, s, pool):
        if not pool:
            return None, None
        s_no_dot = s.replace('.', '')
        reverse_prefix = []
        for t in pool:
            t_no_dot = t.replace('.', '')
            if t_no_dot.startswith(s_no_dot):
                reverse_prefix.append(t)

        if reverse_prefix:
            reverse_prefix.sort(key=lambda x: len(x.replace('.', '')), reverse=True)
            best = reverse_prefix[0]
            best_no_dot = best.replace('.', '')
            dist = len(best_no_dot) - len(s_no_dot)
            if dist <= self.config.threshold:
                return best, dist

        prefix_candidates = []
        for t in pool:
            t_no_dot = t.replace('.', '')
            if s_no_dot.startswith(t_no_dot):
                prefix_candidates.append(t)

        if prefix_candidates:
            prefix_candidates.sort(key=lambda x: len(x.replace('.', '')), reverse=True)
            best = prefix_candidates[0]
            best_no_dot = best.replace('.', '')
            dist = len(s_no_dot) - len(best_no_dot)
            if dist <= self.config.threshold:
                return best, dist

        # Левенштейн
        best_target = None
        best_dist = 10**9
        for t in pool:
            if abs(len(t) - len(s)) >= best_dist:
                continue
            d = levenshtein(s, t)
            if d < best_dist:
                best_dist = d
                best_target = t
                if best_dist == 0:
                    break
        return best_target, best_dist

    # нераспознанное сокращение
    def unrecognized(self, cand):
        if cand.has_dot or cand.text.isupper():
            return Finding(
                fragment=cand.text,
                position=cand.start,
                expected=None,
                verdict=Verdict.UNKNOWN,
                source=Source.DETERMINISTIC,
            )
        return Finding(
            fragment=cand.text,
            position=cand.start,
            expected=None,
            verdict=Verdict.SKIP,
            source=Source.DETERMINISTIC,
        )

    def original_of(self, norm, kind):
        if kind == "allowed":
            return self.rules.allowed_original.get(norm, norm)
        return self.rules.full_original.get(norm, norm)

def main():
    parser = argparse.ArgumentParser(description="Проверка сокращений детерминированным методом")
    parser.add_argument("--input", "-i", required=True, help="Путь к файлу")
    parser.add_argument("--rules", "-r", required=True, help="Путь к rules.yaml")
    parser.add_argument("--output", "-o", default=None, help="Путь к JSON-отчёту")
    args = parser.parse_args()

    raw_text = Path(args.input).read_text(encoding="utf-8")
    clean_text = strip_markdown(raw_text)
    rules = Rules.from_yaml(args.rules)
    det_cfg = DeterministicConfig(threshold=2, short_word_min_len=3)
    checker = DeterministicChecker(rules, det_cfg)
    findings = checker.check_text(clean_text)

    findings_dicts = [f.to_dict() for f in findings]
    errors = [f for f in findings_dicts if f["verdict"] == "ОШИБКА"]
    skipped = [f for f in findings_dicts if f["verdict"] == "ПРОПУСК"]
    ok_list = [f for f in findings_dicts if f["verdict"] == "OK"]
    allowed_used = sorted({f["fragment"] for f in ok_list})

    output = {
        "mode": "det",
        "input": args.input,
        "errors": errors,
        "skipped_count": len(skipped),
        "allowed_found": allowed_used,
    }
    serialized = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
        print(f"Сохранено: {args.output}")
    else:
        print(serialized)

if __name__ == "__main__":
    main()
