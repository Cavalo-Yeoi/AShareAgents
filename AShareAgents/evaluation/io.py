"""Small, atomic, UTF-8 evaluation artifacts."""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


EVALUATION_DIR = Path(__file__).resolve().parent
ROOT = EVALUATION_DIR.parents[1]
RETRIEVAL_CONFIG = EVALUATION_DIR / 'configs' / 'retrieval_eval.yaml'
CORPUS = ROOT / 'data' / 'rag_eval_v1'


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     delete=False, suffix='.tmp') as stream:
        stream.write(text)
        temporary = Path(stream.name)
    # Windows readers may temporarily deny replacement despite a correct path.
    # Keep the previous artifact intact and retry only this bounded sharing race.
    for attempt in range(7):
        try:
            os.replace(temporary,path)
            break
        except PermissionError:
            if attempt==6:
                raise
            time.sleep(.05*2**attempt)


def write_jsonl(path, rows):
    write_text(path, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                             for row in rows))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()
