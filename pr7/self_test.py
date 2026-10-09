"""Швидкий self-test без виклику моделі."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app.rules import check
ROOT=Path(__file__).resolve().parent
exp=json.loads((ROOT/"samples/expected.json").read_text(encoding="utf-8"))
checks={"clean/rahunok-02.png":[],"clean/rahunok-05.png":["arithmetic_line"],"clean/rahunok-06.png":["iban_checksum","iban_registry"],"clean/rahunok-07.png":["buyer"],"clean/rahunok-08.png":["iban_registry"],"clean/nakladna-09.png":["document_type"]}
bad=[]
for rel,want in checks.items():
    got=sorted(set(i.rule for i in check(exp[rel]["fields"])))
    if got!=sorted(want): bad.append((rel,want,got))
print("rule self-test:", "PASS" if not bad else "FAIL")
if bad:
    for x in bad: print(x)
    raise SystemExit(1)
