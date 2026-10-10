"""Перевірка логіки workflow БЕЗ мережі й ключа: модель підмінено простим
правилом-імітацією (`FakeModel`). Це перевіряє код — переходи, пам'ять,
«рівно один раз», обмеження, — а не якість справжньої моделі. Результати
порівняння з реальною моделлю дає `eval/run_eval.py`.

Запуск із папки pr9:  python -m eval.offline_check
"""

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
RUNS = tempfile.mkdtemp(prefix="runs-")
os.environ["RUNS_DIR"] = RUNS

from app import llm, store, workflow  # noqa: E402
from shop import service  # noqa: E402

USAGE = {"prompt_tokens": 300, "completion_tokens": 60, "total_tokens": 360}


class FakeModel:
    """Імітація розбору й відповіді. Розбір — за словами, відповідь — із фактів."""

    calls = 0
    fail_next_reply = False

    @classmethod
    def complete(cls, messages, schema=None, *, name="x", max_tokens=None, timeout=None):
        cls.calls += 1
        if name == "request":
            text = messages[1]["content"].split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
            return {"text": json.dumps(cls.parse(text), ensure_ascii=False), "model": "fake",
                    "usage": dict(USAGE), "elapsed": 0.01}
        if name == "item_choice":
            return {"text": json.dumps({"sku": "SIR-R10"}), "model": "fake",
                    "usage": dict(USAGE), "elapsed": 0.01}
        facts = json.loads(messages[1]["content"].split("Факти:\n", 1)[1])
        if cls.fail_next_reply:
            cls.fail_next_reply = False
            return {"text": f"Все гаразд, код 99999, {facts.get('order_id')}", "model": "fake",
                    "usage": dict(USAGE), "elapsed": 0.01}
        return {"text": workflow._template(facts), "model": "fake", "usage": dict(USAGE),
                "elapsed": 0.01}

    @staticmethod
    def parse(t):
        low = t.lower()
        m = re.search(r"\b(\d{5})\b", t)
        d = dict(intent="other", order_id=m.group(1) if m else None, item_mention=None,
                 quantity=None, reason_category="unknown", reason_quote=None)
        if "скасуйте замовлення 10463 і" in low:
            d["intent"] = "multiple"
        elif low.startswith("дякую"):
            pass
        elif "я менеджер" in low:
            pass
        elif "скасуйте" in low:
            d["intent"] = "cancel"
        elif "де моє" in low or "лежить у відділенні" in low:
            d["intent"] = "status"
            d["item_mention"] = "годинником" if "годин" in low else "пилосос"
        elif "повернути" in low or "повернення" in low or "поверніть" in low:
            d["intent"] = "return"
            for word, sku in (("пилосос", "пилосос"), ("фільтри", "фільтри"), ("навушник", "навушники"),
                              ("роутер", "роутер"), ("годинник", "годинник")):
                if word in low:
                    d["item_mention"] = sku
                    break
            for q, cat in (("не підійшов", "not_suitable"), ("не підійшли", "not_suitable"),
                           ("не підійшло", "not_suitable"), ("не знадобились", "not_suitable"),
                           ("не заряджається", "defect"), ("перестав заряджатися", "defect"), ("з причиною defect", "defect")):
                if q in low:
                    d["reason_category"], d["reason_quote"] = cat, q
                    break
            if "обидві" in low:
                d["quantity"] = 2
        return d


llm.complete = FakeModel.complete
workflow.llm.complete = FakeModel.complete

REQUESTS = json.loads((Path(__file__).parent / "requests.example.json").read_text("utf-8"))
ok_count = fail_count = 0


def check(cond, what):
    global ok_count, fail_count
    if cond:
        ok_count += 1
        print(f"  [ OK ] {what}")
    else:
        fail_count += 1
        print(f"  [FAIL] {what}")


def by_id(i):
    return next(r for r in REQUESTS if r["id"] == i)


def send(rid):
    r = by_id(rid)
    return workflow.start(r["text"], r["customer"])


print("1. Набір звернень (workflow, підтверджуємо все, що чекає)")
service.reset()
for r in REQUESTS:
    if r.get("after"):
        continue
    run = send(r["id"])
    exp = r["expect_action"]
    waiting = run.status == "awaiting_approval"
    check(waiting == bool(exp), f"{r['id']} {r['kind']}: статус {run.status}"
          + (f" → {run.pending['tool']}" if waiting else ""))
    if waiting:
        check(run.pending["tool"] == exp["tool"] and
              all(run.pending["arguments"].get(k) == v for k, v in exp.items() if k != "tool"),
              f"{r['id']} аргументи дії збігаються з очікуваними")
        check(not service.list_returns() or r["id"] not in ("r01",), f"{r['id']} до підтвердження заявок немає")
    check(run.status in workflow.TERMINAL | {"awaiting_approval"}, f"{r['id']} у допустимому стані")
    check(run.reply is not None or waiting, f"{r['id']} є відповідь клієнтові або пауза")
    blob = json.dumps(run.__dict__, default=lambda o: o.__dict__, ensure_ascii=False)
    check("Клієнт двічі скаржився" not in blob and "ZIRKY90" not in blob and "8812" not in blob,
          f"{r['id']} службових даних у стані немає")

print("2. Підтвердження r01 і повтор r16")
run1 = next(r for r in store.list_runs("awaiting_approval") if "робот-пилосос" in r.text and "завеликий" in r.text)
done = workflow.resume(run1.run_id, True, "ок")
check(done.status == "done" and "R-5001" in done.reply, f"r01 виконано: {done.reply}")
check(len(service.list_returns()) == 1, "заявка одна")
try:
    workflow.resume(run1.run_id, True, "ще раз")
    check(False, "подвійне підтвердження має дати Conflict")
except workflow.Conflict as e:
    check(True, f"подвійне підтвердження: Conflict ({e})")
check(len(service.list_returns()) == 1, "після подвійного підтвердження заявка все ще одна")
r16 = send("r16")
check(r16.status == "done" and len(service.list_returns()) == 1, f"r16: нової заявки немає ({r16.reply})")

print("3. Відхилення")
r05 = next(r for r in store.list_runs("awaiting_approval") if r.text.startswith("Скасуйте, будь ласка"))
dec = workflow.resume(r05.run_id, False, "Замовлення вже зібране")
check(dec.status == "declined" and "не підтвердив" in dec.reply.lower() or "не підтвердив" in (dec.reply or ""),
      f"відхилено, клієнт отримав: {dec.reply}")
check(service.get_order("10463")["status"] == "preparing", "замовлення 10463 не скасовано")
try:
    workflow.resume(r05.run_id, True, "")
    check(False, "підтвердження відхиленого має дати Conflict")
except workflow.Conflict:
    check(True, "підтвердження відхиленого звернення відхилено")

print("4. Перезапуск: звернення чекає, процес «перезапущено»")
service.reset()
w = send("r01")
rid = w.run_id
import importlib
importlib.reload(store)
importlib.reload(workflow)
workflow.llm.complete = FakeModel.complete
check(store.load(rid).status == "awaiting_approval", "після перезапуску звернення в черзі")
check(workflow.recover_all() == {"executing": 0, "interrupted": 0}, "recover_all не чіпає паузу")
res = workflow.resume(rid, True, "")
check(res.status == "done" and len(service.list_returns()) == 1, "підтвердження пройшло, заявка одна")

print("5. Збій між «створено» і «записано»")
service.reset()
w = send("r01")
run = store.load(w.run_id)
# Імітація: намір записано, заявку створено, а результат у стан не потрапив.
run.status, run.decision = "executing", {"approved": True, "comment": "", "by": "operator", "at": "x"}
run.data["execution"] = {"state": "intent", "tool": "create_return", "key": run.pending["idempotency_key"]}
store.save(run)
service.create_return(**run.pending["arguments"])
check(len(service.list_returns()) == 1, "заявку створено до «збою»")
rep = workflow.recover_all()
after = store.load(run.run_id)
check(rep["executing"] == 1 and after.status == "done" and len(service.list_returns()) == 1,
      f"після відновлення друга заявка не створена ({after.reply})")
# Те саме, але заявки не було
service.reset()
w = send("r01")
run = store.load(w.run_id)
run.status, run.decision = "executing", {"approved": True, "comment": "", "by": "operator", "at": "x"}
run.data["execution"] = {"state": "intent"}
store.save(run)
workflow.recover_all()
check(store.load(run.run_id).status == "done" and len(service.list_returns()) == 1,
      "якщо заявки не було, відновлення створює рівно одну")

print("6. Умови змінилися за час очікування")
service.reset()
w = send("r05")
service.cancel_order("10463")  # хтось скасував інакше
res = workflow.resume(w.run_id, True, "")
check(res.status == "refused" and service.get_order("10463")["status"] == "cancelled",
      f"перевірено ще раз: {res.reply}")
service.reset()
a = send("r01")
service.create_return("10412", "SIR-R10", "not_suitable", 1, "іншим шляхом")
res = workflow.resume(a.run_id, True, "")
check(res.status == "done" and len(service.list_returns()) == 1 and "R-5001" in res.reply,
      f"дублікат виявлено при повторній перевірці: {res.reply}")

print("7. Обмеження")
service.reset()
for env, val, reason in (("MAX_STEPS", 3, "max_steps"), ("TOKEN_BUDGET", 400, "token_budget"),
                         ("TIMEOUT", 0.0, "timeout")):
    old = getattr(workflow, env)
    setattr(workflow, env, val)
    run = send("r01")
    setattr(workflow, env, old)
    check(run.status == "stopped" and run.stopped == reason and not service.list_returns(),
          f"{env}={val}: {run.status}/{run.stopped}, змін немає; кроків {len(run.steps)}")

print("8. Збій відповіді моделі: повтор, потім шаблон; дію не скасовано")
service.reset()
w = send("r05")
FakeModel.fail_next_reply = True
res = workflow.resume(w.run_id, True, "")
names = [s.name + ":" + s.status for s in res.steps]
check(res.status == "done" and service.get_order("10463")["status"] == "cancelled" and "10463" in res.reply,
      f"відповідь повторена після відмови перевірки: {names[-4:]}")

print("9. Збій моделі на розборі та недоступний сервіс")
def boom(*a, **k):
    raise llm.LLMError("модель недоступна", "unavailable", True)
workflow.llm.complete = boom
run = send("r01")
check(run.status == "failed" and run.reply and "помилка" in run.reply.lower(), f"LLMError → failed: {run.stopped}")
workflow.llm.complete = FakeModel.complete
os.environ["SHOP_FAILURE_RATE"] = "1"
run = send("r01")
check(run.status == "failed" and "недоступний" in run.reply, f"сервіс недоступний → «{run.reply}»")
os.environ["SHOP_FAILURE_RATE"] = "0"

print("10. Чуже замовлення (r09) і вказівка в тексті (r15)")
service.reset()
r9 = send("r09")
check(r9.status == "refused" and "Мельник" not in json.dumps(r9.data, ensure_ascii=False)
      and "10466" in r9.reply and "Оріон" not in r9.reply, f"r09: {r9.reply}")
r15 = send("r15")
check(r15.status == "awaiting_approval" and not service.list_returns()
      and r15.pending["grounds"]["notes"], "r15: чекає підтвердження, заявок немає, оператор бачить попередження")

print("11. Інструменти кроку")
res = tools_call = workflow.tools.call("create_return", json.dumps(
    {"order_id": "10412", "sku": "SIR-R10", "reason": "defect"}), workflow.tools.Context("C-1003"),
    allowed=workflow.STEP_TOOLS["get_order"])
check(res.status == "rejected" and res.code == "not_allowed" and not service.list_returns(),
      "create_return недоступний кроку get_order")
check(all(not (workflow.tools.MUTATING & s) for k, s in workflow.STEP_TOOLS.items() if k != "execute"),
      "операції зі змінами дозволені лише кроку execute")

print(f"\nПідсумок: {ok_count} ок, {fail_count} провалів")
shutil.rmtree(RUNS, ignore_errors=True)
sys.exit(1 if fail_count else 0)
