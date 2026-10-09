"""ПР7: відтворюваний прогін оцінювання зразків.
Потрібен LLM_API_KEY у .env. Скрипт зберігає raw/model result для кожного файла.
"""
import json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app.extraction import process
ROOT=Path(__file__).resolve().parent; SAMPLES=ROOT/"samples"; EXPECTED=json.loads((SAMPLES/"expected.json").read_text(encoding="utf-8"))
OWN_EXPECTED=ROOT/"samples"/"own_expected.json"
if OWN_EXPECTED.exists(): EXPECTED.update(json.loads(OWN_EXPECTED.read_text(encoding="utf-8")))
def norm_money(v):
    if v is None:return None
    return f"{float(str(v).replace(' ','').replace(',','.')):.2f}"
def norm(v,path=""):
    if isinstance(v,dict):return {k:norm(x,f"{path}.{k}".strip(".")) for k,x in v.items()}
    if isinstance(v,list):return [norm(x,f"{path}.{i}") for i,x in enumerate(v)]
    if v is None:return None
    if path.endswith(("price","amount","total_without_vat","vat","total")):return norm_money(v)
    if path.endswith("iban"):return str(v).replace(" ","").upper()
    return v
def compare(exp,got):
    rows={"correct":0,"error":0,"missing":0,"hallucination":0}
    def rec(e,g):
        if isinstance(e,dict):
            for k,v in e.items(): rec(v, g.get(k) if isinstance(g,dict) else None)
        elif isinstance(e,list):
            for i,v in enumerate(e): rec(v,g[i] if isinstance(g,list) and i<len(g) else None)
        else:
            if e is None: rows["hallucination" if g is not None else "correct"]+=1
            elif g is None: rows["missing"]+=1
            elif norm(e)==norm(g): rows["correct"]+=1
            else: rows["error"]+=1
    rec(exp,got); return rows
def main():
    out=ROOT/"eval"/"results.json"; out.parent.mkdir(exist_ok=True)
    results=[]; totals={}
    for rel,meta in EXPECTED.items():
        path=SAMPLES/rel.replace("samples/","")
        if not path.exists(): continue
        t=time.perf_counter()
        try:
            r=process(path.read_bytes())
            got=r.document or {}
            cmp=compare(meta["fields"],got)
            caught={(i.field.split(".")[0],i.rule) for i in r.issues}
            results.append({"file":rel,"decision":r.decision,"expected_decision":meta["expected_decision"],"decision_match":r.decision==meta["expected_decision"],"issues":[i.__dict__ for i in r.issues],"compare":cmp,"image":r.image,"elapsed":r.elapsed,"usage":r.usage,"model":r.model})
        except Exception as e:
            results.append({"file":rel,"error":str(e)})
    for r in results:
        v=r.get("compare")
        if v:
            for k,n in v.items(): totals[k]=totals.get(k,0)+n
    out.write_text(json.dumps({"results":results,"totals":totals},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Збережено {out}; файлів: {len(results)}; totals={totals}")
if __name__=="__main__":main()
