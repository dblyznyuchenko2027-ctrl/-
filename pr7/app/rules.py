"""Детерміновані змістові перевірки документа."""
import json,re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
REFERENCE_DIR=Path(__file__).resolve().parent.parent/"reference"
@dataclass
class Issue:
    field:str; rule:str; message:str; severity:str="error"
def load_reference():
    return json.loads((REFERENCE_DIR/"company.json").read_text(encoding="utf-8")), json.loads((REFERENCE_DIR/"suppliers.json").read_text(encoding="utf-8"))
def _money(v):
    if v is None:return None
    try:return Decimal(str(v).replace(" ","").replace(",", "."))
    except (InvalidOperation,ValueError):return None
def _norm(s): return re.sub(r"[^0-9A-Za-zА-Яа-яІіЇїЄєҐґ]","",str(s or "")).upper().replace("ʼ","'")
def iban_valid(iban):
    x=re.sub(r"\s+","",str(iban or "")).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}",x): return False
    if x.startswith("UA") and len(x)!=29:return False
    moved=x[4:]+x[:4]
    num=''.join(str(ord(c)-55) if c.isalpha() else c for c in moved)
    return int(num)%97==1
def _code_valid(code):
    s=re.sub(r"\D","",str(code or ""))
    if len(s)==8:
        weights=[1,2,3,4,5,6,7]
        r=sum(int(d)*w for d,w in zip(s[:7],weights))%11
        if r==10:r=sum(int(d)*w for d,w in zip(s[:7],[3,4,5,6,7,8,9]))%11
        return r==int(s[7])
    if len(s)==10:
        # РНОКПП: ваги -1,5,7,9,4,6,10,5,7; контроль = (sum mod 11) mod 10.
        r=sum(int(d)*w for d,w in zip(s[:9],[-1,5,7,9,4,6,10,5,7]))%11
        return (r%10)==int(s[9])
    return False
def _date(v):
    if not v:return None
    try:return date.fromisoformat(str(v))
    except ValueError:return None
def check(document):
    issues=[]; company,suppliers=load_reference()
    def add(field,rule,msg,sev="error"): issues.append(Issue(field,rule,msg,sev))
    if document.get("document_type")!="рахунок": add("document_type","document_type","Документ не є рахунком на оплату.")
    required=["number","date","supplier.name","supplier.code","supplier.iban","buyer.name","buyer.code","items","total_without_vat","vat","total"]
    for path in required:
        cur=document
        for p in path.split("."):
            cur=cur.get(p) if isinstance(cur,dict) else None
        if cur is None or cur=="" or cur==[]: add(path,"missing_required","Обов'язкове поле відсутнє.")
    for path in ["supplier.code","buyer.code"]:
        cur=document.get(path.split(".")[0],{}).get(path.split(".")[1])
        if cur is not None and not _code_valid(cur): add(path,"code_checksum","Контрольний розряд коду не пройшов перевірку.")
    iban=document.get("supplier",{}).get("iban")
    if iban is not None:
        if not iban_valid(iban): add("supplier.iban","iban_checksum","IBAN має неправильний формат або контрольне число.")
    supplier=document.get("supplier",{}); matched=None
    for s in suppliers:
        if _norm(s["name"])==_norm(supplier.get("name")) or str(s["code"])==str(supplier.get("code")): matched=s; break
    if matched is None: add("supplier.name","supplier_registry","Постачальника немає у довіднику.")
    else:
        if str(supplier.get("iban") or "").replace(" ","").upper()!=matched["iban"].upper(): add("supplier.iban","iban_registry","IBAN не збігається з довідником постачальника.")
        if str(supplier.get("code") or "")!=matched["code"]: add("supplier.code","supplier_registry","Код постачальника не збігається з довідником.")
    buyer=document.get("buyer",{})
    if _norm(buyer.get("name"))!=_norm(company["name"]): add("buyer.name","buyer","Покупець не є ТОВ «Сузірʼя Рітейл».")
    if str(buyer.get("code") or "")!=company["code"]: add("buyer.code","buyer","Код покупця не збігається з реквізитами «Сузірʼя Рітейл».")
    today=date.today()
    d=_date(document.get("date")); vu=_date(document.get("valid_until"))
    if document.get("date") is not None and d is None:add("date","date_format","Дата має бути YYYY-MM-DD.")
    if d and d>today:add("date","date_future","Дата рахунку не може бути в майбутньому.")
    if vu and d and vu<d:add("valid_until","date_range","Строк дії не може бути раніше дати рахунку.")
    if document.get("valid_until") is not None and vu is None:add("valid_until","date_format","Строк дії має бути YYYY-MM-DD.")
    items=document.get("items") or []; sum_items=Decimal("0.00")
    for i,it in enumerate(items):
        q=_money(it.get("quantity")); p=_money(it.get("price")); a=_money(it.get("amount"))
        if q is None or p is None or a is None: continue
        expected=(q*p).quantize(Decimal("0.01"),ROUND_HALF_UP); sum_items+=a
        if expected!=a:add(f"items.{i}.amount","arithmetic_line",f"Кількість × ціна = {expected}, але в документі {a}.")
    total_wo=_money(document.get("total_without_vat")); vat=_money(document.get("vat")); total=_money(document.get("total"))
    if total_wo is not None:
        if (sum_items-total_wo).copy_abs()>Decimal("0.01"): add("total_without_vat","arithmetic_subtotal",f"Сума позицій {sum_items:.2f} не дорівнює підсумку без ПДВ {total_wo:.2f}.")
    if matched and vat is not None and total_wo is not None:
        expected_vat=(total_wo*Decimal("0.20") if matched.get("vat_payer") else Decimal("0")).quantize(Decimal("0.01"),ROUND_HALF_UP)
        if abs(vat-expected_vat)>Decimal("0.01"): add("vat","arithmetic_vat",f"ПДВ має бути {expected_vat:.2f}.")
    if total is not None and total_wo is not None and vat is not None and abs((total_wo+vat)-total)>Decimal("0.01"): add("total","arithmetic_total",f"Разом + ПДВ = {(total_wo+vat):.2f}, але до сплати {total:.2f}.")
    return issues
