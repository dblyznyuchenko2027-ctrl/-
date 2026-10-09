"""Конвеєр: зображення → модель → правила → рішення."""
import time
from dataclasses import dataclass,field
from . import images,llm
from .rules import Issue,check
@dataclass
class Result:
    decision:str; reasons:list[str]=field(default_factory=list); document:dict|None=None; issues:list[Issue]=field(default_factory=list); image:dict=field(default_factory=dict); model:str|None=None; elapsed:dict=field(default_factory=dict); usage:dict|None=None
def decide(document,issues):
    if document is None:return "reject",["Не вдалося отримати структуровані дані з документа."]
    if document.get("document_type")!="рахунок":return "reject",["Документ не є рахунком на оплату."]
    if not document.get("items") and all(document.get(k) is None for k in ["number","date","total"]):return "reject",["Із документа не вдалося вилучити суттєві дані."]
    errors=[i for i in issues if i.severity=="error"]
    if errors:return "review",[f"{i.field}: {i.message}" for i in errors]
    return "auto",["Усі критичні поля присутні, схема пройдена, програмні перевірки пройдені."]
def process(content):
    t=time.perf_counter(); prep=images.prepare(content); t1=time.perf_counter()
    got=llm.extract(prep); t2=time.perf_counter(); issues=check(got["document"]); t3=time.perf_counter()
    decision,reasons=decide(got["document"],issues)
    return Result(decision,reasons,got["document"],issues,{"original":prep.original,"sent":prep.sent},got["model"],{"prepare":t1-t,"extraction":t2-t1,"checks":t3-t2},got["usage"])
