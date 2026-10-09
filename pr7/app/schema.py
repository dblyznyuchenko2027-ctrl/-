"""JSON Schema та структурна валідація відповіді моделі."""
import json
from jsonschema import Draft202012Validator, FormatChecker

def output_schema()->dict:
    money={"type":["string","null"],"pattern":"^-?\\d+(\\.\\d{1,2})?$"}
    item={"type":"object","additionalProperties":False,"required":["name","unit","quantity","price","amount"],"properties":{
        "name":{"type":["string","null"]},"unit":{"type":["string","null"]},
        "quantity":{"type":["number","null"]},"price":money,"amount":money}}
    party={"type":"object","additionalProperties":False,"required":["name","code","iban"],"properties":{
        "name":{"type":["string","null"]},"code":{"type":["string","null"]},"iban":{"type":["string","null"]}}}
    buyer={"type":"object","additionalProperties":False,"required":["name","code"],"properties":{
        "name":{"type":["string","null"]},"code":{"type":["string","null"]}}}
    return {"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":False,
      "required":["document_type","number","date","valid_until","supplier","buyer","items","total_without_vat","vat","total"],
      "properties":{
        "document_type":{"type":["string","null"],"enum":["рахунок","інший документ",None]},
        "number":{"type":["string","null"]},"date":{"type":["string","null"],"format":"date"},"valid_until":{"type":["string","null"],"format":"date"},
        "supplier":party,"buyer":buyer,"items":{"type":"array","items":item},"total_without_vat":money,"vat":money,"total":money}}

def validate(raw:str)->dict:
    try: data=json.loads(raw)
    except json.JSONDecodeError as e: raise ValueError(f"Відповідь моделі не є JSON: {e.msg}.") from e
    v=Draft202012Validator(output_schema(),format_checker=FormatChecker())
    errors=sorted(v.iter_errors(data),key=lambda e:list(e.path))
    if errors:
        details=[]
        for e in errors[:8]: details.append(f"{'.'.join(map(str,e.path)) or 'root'}: {e.message}")
        raise ValueError("Відповідь не проходить схему: "+"; ".join(details))
    return data
