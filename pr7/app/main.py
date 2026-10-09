"""FastAPI веб-рівень."""
from dataclasses import asdict
from pathlib import Path
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from . import extraction
from .images import ImageError
from .llm import LLMError
app=FastAPI(title="Розбір рахунків — ПР7")
INDEX_PAGE=Path(__file__).parent/"templates"/"index.html"; SAMPLES_DIR=Path(__file__).resolve().parent.parent/"samples"; IMAGE_SUFFIXES={".png",".jpg",".jpeg",".webp"}
app.mount("/samples",StaticFiles(directory=SAMPLES_DIR),name="samples")
def result_to_dict(r): return {"decision":r.decision,"reasons":r.reasons,"document":r.document,"issues":[asdict(i) for i in r.issues],"image":r.image,"model":r.model,"elapsed":r.elapsed,"usage":r.usage}
@app.get("/",response_class=HTMLResponse)
def page():return INDEX_PAGE.read_text(encoding="utf-8")
@app.get("/api/samples")
def api_samples():
    groups={}
    for folder in sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir()):
        fs=sorted(f.name for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if fs:groups[folder.name]=fs
    return groups
@app.post("/api/extract")
async def api_extract(image:UploadFile=File(...)):
    content=await image.read()
    try:return result_to_dict(extraction.process(content))
    except ImageError as e:return JSONResponse(status_code=422,content={"error":"image","message":str(e)})
    except LLMError as e:return JSONResponse(status_code=502,content={"error":"model","message":str(e)})
    except Exception:return JSONResponse(status_code=500,content={"error":"internal","message":"Неочікувана помилка обробки."})
