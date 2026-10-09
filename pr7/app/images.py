"""Підготовка зображення до надсилання моделі."""
import os
from dataclasses import dataclass
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError
load_dotenv()
UPLOAD_MAX_BYTES=int(float(os.getenv("UPLOAD_MAX_MB","10"))*1024*1024)
IMAGE_MAX_SIDE=int(os.getenv("IMAGE_MAX_SIDE","1600"))
MIN_SIDE=int(os.getenv("IMAGE_MIN_SIDE","400"))
class ImageError(Exception): pass
@dataclass
class PreparedImage:
    data: bytes
    mime: str
    original: dict
    sent: dict
def prepare(content: bytes)->PreparedImage:
    if not content: raise ImageError("Файл порожній.")
    if len(content)>UPLOAD_MAX_BYTES: raise ImageError(f"Файл завеликий: {len(content)/1024/1024:.1f} MB; максимум {UPLOAD_MAX_BYTES/1024/1024:.0f} MB.")
    try:
        with Image.open(BytesIO(content)) as im:
            im.verify()
        im=Image.open(BytesIO(content))
        im=ImageOps.exif_transpose(im)
        original={"width":im.width,"height":im.height,"bytes":len(content)}
        if min(im.width,im.height)<MIN_SIDE:
            raise ImageError(f"Зображення замале ({im.width}×{im.height}). Перезніміть документ у вищій роздільності.")
        if max(im.width,im.height)>IMAGE_MAX_SIDE:
            scale=IMAGE_MAX_SIDE/max(im.width,im.height)
            im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
        if im.mode not in ("RGB","L"): im=im.convert("RGB")
        elif im.mode=="L": im=im.convert("RGB")
        out=BytesIO(); im.save(out,format="PNG",optimize=True)
        data=out.getvalue()
        return PreparedImage(data,"image/png",original,{"width":im.width,"height":im.height,"bytes":len(data)})
    except ImageError: raise
    except (UnidentifiedImageError,OSError,ValueError) as e:
        raise ImageError("Файл не є коректним зображенням.") from e
