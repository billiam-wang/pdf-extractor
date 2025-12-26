import base64
import mimetypes
from pathlib import Path


def encode_file_base64(file_path: str) -> str:
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def img_path_to_data_url(path: str) -> str:
    p = Path(path)
    mime, _ = mimetypes.guess_type(p.name)
    if mime is None:
        mime = "image/png"

    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"
