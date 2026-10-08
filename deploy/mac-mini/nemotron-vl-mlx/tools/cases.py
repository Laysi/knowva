"""Small fixed rerank cases over a directory of pages (p-NN.jpg with
p-NN.txt beside it), shaped like a retrieval service's calls: one query, up to
eight passages per request, one modality per request."""

from pathlib import Path

from PIL import Image

from nemotron_vl_mlx.processing import Passage

QUERY = "IoT4Industry D.1 文件的作者與 QA 負責人是誰？"
MAX_LENGTH = {"text": 8192, "image": 2048, "text_image": 10240}


def load_cases(pages_dir):
    pages = sorted(Path(pages_dir).glob("p-*.jpg"))
    texts = [p.with_suffix(".txt").read_text().strip() for p in pages]
    images = [Image.open(p).convert("RGB") for p in pages]
    return {
        "text-8": ("text", [Passage(text=t) for t in texts[:8]]),
        "image-4": ("image", [Passage(image=i) for i in images[:4]]),
        "text_image-4": ("text_image", [Passage(text=t, image=i) for t, i in zip(texts[4:8], images[4:8])]),
        "text_image-1": ("text_image", [Passage(text=texts[8], image=images[8])]),
    }
