# Synthetic evaluation fixtures

All documents are deterministic, synthetic CC0 fixtures. They contain no
production, customer, or externally sourced content. Regenerate and verify them
from this directory with:

```sh
python3 -B generator.py
python3 -B generator.py --check
```

The manifest covers native Traditional Chinese PDFs, an image-only scanned PDF,
a PPTX, a table/chart PDF, a PNG, Markdown, and plain text. It includes paired
answerable and not-found queries, same-filename document isolation, adjacent-page
decoys, multi-page sources, null pages for Markdown/text, and visual-only facts.
