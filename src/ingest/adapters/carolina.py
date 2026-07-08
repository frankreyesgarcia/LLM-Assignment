"""CarolinaAdapter: bespoke loader for carolina-c4ai/corpus-carolina.

Its HF loading script is no longer supported by `datasets>=4` (see
registry.BLOCKED_SOURCES). The repo's real structure (confirmed by
inspecting its file list and one downloaded shard) is gzip-compressed TEI
XML: `corpus/{taxonomy}/**/*.xml.gz`, each file a `<teiCorpus>` containing
many `<TEI>` elements -- one per document, with a `<teiHeader>` (metadata)
and a `<text><body><p>...</p>...</body></text>` (the document text).

This streams each shard (download + gunzip + incremental XML parse)
without ever holding a whole file in memory, per the plan's "streaming
first" principle.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from typing import Iterator

import requests

from src.ingest.base import Document, SourceAdapter

_REPO_ID = "carolina-c4ai/corpus-carolina"
_TEI_NS = "{http://www.tei-c.org/ns/1.0}"

TAXONOMY_FOLDERS = {
    "dat": "datasets_and_other_corpora",
    "jud": "judicial_branch",
    "leg": "legislative_branch",
    "pub": "public_domain_works",
    "soc": "social_media",
    "uni": "university_domains",
    "wik": "wikis",
}


class CarolinaAdapter(SourceAdapter):
    def __init__(
        self,
        name: str = "corpus-carolina",
        taxonomies: list[str] | None = None,
        limit: int | None = None,
    ) -> None:
        self.name = name
        self.taxonomies = taxonomies or list(TAXONOMY_FOLDERS)
        self.limit = limit

    def _list_shard_files(self) -> list[str]:
        resp = requests.get(f"https://huggingface.co/api/datasets/{_REPO_ID}", timeout=30)
        resp.raise_for_status()
        siblings = [s["rfilename"] for s in resp.json().get("siblings", [])]
        folders = {TAXONOMY_FOLDERS[t] for t in self.taxonomies}
        return sorted(
            f for f in siblings if f.startswith("corpus/") and f.endswith(".xml.gz") and f.split("/")[1] in folders
        )

    def _iter_docs_in_shard(self, path: str) -> Iterator[Document]:
        url = f"https://huggingface.co/datasets/{_REPO_ID}/resolve/main/{path}"
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with gzip.GzipFile(fileobj=resp.raw) as gz:
                for _, elem in ET.iterparse(gz, events=("end",)):
                    if elem.tag != f"{_TEI_NS}TEI":
                        continue
                    body = elem.find(f".//{_TEI_NS}body")
                    if body is None:
                        elem.clear()
                        continue
                    paragraphs = [p.text.strip() for p in body.iter(f"{_TEI_NS}p") if p.text and p.text.strip()]
                    text = "\n".join(paragraphs)
                    if not text:
                        elem.clear()
                        continue
                    header = elem.find(f"{_TEI_NS}teiHeader")
                    meta_xml = ET.tostring(header, encoding="unicode") if header is not None else ""
                    yield Document(
                        text=text,
                        language="pt",
                        source=self.name,
                        source_id="",
                        url=None,
                        metadata={"taxonomy": path.split("/")[1], "shard": path, "meta_xml": meta_xml},
                    )
                    elem.clear()

    def iter_documents(self) -> Iterator[Document]:
        count = 0
        for path in self._list_shard_files():
            for doc in self._iter_docs_in_shard(path):
                if self.limit is not None and count >= self.limit:
                    return
                yield doc
                count += 1
