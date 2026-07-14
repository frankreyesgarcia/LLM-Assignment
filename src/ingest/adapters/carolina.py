"""CarolinaAdapter: bespoke loader for carolina-c4ai/corpus-carolina.

Its HF loading script is no longer supported by `datasets>=4` (see
registry.BLOCKED_SOURCES). The repo's real structure (confirmed by
inspecting its file list and one downloaded shard) is gzip-compressed TEI
XML: `corpus/{taxonomy}/**/*.xml.gz`, each file a `<teiCorpus>` containing
many `<TEI>` elements -- one per document, with a `<teiHeader>` (metadata)
and a `<text><body><p>...</p>...</body></text>` (the document text).

This gunzips + incrementally XML-parses each shard without ever holding a
whole file in memory, per the plan's "streaming first" principle. Shard
bytes themselves go through huggingface_hub.hf_hub_download (not a raw
`requests` stream): a bare `resolve/main/<path>` GET 302s to a presigned
`cas-bridge.xethub.hf.co` URL, and hitting that directly with plain
`requests` was observed to 403/500 intermittently -- hf_hub_download
already knows how to follow/retry that indirection correctly (it's the
same path scripts/download_sources.py's snapshot_download uses).
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from huggingface_hub import HfApi, hf_hub_download

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
        local_dir: Path | None = None,
    ) -> None:
        self.name = name
        self.taxonomies = taxonomies or list(TAXONOMY_FOLDERS)
        self.limit = limit
        # If set (scripts/download_sources.py already pulled the matching
        # corpus/{taxonomy}/**/*.xml.gz shards to local disk), read those
        # directly instead of streaming each shard over HTTP one at a time
        # -- see GenericTextAdapter.local_files for the same local-vs-
        # streamed rationale.
        self.local_dir = local_dir

    def _list_shard_files(self) -> list[str]:
        folders = {TAXONOMY_FOLDERS[t] for t in self.taxonomies}
        if self.local_dir is not None:
            found: list[str] = []
            for folder in folders:
                found.extend(
                    str(p.relative_to(self.local_dir))
                    for p in (self.local_dir / "corpus" / folder).rglob("*.xml.gz")
                )
            return sorted(found)
        # Use huggingface_hub (not a raw `requests.get`) for the same reason
        # _iter_docs_in_shard uses hf_hub_download below -- it has its own
        # retry/auth handling rather than a bare unauthenticated HTTP call.
        siblings = HfApi().list_repo_files(repo_id=_REPO_ID, repo_type="dataset")
        return sorted(
            f for f in siblings if f.startswith("corpus/") and f.endswith(".xml.gz") and f.split("/")[1] in folders
        )

    def _iter_docs_in_shard(self, path: str) -> Iterator[Document]:
        if self.local_dir is not None:
            local_path = self.local_dir / path
        else:
            local_path = Path(hf_hub_download(repo_id=_REPO_ID, repo_type="dataset", filename=path))
        with gzip.open(local_path, "rb") as gz:
            yield from self._parse_shard(gz, path)

    def _parse_shard(self, gz, path: str) -> Iterator[Document]:
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
