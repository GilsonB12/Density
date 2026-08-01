"""Leitura de arquivos e extração de texto."""

from pathlib import Path

from pypdf import PdfReader

from density.models import Document


class UnsupportedFormatError(ValueError):
    """Extensão de arquivo sem leitor registrado."""


class EmptyDocumentError(ValueError):
    """Arquivo lido, mas nenhum texto extraível."""


_TEXT_SUFFIXES = {".txt", ".md"}


def load_document(path: str | Path) -> Document:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        content, page_offsets = _read_pdf(path)
        metadata = {"format": "pdf", "pages": len(page_offsets), "page_offsets": page_offsets}
    elif suffix in _TEXT_SUFFIXES:
        content = _read_text(path)
        metadata = {"format": suffix.removeprefix(".")}
    else:
        supported = ", ".join(sorted(_TEXT_SUFFIXES | {".pdf"}))
        raise UnsupportedFormatError(f"formato '{suffix}' não suportado (aceitos: {supported})")

    if not content.strip():
        raise EmptyDocumentError(f"nenhum texto extraído de {path}")
    return Document(source=str(path), content=content, metadata=metadata)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # arquivos legados criados no Windows (Notepad antigo, exports)
        return path.read_text(encoding="cp1252")


def _read_pdf(path: Path) -> tuple[str, list[dict[str, int]]]:
    """Concatena o texto das páginas registrando o offset onde cada uma começa.

    Os offsets permitem mapear qualquer posição do content de volta à página
    de origem — é o que sustenta as citações "p. 12" na resposta final.
    """
    reader = PdfReader(str(path))
    content = ""
    page_offsets: list[dict[str, int]] = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_offsets.append({"page": number, "start": len(content)})
        content += text
        if text and not text.endswith("\n"):
            content += "\n"
    return content, page_offsets
