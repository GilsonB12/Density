"""CLI do density (Typer + Rich)."""

from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from density.chunking import Chunker, FixedChunker, RecursiveChunker
from density.ingestion import annotate_pages, load_document

app = typer.Typer(
    name="density",
    help="RAG com avaliação integrada: ingestão, busca híbrida e métricas de qualidade.",
    no_args_is_help=True,
)
console = Console()

_STRATEGIES: dict[str, type[FixedChunker] | type[RecursiveChunker]] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
}


@app.callback()
def main() -> None:
    """density — pergunte aos seus documentos, com métricas."""


@app.command()
def version() -> None:
    """Mostra a versão instalada."""
    console.print(f"density [bold cyan]{pkg_version('density')}[/bold cyan]")


@app.command()
def ingest(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    strategy: Annotated[str, typer.Option(help="fixed | recursive")] = "recursive",
    chunk_size: Annotated[int, typer.Option(min=1, help="orçamento por chunk, em tokens")] = 512,
    overlap: Annotated[int, typer.Option(min=0, help="tokens repetidos entre chunks")] = 64,
    show: Annotated[int, typer.Option(min=0, help="chunks exibidos no preview")] = 3,
) -> None:
    """Extrai texto e fatia em chunks (a gravação no banco chega na etapa 2)."""
    if strategy not in _STRATEGIES:
        valid = ", ".join(_STRATEGIES)
        console.print(f"[red]Estratégia inválida:[/red] '{strategy}'. Use: {valid}")
        raise typer.Exit(code=1)

    try:
        document = load_document(file)
        chunker: Chunker = _STRATEGIES[strategy](chunk_size=chunk_size, overlap=overlap)
    except ValueError as exc:
        console.print(f"[red]Erro:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    chunks = chunker.chunk(document)
    annotate_pages(document, chunks)

    total_tokens = sum(c.metadata["token_count"] for c in chunks)
    summary = Table(title=f"Ingestão: {file.name}", show_header=False)
    summary.add_row("Formato", document.metadata["format"])
    if "pages" in document.metadata:
        summary.add_row("Páginas", str(document.metadata["pages"]))
    summary.add_row("Caracteres", f"{len(document.content):,}")
    summary.add_row("Estratégia", f"{strategy} (chunk_size={chunk_size}, overlap={overlap})")
    summary.add_row("Chunks", str(len(chunks)))
    summary.add_row("Tokens (total)", f"{total_tokens:,}")
    if chunks:
        summary.add_row("Tokens por chunk (média)", f"{total_tokens / len(chunks):.0f}")
    console.print(summary)

    if show and chunks:
        preview = Table(title=f"Primeiros {min(show, len(chunks))} chunks")
        preview.add_column("#", justify="right")
        preview.add_column("página")
        preview.add_column("tokens", justify="right")
        preview.add_column("início do conteúdo")
        for chunk in chunks[:show]:
            page_start = chunk.metadata.get("page_start")
            page_end = chunk.metadata.get("page_end")
            page = "-" if page_start is None else f"{page_start}"
            if page_end is not None and page_end != page_start:
                page = f"{page_start}-{page_end}"
            snippet = " ".join(chunk.content.split())
            if len(snippet) > 70:
                snippet = snippet[:70] + "…"
            preview.add_row(str(chunk.index), page, str(chunk.metadata["token_count"]), snippet)
        console.print(preview)
