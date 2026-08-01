"""CLI do density (Typer + Rich)."""

from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Annotated

import psycopg
import typer
from openai import OpenAIError
from rich.console import Console
from rich.table import Table

from density.chunking import Chunker, FixedChunker, RecursiveChunker
from density.config import get_settings
from density.embedding import OpenAIEmbeddingProvider
from density.ingestion import annotate_pages, load_document
from density.models import Chunk, Document
from density.pipeline import Indexer
from density.storage import PgVectorStore

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
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="só extrai e fatia — não embeda nem grava")
    ] = False,
) -> None:
    """Extrai texto, fatia em chunks, embeda e grava no pgvector."""
    if strategy not in _STRATEGIES:
        valid = ", ".join(_STRATEGIES)
        console.print(f"[red]Estratégia inválida:[/red] '{strategy}'. Use: {valid}")
        raise typer.Exit(code=1)

    try:
        chunker: Chunker = _STRATEGIES[strategy](chunk_size=chunk_size, overlap=overlap)
    except ValueError as exc:
        console.print(f"[red]Erro:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    extra_rows: list[tuple[str, str]] = []
    try:
        if dry_run:
            document = load_document(file)
            chunks = chunker.chunk(document)
            annotate_pages(document, chunks)
        else:
            document, chunks, extra_rows = _index(file, chunker)
    except ValueError as exc:
        console.print(f"[red]Erro:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    strategy_label = f"{strategy} (chunk_size={chunk_size}, overlap={overlap})"
    _render_summary(file, document, chunks, strategy_label, extra_rows)
    _render_preview(chunks, show)
    if dry_run:
        console.print("[dim]--dry-run: nada foi embedado nem gravado no banco.[/dim]")


def _index(file: Path, chunker: Chunker) -> tuple[Document, list[Chunk], list[tuple[str, str]]]:
    """Roda o pipeline completo e devolve linhas extras para o resumo."""
    settings = get_settings()
    try:
        embedder = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    except OpenAIError as exc:
        console.print(
            "[red]Erro:[/red] chave da OpenAI ausente — defina OPENAI_API_KEY no arquivo "
            ".env (modelo em .env.example) ou use --dry-run para só visualizar chunks."
        )
        raise typer.Exit(code=1) from exc

    try:
        store = PgVectorStore(settings.database_url)
    except psycopg.OperationalError as exc:
        console.print(f"[red]Erro:[/red] Postgres inacessível — rode: docker compose up -d\n{exc}")
        raise typer.Exit(code=1) from exc

    try:
        document, chunks = Indexer(chunker=chunker, embedder=embedder, store=store).index(file)
        stored_total = store.count_chunks()
    finally:
        store.close()

    extra_rows = [
        ("Gravado em", f"pgvector ({embedder.model_name}, {embedder.dimensions}d)"),
        ("Chunks no banco (total)", str(stored_total)),
    ]
    tokens = sum(c.metadata["token_count"] for c in chunks)
    price = embedder.price_per_million_tokens
    if price is not None:
        extra_rows.append(("Custo estimado", f"US$ {tokens / 1_000_000 * price:.6f}"))
    return document, chunks, extra_rows


def _render_summary(
    file: Path,
    document: Document,
    chunks: list[Chunk],
    strategy_label: str,
    extra_rows: list[tuple[str, str]],
) -> None:
    total_tokens = sum(c.metadata["token_count"] for c in chunks)
    summary = Table(title=f"Ingestão: {file.name}", show_header=False)
    summary.add_row("Formato", document.metadata["format"])
    if "pages" in document.metadata:
        summary.add_row("Páginas", str(document.metadata["pages"]))
    summary.add_row("Caracteres", f"{len(document.content):,}")
    summary.add_row("Estratégia", strategy_label)
    summary.add_row("Chunks", str(len(chunks)))
    summary.add_row("Tokens (total)", f"{total_tokens:,}")
    if chunks:
        summary.add_row("Tokens por chunk (média)", f"{total_tokens / len(chunks):.0f}")
    for label, value in extra_rows:
        summary.add_row(label, value)
    console.print(summary)


def _render_preview(chunks: list[Chunk], show: int) -> None:
    if not show or not chunks:
        return
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
