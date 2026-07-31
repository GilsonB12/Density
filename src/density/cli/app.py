"""CLI do density (Typer + Rich)."""

from importlib.metadata import version as pkg_version

import typer
from rich.console import Console

app = typer.Typer(
    name="density",
    help="RAG com avaliação integrada: ingestão, busca híbrida e métricas de qualidade.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """density — pergunte aos seus documentos, com métricas."""


@app.command()
def version() -> None:
    """Mostra a versão instalada."""
    console.print(f"density [bold cyan]{pkg_version('density')}[/bold cyan]")
