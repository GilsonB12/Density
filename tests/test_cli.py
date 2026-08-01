from typer.testing import CliRunner

from density.cli.app import app

runner = CliRunner()


def test_ingest_txt_shows_chunk_summary(tmp_path):
    file = tmp_path / "doc.txt"
    file.write_text(
        "Primeiro parágrafo de teste com conteúdo.\n\nSegundo parágrafo de teste.",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["ingest", str(file), "--chunk-size", "32", "--overlap", "4"])
    assert result.exit_code == 0
    assert "chunk" in result.output.lower()


def test_ingest_pdf_reports_pages(tmp_path):
    result = runner.invoke(app, ["ingest", "tests/fixtures/sample.pdf"])
    assert result.exit_code == 0
    assert "2" in result.output


def test_ingest_rejects_unknown_strategy(tmp_path):
    file = tmp_path / "doc.txt"
    file.write_text("texto qualquer", encoding="utf-8")
    result = runner.invoke(app, ["ingest", str(file), "--strategy", "banana"])
    assert result.exit_code != 0


def test_ingest_rejects_missing_file():
    result = runner.invoke(app, ["ingest", "nao_existe.txt"])
    assert result.exit_code != 0
