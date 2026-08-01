from typer.testing import CliRunner

from density.cli.app import app
from density.config import Settings

runner = CliRunner()


def write_doc(tmp_path, text="Primeiro parágrafo de teste com conteúdo.\n\nSegundo parágrafo."):
    file = tmp_path / "doc.txt"
    file.write_text(text, encoding="utf-8")
    return file


def test_ingest_dry_run_shows_chunk_summary(tmp_path):
    file = write_doc(tmp_path)
    result = runner.invoke(
        app, ["ingest", str(file), "--dry-run", "--chunk-size", "32", "--overlap", "4"]
    )
    assert result.exit_code == 0
    assert "chunk" in result.output.lower()


def test_ingest_dry_run_pdf_reports_pages():
    result = runner.invoke(app, ["ingest", "tests/fixtures/sample.pdf", "--dry-run"])
    assert result.exit_code == 0
    assert "2" in result.output


def test_ingest_dry_run_does_not_require_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("density.cli.app.get_settings", lambda: Settings(_env_file=None))
    result = runner.invoke(app, ["ingest", str(write_doc(tmp_path)), "--dry-run"])
    assert result.exit_code == 0


def test_ingest_without_key_fails_with_guidance(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("density.cli.app.get_settings", lambda: Settings(_env_file=None))
    result = runner.invoke(app, ["ingest", str(write_doc(tmp_path))])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_ingest_rejects_unknown_strategy(tmp_path):
    result = runner.invoke(app, ["ingest", str(write_doc(tmp_path)), "--strategy", "banana"])
    assert result.exit_code != 0


def test_ingest_rejects_missing_file():
    result = runner.invoke(app, ["ingest", "nao_existe.txt"])
    assert result.exit_code != 0
