from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import typer

from anonymization import anonymize_text

app = typer.Typer(add_completion=False)


@dataclass
class Record:
    instruction: str
    context: str
    response: str
    source: str
    meta: dict[str, str]


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def load_synthetic_document(path: Path, name_blocklist: list[str]) -> Record:
    content = path.read_text(encoding="utf-8")
    cleaned = normalize_text(anonymize_text(content, name_blocklist=name_blocklist))
    return Record(
        instruction="Forneça o conteúdo técnico do protocolo abaixo.",
        context=cleaned,
        response=cleaned,
        source="Interno-Sintetico",
        meta={
            "filename": path.name,
            "type": "protocolo",
        },
    )


def load_internal_folder(folder: Path, name_blocklist: list[str]) -> Iterable[Record]:
    for path in folder.rglob("*.md"):
        yield load_synthetic_document(path, name_blocklist)


def write_jsonl(records: Iterable[Record], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handler:
        for record in records:
            payload = {
                "instruction": record.instruction,
                "context": record.context,
                "response": record.response,
                "source": record.source,
                "meta": record.meta,
            }
            handler.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


@app.command()
def main(
    internal_dir: Path = typer.Option(
        Path("data/internal"), help="Diretório de documentos internos"
    ),
    output_path: Path = typer.Option(
        Path("data/normalized/internal.jsonl"), help="JSONL de saída"
    ),
    name_blocklist_path: Path | None = typer.Option(
        Path("config/name_blocklist.txt"), help="Lista de nomes para mascarar"
    ),
) -> None:
    if not internal_dir.exists():
        raise typer.Exit(code=1)

    name_blocklist: list[str] = []
    if name_blocklist_path and name_blocklist_path.exists():
        name_blocklist = [
            line.strip()
            for line in name_blocklist_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    records = list(load_internal_folder(internal_dir, name_blocklist))
    total = write_jsonl(records, output_path)
    typer.echo(f"Gerado {total} registros internos em {output_path}")


if __name__ == "__main__":
    app()
