from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import typer

app = typer.Typer(add_completion=False)


@dataclass
class Record:
    instruction: str
    context: str
    response: str
    source: str
    meta: dict[str, str]


def record_key(record: Record) -> str:
    payload = f"{record.instruction}\n{record.context}\n{record.response}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterable[Record]:
    with path.open("r", encoding="utf-8") as handler:
        for line in handler:
            item = json.loads(line)
            yield Record(
                instruction=item.get("instruction", ""),
                context=item.get("context", ""),
                response=item.get("response", ""),
                source=item.get("source", ""),
                meta=item.get("meta", {}),
            )


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
    external_path: Path = typer.Option(
        Path("data/normalized/train.jsonl"), help="JSONL externo"
    ),
    internal_path: Path = typer.Option(
        Path("data/normalized/internal.jsonl"), help="JSONL interno"
    ),
    output_path: Path = typer.Option(
        Path("data/normalized/train_merged.jsonl"), help="JSONL final"
    ),
) -> None:
    seen: set[str] = set()
    merged: list[Record] = []

    for path in [external_path, internal_path]:
        if not path.exists():
            raise typer.Exit(code=1)
        for record in iter_jsonl(path):
            key = record_key(record)
            if key in seen:
                continue
            seen.add(key)
            merged.append(record)

    total = write_jsonl(merged, output_path)
    typer.echo(f"Gerado {total} registros em {output_path}")


if __name__ == "__main__":
    app()
