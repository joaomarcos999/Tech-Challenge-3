from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

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


def is_conitec_relevant(url: str, text: str) -> bool:
    admin_keywords = [
        "termos de uso",
        "acessibilidade",
        "dados abertos",
        "governodigital",
        "governo digital",
        "faq-login-unico",
        "atendimento-gov.br",
        "biblioteca-virtual",
    ]
    if any(keyword in url for keyword in admin_keywords):
        return False
    if any(keyword in text.lower() for keyword in admin_keywords):
        return False
    if "protocolos-clinicos-e-diretrizes-terapeuticas" in url:
        return True
    return len(text) > 800


def clean_conitec_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    skip_phrases = {
        "compartilhe",
        "link para copiar",
        "publicado em",
        "atualizado em",
        "tags:",
        "facebook",
        "twitter",
        "linkedin",
        "whatsapp",
    }
    for line in lines:
        if not line:
            continue
        lower = line.lower()
        if any(phrase in lower for phrase in skip_phrases):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def load_conitec(
    conitec_path: Path,
    limit: int | None,
    name_blocklist: list[str],
) -> Iterable[Record]:
    with conitec_path.open("r", encoding="utf-8") as handler:
        for index, line in enumerate(handler):
            if limit is not None and index >= limit:
                break
            item = json.loads(line)
            text = item.get("text") or ""
            if not text.strip():
                continue
            url = item.get("url", "")
            if not is_conitec_relevant(url, text):
                continue
            text = clean_conitec_text(text)
            text = anonymize_text(text, name_blocklist=name_blocklist)
            if len(text) < 500:
                continue
            title = item.get("title", "CONITEC")
            instruction = "Forneça o conteúdo técnico do protocolo abaixo."
            context = f"Título: {title}\n\n{text}"
            response = text
            yield Record(
                instruction=instruction,
                context=context,
                response=response,
                source="CONITEC",
                meta={
                    "url": url,
                    "content_type": item.get("content_type", ""),
                },
            )


def load_pubmedqa(
    pubmedqa_path: Path,
    limit: int | None,
    name_blocklist: list[str],
) -> Iterable[Record]:
    data = json.loads(pubmedqa_path.read_text(encoding="utf-8"))
    for index, (pmid, item) in enumerate(data.items()):
        if limit is not None and index >= limit:
            break
        question = item.get("QUESTION", "").strip()
        contexts = item.get("CONTEXTS", [])
        long_answer = item.get("LONG_ANSWER", "").strip()
        final_decision = item.get("final_decision", "").strip()
        if not question or not long_answer:
            continue
        context = anonymize_text("\n\n".join(contexts), name_blocklist=name_blocklist)
        response = anonymize_text(long_answer, name_blocklist=name_blocklist)
        if final_decision:
            response = f"Resposta: {final_decision}.\n\n{long_answer}"
        yield Record(
            instruction=question,
            context=context,
            response=response,
            source="PubMedQA",
            meta={
                "pmid": pmid,
                "year": item.get("YEAR", ""),
            },
        )


def parse_medquad_xml(xml_path: Path, name_blocklist: list[str]) -> Iterable[Record]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    focus = root.findtext("Focus", default="").strip()
    for qa_pair in root.findall(".//QAPair"):
        question = qa_pair.findtext("Question", default="").strip()
        answer = qa_pair.findtext("Answer", default="").strip()
        if not question or not answer:
            continue
        context = anonymize_text(
            f"Foco: {focus}" if focus else "",
            name_blocklist=name_blocklist,
        )
        yield Record(
            instruction=question,
            context=context,
            response=anonymize_text(answer, name_blocklist=name_blocklist),
            source="MedQuAD",
            meta={
                "doc_id": root.attrib.get("id", ""),
                "url": root.attrib.get("url", ""),
            },
        )


def load_medquad(
    medquad_root: Path,
    limit: int | None,
    name_blocklist: list[str],
) -> Iterable[Record]:
    count = 0
    for xml_path in medquad_root.rglob("*.xml"):
        for record in parse_medquad_xml(xml_path, name_blocklist):
            yield record
            count += 1
            if limit is not None and count >= limit:
                return


@app.command()
def main(
    output_path: Path = typer.Option(
        Path("data/normalized/train.jsonl"), help="Arquivo JSONL de saída"
    ),
    conitec_path: Path = typer.Option(
        Path("data/external/conitec/conitec.jsonl"),
        help="JSONL do CONITEC",
    ),
    pubmedqa_path: Path = typer.Option(
        Path("data/external/pubmedqa/pubmedqa-master/data/ori_pqal.json"),
        help="JSON do PubMedQA",
    ),
    medquad_root: Path = typer.Option(
        Path("data/external/medquad/MedQuAD-master"),
        help="Diretório raiz do MedQuAD",
    ),
    limit_per_source: int | None = typer.Option(
        None, help="Limite de registros por fonte"
    ),
    name_blocklist_path: Path | None = typer.Option(
        Path("config/name_blocklist.txt"), help="Lista de nomes para mascarar"
    ),
) -> None:
    name_blocklist: list[str] = []
    if name_blocklist_path and name_blocklist_path.exists():
        name_blocklist = [
            line.strip()
            for line in name_blocklist_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    records = []
    records.extend(load_conitec(conitec_path, limit_per_source, name_blocklist))
    records.extend(load_pubmedqa(pubmedqa_path, limit_per_source, name_blocklist))
    records.extend(load_medquad(medquad_root, limit_per_source, name_blocklist))

    total = write_jsonl(records, output_path)
    typer.echo(f"Gerado {total} registros em {output_path}")


if __name__ == "__main__":
    app()
