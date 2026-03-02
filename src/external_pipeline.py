from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests
import typer
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from tqdm import tqdm

app = typer.Typer(add_completion=False)


@dataclass
class HttpClient:
    timeout: int = 30

    def get_text(self, url: str) -> str:
        response = requests.get(
            url,
            timeout=self.timeout,
            headers={
                "User-Agent": "TechChallenge3/1.0 (+https://example.org)"
            },
        )
        response.raise_for_status()
        return response.text

    def download(self, url: str, target_path: Path) -> None:
        response = requests.get(
            url,
            stream=True,
            timeout=self.timeout,
            headers={
                "User-Agent": "TechChallenge3/1.0 (+https://example.org)"
            },
        )
        response.raise_for_status()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as handler:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handler.write(chunk)


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower())
    return cleaned.strip("-")[:120] or "item"


def extract_main_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.text.strip() if soup.title else "CONITEC"

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    main = soup.find("main") or soup
    text = main.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n", text)
    return title, text.strip()


def find_links(base_url: str, html: str, max_links: int) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)
    links: list[str] = []
    for anchor in anchors:
        href = anchor["href"].strip()
        if href.startswith("/"):
            href = base_url.rstrip("/") + href
        if href.startswith("http"):
            links.append(href)
        if len(links) >= max_links:
            break
    return links


def iter_unique(items: Iterable[str]) -> list[str]:
    seen = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def scrape_conitec(
    client: HttpClient,
    base_url: str,
    output_dir: Path,
    prefer_html: bool,
    include_pdf: bool,
    max_links: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir = output_dir / "html"
    pdf_dir = output_dir / "pdf"
    html_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "conitec.jsonl"

    base_html = client.get_text(base_url)
    links = iter_unique(find_links(base_url, base_html, max_links))

    with records_path.open("w", encoding="utf-8") as handler:
        for link in tqdm(links, desc="CONITEC", unit="link"):
            is_pdf = link.lower().endswith(".pdf")
            if is_pdf and not include_pdf:
                continue

            try:
                if is_pdf:
                    filename = slugify(link.split("/")[-1]) + ".pdf"
                    target = pdf_dir / filename
                    client.download(link, target)
                    pdf_text = extract_text(str(target))
                    record = {
                        "source": "CONITEC",
                        "url": link,
                        "content_type": "application/pdf",
                        "text": pdf_text.strip(),
                        "local_path": str(target),
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    }
                else:
                    html = client.get_text(link)
                    title, text = extract_main_text(html)
                    filename = slugify(title) + ".html"
                    html_path = html_dir / filename
                    html_path.write_text(html, encoding="utf-8")
                    record = {
                        "source": "CONITEC",
                        "url": link,
                        "title": title,
                        "content_type": "text/html",
                        "text": text,
                        "local_path": str(html_path),
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    }
                handler.write(json.dumps(record, ensure_ascii=False) + "\n")
            except requests.RequestException as exc:
                typer.echo(f"Aviso: falha ao processar {link}: {exc}")

    return records_path


def download_pubmedqa(
    client: HttpClient,
    dataset_url: str | None,
    output_dir: Path,
) -> Path | None:
    if not dataset_url or dataset_url.endswith("/"):
        typer.echo(
            "PubMedQA: forneça um link direto para o arquivo de dados via --pubmedqa-url"
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = dataset_url.split("/")[-1]
    target = output_dir / filename
    client.download(dataset_url, target)

    if target.suffix.lower() == ".zip":
        with zipfile.ZipFile(target, "r") as zip_ref:
            zip_ref.extractall(output_dir)

    return target


def download_medquad(
    client: HttpClient,
    zip_url: str | None,
    output_dir: Path,
) -> Path | None:
    if not zip_url:
        typer.echo("MedQuAD: forneça o zip via --medquad-zip-url")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = zip_url.split("/")[-1]
    target = output_dir / filename
    client.download(zip_url, target)

    if target.suffix.lower() == ".zip":
        with zipfile.ZipFile(target, "r") as zip_ref:
            zip_ref.extractall(output_dir)

    return target


@app.command()
def main(
    output_dir: Path = typer.Option(Path("data/external"), help="Diretório de saída"),
    config_path: Path = typer.Option(
        Path("config/sources.json"), help="Arquivo de configuração"
    ),
    conitec: bool = typer.Option(True, help="Coletar CONITEC"),
    pubmedqa: bool = typer.Option(False, help="Coletar PubMedQA"),
    medquad: bool = typer.Option(False, help="Coletar MedQuAD"),
    pubmedqa_url: str | None = typer.Option(None, help="URL direta do PubMedQA"),
    medquad_zip_url: str | None = typer.Option(
        "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip",
        help="URL do zip do MedQuAD",
    ),
) -> None:
    config = load_config(config_path)
    client = HttpClient()

    if conitec:
        conitec_cfg = config.get("conitec", {})
        records = scrape_conitec(
            client,
            base_url=conitec_cfg.get("base_url"),
            output_dir=output_dir / "conitec",
            prefer_html=conitec_cfg.get("prefer_html", True),
            include_pdf=conitec_cfg.get("include_pdf", False),
            max_links=int(conitec_cfg.get("max_links", 200)),
        )
        typer.echo(f"CONITEC salvo em {records}")

    if pubmedqa:
        url = pubmedqa_url or config.get("pubmedqa", {}).get("dataset_url")
        target = download_pubmedqa(client, url, output_dir / "pubmedqa")
        if target:
            typer.echo(f"PubMedQA baixado em {target}")

    if medquad:
        url = medquad_zip_url
        target = download_medquad(client, url, output_dir / "medquad")
        if target:
            typer.echo(f"MedQuAD baixado em {target}")


if __name__ == "__main__":
    app()
