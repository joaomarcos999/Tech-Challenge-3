from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from pydantic import BaseModel, Field

app = typer.Typer(add_completion=False)


class Section(BaseModel):
    title: str
    body: str


class SyntheticDocument(BaseModel):
    title: str
    specialty_focus: str
    audience: str
    language_style: str
    created_at: str
    safety_alerts_included: bool
    sections: list[Section]
    metadata: dict[str, str] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        lines: list[str] = [f"# {self.title}", ""]
        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append(section.body)
            lines.append("")
        lines.append("---")
        lines.append("**Metadados**")
        for key, value in self.metadata.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
        return "\n".join(lines)


def build_document() -> SyntheticDocument:
    return SyntheticDocument(
        title="Protocolo Integrado de Plantão — Dor Torácica e Trauma de Extremidades",
        specialty_focus="Ortopedia, Cardiologia e Clínica Geral",
        audience="Médicos plantonistas",
        language_style="Técnico formal",
        created_at=datetime.now(timezone.utc).isoformat(),
        safety_alerts_included=True,
        sections=[
            Section(
                title="Objetivo",
                body=(
                    "Padronizar a abordagem inicial de pacientes com dor torácica e trauma "
                    "de extremidades em contexto de plantão, garantindo segurança clínica, "
                    "estabilidade hemodinâmica e tomada de decisão em tempo oportuno."
                ),
            ),
            Section(
                title="Escopo e exclusões",
                body=(
                    "Aplicável a adultos e adolescentes. Exclui gestantes com instabilidade, "
                    "crianças pequenas e pacientes com choque refratário que devem seguir "
                    "protocolos específicos de emergência avançada."
                ),
            ),
            Section(
                title="Triagem e avaliação inicial",
                body=(
                    "1) Avaliar sinais vitais, dor e nível de consciência.\n"
                    "2) Identificar red flags: dor torácica súbita intensa, síncope, hipotensão, "
                    "dessaturação, sangramento ativo, deformidade aberta, déficit neurovascular.\n"
                    "3) Classificar risco e priorizar atendimento conforme gravidade."
                ),
            ),
            Section(
                title="Fluxo de decisão — Dor torácica",
                body=(
                    "Se dor torácica: \n"
                    "- Realizar ECG em até 10 minutos e dosar marcadores conforme protocolo.\n"
                    "- Classificar risco clínico (alto, intermediário, baixo) conforme sinais e ECG.\n"
                    "- Alto risco: monitorização contínua, analgesia, suporte avançado e avaliação imediata.\n"
                    "- Intermediário: repetir ECG e exames seriados, manter observação.\n"
                    "- Baixo risco: considerar causas alternativas, analgesia e alta orientada se estável."
                ),
            ),
            Section(
                title="Fluxo de decisão — Trauma ortopédico",
                body=(
                    "Se trauma de extremidade: \n"
                    "- Inspecionar deformidade, sangramento e integridade cutânea.\n"
                    "- Avaliar perfusão distal, sensibilidade e motricidade.\n"
                    "- Suspeita de fratura: imobilizar, analgesia e solicitar radiografia adequada.\n"
                    "- Fratura exposta ou déficit neurovascular: acionar ortopedia imediatamente."
                ),
            ),
            Section(
                title="Exames complementares",
                body=(
                    "- ECG, troponina e radiografia conforme suspeita.\n"
                    "- Hemograma e eletrólitos se houver sinais sistêmicos.\n"
                    "- Radiografia em duas incidências para suspeita de fratura."
                ),
            ),
            Section(
                title="Tratamento inicial",
                body=(
                    "- Analgesia escalonada conforme intensidade.\n"
                    "- Oxigenoterapia se saturação < 94%.\n"
                    "- Imobilização de extremidades com suspeita de fratura.\n"
                    "- Monitorização cardíaca em casos de dor torácica com alto risco."
                ),
            ),
            Section(
                title="Critérios de internação",
                body=(
                    "- Dor torácica com alterações de ECG, marcadores elevados ou instabilidade.\n"
                    "- Trauma com fratura instável, exposta, ou déficit neurovascular.\n"
                    "- Necessidade de analgesia parenteral contínua ou monitorização."
                ),
            ),
            Section(
                title="Critérios de alta",
                body=(
                    "- Dor torácica de baixo risco com exames normais e estabilidade clínica.\n"
                    "- Trauma leve sem sinais de complicação, após analgesia e imobilização adequada.\n"
                    "- Orientações claras e retorno imediato em caso de piora."
                ),
            ),
            Section(
                title="Alertas de segurança (red flags)",
                body=(
                    "- Dor torácica com irradiação e sudorese, dispneia ou síncope.\n"
                    "- Hipotensão persistente ou alteração do nível de consciência.\n"
                    "- Extremidade fria, pálida ou sem pulso.\n"
                    "- Dor desproporcional após trauma (suspeita de síndrome compartimental)."
                ),
            ),
            Section(
                title="Documentação clínica mínima",
                body=(
                    "Registrar queixa principal, sinais vitais, exame físico objetivo, "
                    "exames solicitados, conduta e orientações de alta/internação."
                ),
            ),
        ],
        metadata={
            "origem": "Documento sintético",
            "especialidades": "Ortopedia, Cardiologia, Clínica Geral",
            "formato": "Protocolo detalhado, fluxo textual",
            "uso": "Fine-tuning LLM com dados sintéticos",
        },
    )


@app.command()
def main(output_dir: Path = typer.Option(..., help="Diretório de saída")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    document = build_document()
    markdown_path = output_dir / "documento_sintetico.md"
    jsonl_path = output_dir / "documento_sintetico.jsonl"

    markdown_path.write_text(document.to_markdown(), encoding="utf-8")

    with jsonl_path.open("w", encoding="utf-8") as handler:
        handler.write(document.model_dump_json())
        handler.write("\n")

    typer.echo(f"Gerado: {markdown_path}")
    typer.echo(f"Gerado: {jsonl_path}")


if __name__ == "__main__":
    app()
