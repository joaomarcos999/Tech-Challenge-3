# Tech Challenge 3 — Pipeline de Dados Clínicos (Sintéticos + Fontes Externas)

Este projeto prepara dados para fine‑tuning de LLMs clínicos usando:
- **Documentos sintéticos** (técnico‑formal) voltados a plantão médico.
- **Fontes externas**: CONITEC, PubMedQA e MedQuAD.

O foco é **rastreabilidade, limpeza e normalização** para um formato único de treino.

---

## ✅ O que foi feito até agora

### 1) Documento clínico sintético (técnico‑formal)
Gera **1 protocolo detalhado** integrando **ortopedia, cardiologia e clínica geral**, com:
- Objetivo
- Escopo e exclusões
- Triagem e avaliação inicial
- Fluxos de decisão (dor torácica e trauma ortopédico)
- Exames complementares
- Tratamento inicial
- Critérios de internação e alta
- **Alertas de segurança (red flags)**

Arquivos gerados:
- `data/output/documento_sintetico.md`
- `data/output/documento_sintetico.jsonl`

### 2) Pipeline de coleta das fontes externas
Coleta dados externos em `data/external/`:
- **CONITEC**: HTML e PDF (com extração de texto)
- **PubMedQA**: download do repositório oficial
- **MedQuAD**: download do repositório oficial

### 3) Normalização para JSONL unificado
Consolida todas as fontes no schema:
```json
{
	"instruction": "...",
	"context": "...",
	"response": "...",
	"source": "CONITEC | PubMedQA | MedQuAD",
	"meta": {"...": "..."}
}
```
O arquivo final fica em `data/normalized/train.jsonl`.

---

## 📦 Requisitos
- Python 3.10+

---

## ⚙️ Instalação
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## ▶️ Como usar

### 1) Gerar documento sintético
```powershell
& "C:/Program Files/Python313/python.exe" -m src.generate_document --output-dir data\output
```

### 1.1) Preparar dados internos (sintéticos)
Copie os documentos internos para `data/internal/` (ex.: `data/output/documento_sintetico.md`) e execute:
```powershell
& "C:/Program Files/Python313/python.exe" -m src.internal_pipeline --internal-dir data\internal --output-path data\normalized\internal.jsonl
```

### 2) Baixar fontes externas
```powershell
& "C:/Program Files/Python313/python.exe" -m src.external_pipeline --output-dir data\external
```

Para baixar PubMedQA e MedQuAD:
```powershell
& "C:/Program Files/Python313/python.exe" -m src.external_pipeline --output-dir data\external --pubmedqa --medquad --pubmedqa-url https://github.com/pubmedqa/pubmedqa/archive/refs/heads/master.zip
```

### 3) Normalizar tudo em JSONL único
```powershell
& "C:/Program Files/Python313/python.exe" -m src.normalize_datasets --output-path data\normalized\train.jsonl
```

### 4) Mesclar interno + externo (com deduplicação)
```powershell
& "C:/Program Files/Python313/python.exe" -m src.merge_datasets --external-path data\normalized\train.jsonl --internal-path data\normalized\internal.jsonl --output-path data\normalized\train_merged.jsonl
```

Opcional: limitar registros por fonte.
```powershell
& "C:/Program Files/Python313/python.exe" -m src.normalize_datasets --limit-per-source 1000
```

---

## 📂 Estrutura de saída
- `data/output/` → documento sintético
- `data/external/` → downloads das fontes externas
- `data/normalized/train.jsonl` → dataset consolidado final

---

## 🧹 Limpeza aplicada (CONITEC)
- Remove conteúdo administrativo (termos de uso, acessibilidade, etc.)
- Remove blocos de compartilhamento e metadados sociais
- Descarta textos muito curtos após limpeza

## 🔒 Estrutura de anonimização
O projeto já inclui uma camada de anonimização pronta para dados com PII/PHI:
- Módulo: `src/anonymization.py`
- Aplicado em: `src/internal_pipeline.py` e `src/normalize_datasets.py`
- Regras padrão: CPF, CNPJ, RG, CNS, CEP, e‑mail, telefone e datas
- Blocklist de nomes: `config/name_blocklist.txt`

---

## ⚠️ Observações
- Não há dados reais de pacientes.
- Algumas URLs da CONITEC podem retornar 404; o pipeline ignora automaticamente.
- PubMedQA exige link direto para o arquivo de dados.

---

## 📌 Arquivos principais
- `src/generate_document.py` → documento sintético
- `src/external_pipeline.py` → coleta das fontes externas
- `src/normalize_datasets.py` → normalização final
- `config/sources.json` → configuração das fontes

---

## 🧠 Fine‑tuning com LLaMA (suporte à decisão clínica)

### Visão geral
- **Estratégia**: LoRA/QLoRA para reduzir custo e acelerar iterações
- **Modelo**: LLaMA‑3 Instruct (ex.: `meta-llama/Meta-Llama-3-8B-Instruct`)
- **Entrada**: `data/normalized/train_merged.jsonl`

### Configuração
Arquivo YAML padrão: `configs/llama_lora.yaml`

### Treino
```powershell
& "C:/Program Files/Python313/python.exe" -m src.train_llama --config-path configs\llama_lora.yaml
```

### Treino no Google Colab (GPU gratuita)
Guia completo em `colab_train.md`.

### Saída
- Pesos LoRA e tokenizer em `outputs/llama3-lora/`

### Observações
- Ajuste `max_seq_length` e `batch_size` conforme a GPU disponível.
- Para ambientes sem `bf16`, ajuste `bf16` para `false` no código.
