# Treino no Google Colab (GPU gratuita)

Este guia executa o fine‑tuning usando LLaMA + LoRA no Colab.

## 1) Abra o Colab
- https://colab.research.google.com/
- Menu **Runtime → Change runtime type → GPU**

## 2) Monte o Google Drive
```python
from google.colab import drive

drive.mount('/content/drive')
```

## 3) Clone o repositório (no Drive)
```bash
%cd /content/drive/MyDrive
!git clone https://SEU_REPO_AQUI.git
%cd Tech-Challenge-3
```

## 4) Instale dependências
```bash
!pip install -r requirements.txt
```

Se aparecer erro de `triton` ou `bitsandbytes`, rode:
```bash
!pip install -U triton bitsandbytes
```

## 5) Login no Hugging Face
```python
from huggingface_hub import login

login()
```

## 6) Treinar
```bash
!python -m src.train_llama --config-path configs/llama_lora.yaml
```

## 7) Saídas
- `outputs/llama3-lora/` contém os pesos LoRA

## Dicas
- Se faltar memória GPU, reduza `max_seq_length` e `gradient_accumulation_steps`.
- Para persistir checkpoints, mantenha o projeto no Drive.
