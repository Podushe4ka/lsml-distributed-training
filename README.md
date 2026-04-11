# LSML: Distributed training

Код для семинаров по распределенному обучению LLM в рамках курса Large Scale ML.

Сетап:
```bash
source .env && make vendor
```

В качестве примера будем учить GPT2 (124M параметров).

Запуск обучения на одной карте:
```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/train_single.py \
    -d tatsu-lab/alpaca \
    -m openai-community/gpt2 \
    -e test01-single
```

На 4 картах в DDP:
```bash
export TORCHELASTIC_ERROR_FILE=error.json; \
export OMP_NUM_THREADS=1; \
export CUDA_VISIBLE_DEVICES=0,1,2,3; \
uv run torchrun \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir logs \
    scripts/train_ddp.py \
    -d tatsu-lab/alpaca \
    -m openai-community/gpt2 \
    -e test01-ddp
```

Обучаем Llama-1.1B в DP+TP сетапе:
```bash
export TORCHELASTIC_ERROR_FILE=error.json; \
export OMP_NUM_THREADS=1; \
uv run torchrun \
    --standalone \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir logs \
    scripts/train_dp_tp.py \
    --experiment-name tp-llama-1b \
    -d tatsu-lab/alpaca \
    -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tp 4 \
    --batch-size 16 \
    --num-epochs 5
```
