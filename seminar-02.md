# Семинар 2: Tensor Parallelism + Multi-Node

На прошлом семинаре мы запустили обучение GPT-2 на одной ноде с помощью DDP.

Сегодня мы обучим Llama на 1.1B параметров:
1. Научимся запускать обучение на **нескольких нодах**
2. И запустим **Tensor Parallelism**

Весь код лежит в `scripts/train_dp_tp.py`.

## Базовые понятия

- `device mesh` — N-мерная сетка GPU с именованными осями; удобный способ описать, как GPU организованы для разных видов параллелизма
- `ColwiseParallel` — шардирование матрицы по столбцам; вход реплицирован, выход шардирован
- `RowwiseParallel` — шардирование матрицы по строкам; вход шардирован, выход собирается через AllReduce

## Multi-Node запуск

До сих пор мы запускали `torchrun` только на одной ноде с `--standalone`. При переходе на несколько нод нужно:

1. Запустить `torchrun` **на каждой ноде отдельно**
2. Указать, как ноды находят друг друга через `--rdzv-*` параметры

### Параметры torchrun для multi-node

```bash
torchrun \
    --rdzv-id <уникальный_id_задачи> \        # любая строка, одинаковая на всех нодах
    --rdzv-backend c10d \                     # бэкенд rendezvous (c10d — рекомендуется)
    --rdzv-endpoint <IP_главной_ноды>:5001 \  # адрес мастер-ноды
    --nnodes 2 \                              # число нод
    --nproc-per-node gpu \                    # используем все GPU на ноде
    train_dp_tp.py ...
```

### Запуск

Single node:
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

Multinode, запускаем на каждой ноде:
```bash
export JOB_NAME=seminar-02-tp-multinode; \
export TORCHELASTIC_ERROR_FILE=error.json; \
export OMP_NUM_THREADS=1; \
uv run torchrun \
    --rdzv-id $JOB_NAME \
    --rdzv-backend c10d \
    --rdzv-endpoint node0:5001 \
    --nnodes 2 \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir logs \
    scripts/train_dp_tp.py \
    --experiment-name tp-llama-1b-multinode \
    -d tatsu-lab/alpaca \
    -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tp 4 \
    --batch-size 16 \
    --num-epochs 5
```
