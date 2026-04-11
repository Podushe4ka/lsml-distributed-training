# Семинар 1: DDP

Сегодня мы обучим GPT-2 на нескольких GPU используя DDP.

Весь код лежит в `scripts/train_ddp.py`.

## Базовые понятия

- `world size`: общее число участвующих в обучении GPU
- `rank`: глобальный уникальный id конкретной GPU (от `0` до `world_size - 1` включительно)
- `local_rank`: локальный id для текущей машины (от `0` до `torch.cuda.device_count() - 1` включительно)

## Как устроено распределенное обучение

Как мы добьемся параллелизации? Аналогично как это делает `multiprocessing.Pool.map` в Python: распределим нагрузку между всеми 'ядрами'.
Нагрузка = батчи из датасета. Но есть челлендж: как убедиться, что модель на всех GPU одинакова?

Для простоты предположим:
1. Модель и оптимизатор полностью помещаются на каждом GPU
2. Мы инициализируем модель одинаково на всех GPU
3. Оптимизатор настроен одинаково на всех GPU

Канонический PyTorch training loop:
```python
loss = model(**batch) # 1. Forward pass asynchronously
optimizer.zero_grad() # 2. Reset gradients asynchronously
loss.backward()       # 3. calculates gradients asynchronously
optimizer.step()      # 4. synchronize gradients & update weights
```

Первые 3 строки выше могут быть выполнены асинхронно. `loss.backward()` вычислит градиенты в каждом из процессов. Особенность в том, что `optimizer.step()` должен синхронизировать градиенты между всеми процессами перед обновлением параметров модели.

Как Pytorch этого добивается?
1. Запуск N копий скрипта с помощью torchrun (парадигма SPMD)
2. Разделение данных между воркерами с помощью DistributedSampler
3. Синхронизация градиентов с помощью DistributedDataParallel

## Использование `torchrun`

Когда мы используем `torchrun` для запуска распределённого обучения, он **запускает N отдельных процессов** (где N — число GPU, указанное в `--nproc-per-node`), все запускают один и тот же скрипт обучения (Single Program Multiple Data, SPMD):

```
> torchrun --nproc-per-node 3 train_ddp.py ...
Launches subproc `$RANK=0 $WORLD_SIZE=3 train_ddp.py ...`
Launches subproc `$RANK=1 $WORLD_SIZE=3 train_ddp.py ...`
Launches subproc `$RANK=2 $WORLD_SIZE=3 train_ddp.py ...`
```

Он также настраивает синхронизацию между процессами. Каждый процесс выполняет один и тот же код и должен синхронизироваться в определённых точках, поэтому процесс получает идентификатор (`rank`), который указывает ему, какую GPU использовать.

При работе на нескольких нодах нужно запустить `torchrun` на каждой машине, но в остальном всё работает одинаково.

Вот некоторые базовые параметры `torchrun`:
- `--standalone` — используется при работе только на одной ноде
- `--nnodes` — количество нод; в данном случае 1, но при переходе к нескольким нодам будет > 1
- `--nproc-per-node` — количество процессов. `gpu` означает использование всех доступных GPU
- `--redirects 3` — перенаправляет stdout и stderr в файлы
- `--log-dir ../logs` — настраивает директорию для логов

### TORCHELASTIC_ERROR_FILE

**Очень важно включать это для отладки!**

Когда один из воркеров (включая потоки внутри) выдаёт ошибку, `torchrun` сохраняет её по пути, указанному в этой переменной окружения.

Также нужно добавить декоратор `@record` (импортируется из `torch.distributed.elastic.multiprocessing.errors import record`) к мейну:

```diff
+@record
 def main():
```

### OMP_NUM_THREADS

По умолчанию PyTorch пытается использовать все доступные ядра при вычислениях, даже при работе на GPU. Поскольку у нас запущено несколько процессов PyTorch, без установки `OMP_NUM_THREADS` все они попытаются использовать все доступные ядра.

Можно вручную проверить количество доступных ядер и разделить их соответственно. Например, если на машине 32 ядра и 8 GPU, можно установить `OMP_NUM_THREADS=4`.

## Изменения в коде

### Вызов `dist.init_process_group()` и `torch.cuda.set_device()`

Оба эти вызова необходимо сделать перед вызовом других dist API.

`dist.init_process_group()` будет блокироваться до тех пор, пока его не вызовут `WORLD_SIZE` процессов.

```diff
+from torch import distributed as dist


+   rank = int(os.getenv("RANK", "0"))
+   local_rank = rank % torch.cuda.device_count()
+   world_size = int(os.getenv("WORLD_SIZE", "1"))

-   device = torch.device(f"cuda") 
+   device = torch.device(f"cuda:{local_rank}")
+   torch.cuda.set_device(device)

+   dist.init_process_group(rank=rank, world_size=world_size, device_id=device)
```

Если не вызвать `torch.cuda.set_device`, процессы могут использовать неправильное CUDA-устройство.

### Включение rank в логи

Это полезно, когда все процессы пишут в один файл, или просто при просмотре лог-файла — удобно видеть `rank` в каждой строке:

```diff
 logging.basicConfig(
-    format=f"[%(asctime)s] %(levelname)s:%(message)s",
+    format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
     level=logging.INFO,
 )
```

### Использование DistributedDataParallel

```diff
+from torch.nn.parallel import DistributedDataParallel

 with device: 
     model = AutoModelForCausalLM.from_config(config, dtype=dtype)

+model = DistributedDataParallel(model, device_ids=[local_rank])
```

Можно предположить, что модуль DDP разделяет батчи между процессами, но это совсем не так!

Это класс-обёртка модели, который обеспечивает **синхронизацию градиентов перед вызовом `optimizer.step()`**. Рекомендуется прочитать документацию по нему: [https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

Этот класс также следит за тем, чтобы *веса модели были одинаковыми на воркерах*.

Это достигается через [специальные хуки модели](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/parallel/distributed.py#L939), суммирующие все градиенты со всех рангов:

```python
# NOTE: код PyTorch https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/parallel/distributed.py#L939
gradient = param.grad / self.process_group.size()
gradient = fcol.all_reduce(gradient, "sum", self.process_group)
```

### Использование DistributedSampler

В обычном скрипте обучения мы используем `torch.utils.data.DataLoader` для батчинга данных. Один из аргументов DataLoader — `sampler`, который выбирает элементы из датасета при формировании батчей. Можно представить сэмплер примерно так:

```python
def simple_sampler():
    worker_len = len(dataset)
    return random.choice(range(worker_len))
```

Хитрость `DistributedSampler` в том, что он под капотом 'делит' весь датасет между воркерами. Делить сам датасет не нужно — сэмплер просто выбирает индексы из определённого подмножества:

```python
def distributed_sampler():
    worker_len = len(dataset) // dist.get_world_size()
    return dist.get_rank() * worker_len + random.choice(range(worker_len))
```

Изменений в коде минимум:

```diff
+from torch.utils.data.distributed import DistributedSampler

 dataloader = DataLoader(
     train_data,
     batch_size=args.batch_size,
-    shuffle=True,
-    drop_last=True,
     collate_fn=default_data_collator,
+    sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
 )
```

Также необходимо вызывать [DistributedSampler.set_epoch](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler). Из документации PyTorch:

```diff
 for state["epoch"] in range(state["epoch"], args.num_epochs):
+    dataloader.sampler.set_epoch(state["epoch"])
     batches = iter(dataloader)
```

> In distributed mode, calling the set_epoch() method at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.

### Загрузка модели и данных сначала на rank 0

Это нужно в первую очередь потому, что это операции записи на диск.

Если не ограничить это rank-ом 0, все ранги могут попытаться скачать данные одновременно, и это всё замедлит.

Добавим простой контекстный менеджер:

```python
@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()
```

Загрузка весов модели и токенизатора:

```diff
+with rank0_first():
     config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
     with device:
         model = AutoModelForCausalLM.from_config(config, dtype=dtype)
```

Загрузка данных:

```diff
+with rank0_first():
     train_data = _load_and_preprocess_data(args, tokenizer, config)
```

### Создание директории эксперимента только на rank 0

Обратите внимание на вызовы `dist.barrier()` до и после создания директории. **Они очень важны!**

Поскольку мы проверяем существование директории прямо перед её созданием, нужно убедиться, что **все процессы уже сделали эту проверку**. Первый вызов `dist.barrier()` гарантирует, что все воркеры проверили существование директории. И только после этого мы создаём её на rank 0.

```diff
+dist.barrier()
+if rank == 0:
     exp_dir.mkdir(parents=True, exist_ok=True)
+dist.barrier()
```

### Сохранение чекпоинтов только на rank 0

Мы также хотим, чтобы чекпоинты сохранял только один ранг. Иначе ранги могут писать в один файл и повредить его.

```diff
 if state["global_step"] % args.ckpt_freq == 0:
+    if rank == 0:
         torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
         torch.save(model.state_dict(), exp_dir / "model.pt")
         torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
         with open(exp_dir / "state.json", "w") as fp:
              json.dump(state, fp)
+    dist.barrier()
```

## Оптимизация памяти — Zero Redundancy Optimizer

DDP хранит полную модель и оптимизатор на каждом GPU. Особенно расточительным является хранение оптимизатора. К счастью, есть [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), который легко добавить для снижения потребления памяти:

```diff
+ from torch.distributed.optim import ZeroRedundancyOptimizer

-optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
+optimizer = ZeroRedundancyOptimizer(
+    model.parameters(), optimizer_class=torch.optim.AdamW, lr=args.lr
+)
```

К сожалению, код сохранения state dict для ZeRO работает очень медленно.
