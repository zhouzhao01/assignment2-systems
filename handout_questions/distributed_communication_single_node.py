import os
import statistics
import timeit
from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank: int, world_size: int, device: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    backend = "gloo" if device == "cpu" else "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def reduce_step(x: torch.Tensor, times: int) -> float:
    """Return total execution time (seconds) for `times` local reductions."""
    time_start = timeit.default_timer()
    for _ in range(times):
        torch.sum(x, dim=-1)
    if x.is_cuda:
        torch.cuda.synchronize()
    return timeit.default_timer() - time_start


def log_records(records: List[Dict[str, float]], device: str, measure_steps: int, tensor_size: int) -> None:
    for entry in records:
        print(
            "[RUN] device={device} steps={steps} size={size} rank={rank} exp={exp} time_ms={time:.3f}".format(
                device=entry["device"],
                steps=entry["measure_steps"],
                size=entry["tensor_size"],
                rank=entry["rank"],
                exp=entry["experiment"],
                time=entry["time_ms"],
            )
        )

    times = [entry["time_ms"] for entry in records]
    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    print(
        "[SUMMARY] device={device} steps={steps} size={size} samples={samples} mean_ms={mean:.3f} std_ms={std:.3f}".format(
            device=device,
            steps=measure_steps,
            size=tensor_size,
            samples=len(times),
            mean=mean_ms,
            std=std_ms,
        )
    )


def distributed_demo(rank: int, world_size: int, device: str, config: Dict) -> None:
    setup(rank, world_size, device)

    tensor_sizes = config["tensor_sizes"]
    measure_steps_list = config["measure_steps"]
    experiments_per_config = config["experiments_per_config"]
    warmup_steps = config["warmup_steps"]

    if device == "cuda":
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            raise RuntimeError("CUDA requested but no GPUs are visible")
        local_device_index = rank % gpu_count
        torch.cuda.set_device(local_device_index)
        torch_device = torch.device(f"cuda:{local_device_index}")
        device_label = f"cuda:{local_device_index}"
        barrier_device_ids: Optional[List[int]] = [local_device_index]
        torch.cuda.manual_seed_all(1234 + rank)
    else:
        torch_device = torch.device("cpu")
        device_label = "cpu"
        barrier_device_ids = None

    torch.manual_seed(1234 + rank)

    def barrier() -> None:
        if barrier_device_ids is not None:
            dist.barrier(device_ids=barrier_device_ids)
        else:
            dist.barrier()

    for tensor_size in tensor_sizes:
        data = torch.randint(0, 10, (tensor_size,), dtype=torch.float32, device=torch_device)

        reduce_step(data, warmup_steps)
        barrier()

        for measure_steps in measure_steps_list:
            rank_records: List[Dict] = []
            for exp_idx in range(experiments_per_config):
                barrier()
                elapsed_ms = reduce_step(data, measure_steps) * 1000.0
                rank_records.append(
                    {
                        "device": device_label,
                        "measure_steps": measure_steps,
                        "tensor_size": tensor_size,
                        "rank": rank,
                        "experiment": exp_idx,
                        "time_ms": elapsed_ms,
                    }
                )

            gathered_records: List[List[Dict]] = [None for _ in range(world_size)]  # type: ignore
            dist.all_gather_object(gathered_records, rank_records)

            if rank == 0:
                flattened = [item for bucket in gathered_records for item in bucket]
                log_records(flattened, device_label, measure_steps, tensor_size)

            barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4

    experiment_config = {
        # Sweep tensor sizes (number of float32 elements) to cover ~1MB-100MB.
        # "tensor_sizes": [262_144, 2_621_440, 26_214_400],
        "tensor_sizes": [262_144,],
        # "measure_steps": [50, 100],
        "measure_steps": [50],
        "experiments_per_config": 100,
        "warmup_steps": 5,
    }

    devices_to_test: Iterable[str]
    if torch.cuda.is_available():
        devices_to_test = ("cpu", "cuda")
    else:
        devices_to_test = ("cpu",)

    for dev in devices_to_test:
        print(f"\n=== Starting experiments on device: {dev} ===")
        mp.spawn(
            fn=distributed_demo,
            args=(world_size, dev, experiment_config),
            nprocs=world_size,
            join=True,
        )
