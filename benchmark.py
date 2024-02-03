import time
import torch
import torch.nn.functional as F

from __init__ import HardSwish

def benchmark_activation_fn(activation_fn, dtype, input_size=(1, 3, 224, 224), num_iter=1000, warmup_iter=100):
    if isinstance(activation_fn, type):
        model = activation_fn()
    else:
        model = activation_fn

    # Warm-up phase
    warmup_data = torch.rand(input_size).to(dtype)
    for _ in range(warmup_iter):
        model(warmup_data)

    time_list = []
    for _ in range(5):
        # Randomize input data for each iteration
        input_data = torch.rand(input_size).to(dtype)

        start_time = time.process_time()  # Use process time for more accurate benchmarking
        for _ in range(num_iter):
            model(input_data)
        end_time = time.process_time()
        time_list.append(end_time - start_time)
    return sum(time_list)/len(time_list)

# Benchmark HardSwish and other activation functions
activation_fns = [HardSwish, F.hardswish, F.relu6]
dtypes = [torch.float64, torch.float32, torch.float16]

for activation_fn in activation_fns:
    for dtype in dtypes:
        print(f"Benchmarking {activation_fn.__name__} on {dtype}...")
        time_taken = benchmark_activation_fn(activation_fn, dtype)
        print(f"{activation_fn.__name__} time: {time_taken} seconds\n")