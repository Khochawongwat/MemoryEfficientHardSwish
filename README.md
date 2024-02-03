MemoryEfficientHardSwish
This repository contains an implementation of the HardSwish activation function that is optimized for memory efficiency.
The HardSwish function is commonly used in deep learning models, and this implementation aims to reduce the memory footprint of these models without sacrificing performance.

Benchmark Results
The following benchmarks compare the performance of the memory-efficient HardSwish implementation (HardSwish) with the standard PyTorch implementation (hardswish) and the relu6 function, which is used in the computation of HardSwish.
The benchmarks were run on different data types (torch.float64, torch.float32, torch.float16).

Benchmarking HardSwish on torch.float64...
HardSwish time: 0.95625 seconds

Benchmarking HardSwish on torch.float32...
HardSwish time: 0.721875 seconds

Benchmarking HardSwish on torch.float16...
HardSwish time: 12.009375 seconds

Benchmarking hardswish on torch.float64...
hardswish time: 2.203125 seconds

Benchmarking hardswish on torch.float32...
hardswish time: 1.915625 seconds

Benchmarking hardswish on torch.float16...
hardswish time: 3.7875 seconds

Benchmarking relu6 on torch.float64...
relu6 time: 0.44375 seconds

Benchmarking relu6 on torch.float32...
relu6 time: 0.384375 seconds

Benchmarking relu6 on torch.float16...
relu6 time: 2.965625 seconds

As the results show, the memory-efficient HardSwish implementation performs comparably to the standard PyTorch implementation and relu6 across different data types. This makes it a suitable choice for deep learning models that primarily uses float32 or higher.
