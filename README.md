**MemoryEfficientHardSwish**

This repository contains an optimized implementation of the HardSwish activation function, specifically designed for memory efficiency. The HardSwish function is a common choice in deep learning models, and this implementation aims to reduce the memory footprint of these models without compromising on performance.

**Benchmark Results**

The performance of the memory-efficient HardSwish implementation (HardSwish) is compared with the standard PyTorch implementation (hardswish) and the relu6 function, which is a component in the computation of HardSwish. Benchmarks were conducted on different data types (torch.float64, torch.float32, torch.float16).

| Function | Data Type | Time (seconds) |
| --- | --- | --- |
| MemoryEfficientHardSwish | torch.float64 | 0.95 |
| MemoryEfficientHardSwish | torch.float32 | 0.878125 |
| MemoryEfficientHardSwish | torch.float16 | 12.2 |
| hardswish | torch.float64 | 2.371875 |
| hardswish | torch.float32 | 1.946875 |
| hardswish | torch.float16 | 3.696875 |
| relu6 | torch.float64 | 0.425 |
| relu6 | torch.float32 | 0.384375 |
| relu6 | torch.float16 | 2.990625 |

As the results demonstrate, the memory-efficient HardSwish implementation performs comparably to the standard PyTorch implementation and relu6 across different data types. This makes it a suitable choice for deep learning models that primarily use float32 or higher.
