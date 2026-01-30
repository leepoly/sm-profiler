# SM Profiler for Triton

A simple profiling library for Triton kernels with Chrome Trace export.

## Installation

```bash
cd python
pip install -e .
```

## Quick Start

### 1. Create Profiler (Host Side)

```python
from sm_profiler import SmProfiler

profiler = SmProfiler(
    num_blocks=16,
    num_groups=1,  # groups per block (usually 1 for simple kernels)
    max_events_per_group=100
)

# Register event names
profiler.register_event(0, "compute")
profiler.register_event(1, "memory")

# Get buffer to pass to kernel
buffer = profiler.get_buffer()
```

### 2. Use in Triton Kernel

```python
import triton
import triton.language as tl
from sm_profiler.triton_ops import (
    profiler_parse_buffer,
    profiler_event_start,
    profiler_event_end,
    profiler_event_instant,
)

EVT_COMPUTE = 0
EVT_MEMORY = 1

@triton.jit
def my_kernel(profiler_buffer_ptr, ...):
    # Parse buffer once at kernel start
    num_blocks, num_groups, max_events, counters_ptr, active_ptr, events_ptr = \
        profiler_parse_buffer(profiler_buffer_ptr)
    
    block_idx = tl.program_id(0)
    group_idx = 0  # or compute based on your kernel structure
    
    # Record events
    profiler_event_start(counters_ptr, active_ptr, events_ptr,
                        num_groups, max_events, block_idx, group_idx, EVT_COMPUTE)
    
    # ... your computation ...
    
    profiler_event_end(active_ptr, events_ptr,
                      num_groups, max_events, block_idx, group_idx, EVT_COMPUTE)
```

### 3. Export Trace (Host Side)

```python
# After kernel execution
torch.cuda.synchronize()

profiler.export("trace.json")
```

### 4. View Trace

Open `trace.json` in:
- Chrome: `chrome://tracing`
- Perfetto UI: https://ui.perfetto.dev/

## API Reference

### Host API (Python)

| Method | Description |
|--------|-------------|
| `SmProfiler(num_blocks, num_groups, max_events_per_group)` | Create profiler |
| `profiler.register_event(event_no, name)` | Register event name |
| `profiler.get_buffer()` | Get buffer tensor for kernel |
| `profiler.init_buffer()` | Reset buffer (auto-called on creation) |
| `profiler.export(filename)` | Export to Chrome Trace JSON |
| `profiler.print_stats()` | Print profiling statistics |

### Device API (Triton)

```python
from sm_profiler.triton_ops import (
    profiler_parse_buffer,      # Parse buffer header
    profiler_event_start,       # Start range event
    profiler_event_end,         # End range event
    profiler_event_instant,     # Record instant event
)
```

## Demo

```bash
cd python
pip install -e .
python demo.py
```
