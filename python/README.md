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
    profiler_init,
    profiler_event_start,
    profiler_event_end,
    profiler_event_instant,
)

# Define event type constants (both regular and tl.constexpr versions)
EVT_COMPUTE = 0
EVT_MEMORY = 1
TL_EVT_COMPUTE = tl.constexpr(0)
TL_EVT_MEMORY = tl.constexpr(1)

@triton.jit
def my_kernel(
    profiler_buffer_ptr,
    ...,
    NUM_GROUPS: tl.constexpr,
):
    # Initialize profiler context once at kernel start
    ctx = profiler_init(profiler_buffer_ptr)
    
    block_idx = tl.program_id(0)
    group_idx = 0  # or compute based on your kernel structure
    
    # Record range events (start/end)
    profiler_event_start(ctx, block_idx, group_idx, TL_EVT_COMPUTE)
    
    # ... your computation ...
    
    profiler_event_end(ctx, block_idx, group_idx, TL_EVT_COMPUTE)
    
    # Record instant events
    profiler_event_instant(ctx, block_idx, group_idx, TL_EVT_SYNC)
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
    profiler_init,              # Initialize profiler context
    profiler_event_start,       # Start range event
    profiler_event_end,         # End range event
    profiler_event_instant,     # Record instant event
)
```

**Usage:**
- `ctx = profiler_init(profiler_buffer_ptr)` - Initialize once at kernel start, returns context object
- `profiler_event_start(ctx, block_idx, group_idx, event_type)` - Start a range event
- `profiler_event_end(ctx, block_idx, group_idx, event_type)` - End a range event
- `profiler_event_instant(ctx, block_idx, group_idx, event_type)` - Record an instant event

**Note:** Event types should be `tl.constexpr` values when used in kernels (e.g., `TL_EVT_COMPUTE = tl.constexpr(0)`).

## Demo

```bash
cd python
pip install -e .
python demo.py
```
