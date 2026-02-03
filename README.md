# SM Profiler Library

A simplified C/CUDA and Python library for SM-level profiling with export to Perfetto-compatible trace format.

## Overview

This library provides:
- **Zero-sync profiling** - no atomics needed, each warp has its own counter
- **Simple API** - just call `sm_profiler_event_start/end/instant`
- **Host-side setup** - register event type names and initialize buffer before kernel launch
- **SM-aware tracing** - each SM is a Process, each Block+Group is a Thread in Perfetto
- **Trace export** to Chrome Trace JSON format (compatible with Perfetto UI)
- **Dual implementations** - native CUDA C++ and Python/Triton

## Quick Start - CUDA

### 1. Include the Header

```cpp
#include "sm_profiler.h"
```

### 2. Define Event Type Constants

```cpp
#define EVT_COMPUTE  0
#define EVT_MEMORY   1
#define EVT_SYNC     2
```

### 3. Create and Setup Profiler Buffer (Host Side)

```cpp
sm_profiler_buffer_t profiler = sm_profiler_create_buffer(
    /*num_blocks=*/4, 
    /*num_groups=*/4,             // warps per block
    /*max_events_per_group=*/100, // per warp
    /*enabled=*/1
);

sm_profiler_register_event(profiler, EVT_COMPUTE, "compute");
sm_profiler_register_event(profiler, EVT_MEMORY, "memory");
sm_profiler_register_event(profiler, EVT_SYNC, "sync");

sm_profiler_init_buffer(profiler);

uint64_t* device_ptr = sm_profiler_get_device_ptr(profiler);
```

### 4. Use in Your Kernel

```cpp
__global__ void my_kernel(uint64_t* profiler_buffer) {
    bool is_lane_0 = (sm_profiler_get_lane_id() == 0);
    
    sm_profiler_event_start(profiler_buffer, EVT_COMPUTE, is_lane_0);
    // ... your computation ...
    sm_profiler_event_end(profiler_buffer, EVT_COMPUTE, is_lane_0);
    
    sm_profiler_event_instant(profiler_buffer, EVT_SYNC, is_lane_0);
}
```

### 5. Export to Trace (Host Side)

```cpp
sm_profiler_export_to_file(profiler, "trace.json");
sm_profiler_destroy_buffer(profiler);
```

## Quick Start - Python/Triton

### 1. Install the Package

```bash
cd python/
pip install -e .
```

### 2. Define Event Type Constants

```python
import triton.language as tl

# Use tl.constexpr for kernel-side constants
EVT_COMPUTE = tl.constexpr(0)
EVT_MEMORY = tl.constexpr(1)
EVT_SYNC = tl.constexpr(2)
```

### 3. Create and Setup Profiler (Host Side)

```python
from sm_profiler import SmProfiler

profiler = SmProfiler(
    num_blocks=16,
    num_groups=1,
    max_events_per_group=100,
    enabled=True
)

profiler.register_event(EVT_COMPUTE, "compute")
profiler.register_event(EVT_MEMORY, "memory")
profiler.register_event(EVT_SYNC, "sync")

buffer = profiler.get_buffer()
```

### 4. Use in Your Triton Kernel

```python
from sm_profiler.triton_ops import (
    profiler_init,
    profiler_event_start,
    profiler_event_end,
    profiler_event_instant,
)

@triton.jit
def my_kernel(profiler_buffer_ptr, ...):
    ctx = profiler_init(profiler_buffer_ptr)
    block_idx = tl.program_id(0)
    group_idx = 0  # or compute based on your kernel structure
    
    profiler_event_start(ctx, block_idx, group_idx, EVT_COMPUTE)
    # ... your computation ...
    profiler_event_end(ctx, block_idx, group_idx, EVT_COMPUTE)
    
    profiler_event_instant(ctx, block_idx, group_idx, EVT_SYNC)
```

### 5. Export to Trace (Host Side)

```python
torch.cuda.synchronize()
profiler.export("trace.json")
profiler.print_stats()  # Optional: print profiling statistics
```

## Viewing Traces

Open generated JSON files in:
- Chrome: `chrome://tracing`
- Perfetto UI: https://ui.perfetto.dev/

## API Reference

### CUDA Host API

| Function | Description |
|----------|-------------|
| `sm_profiler_create_buffer(num_blocks, num_groups, max_events_per_group, enabled)` | Create a profiler buffer |
| `sm_profiler_register_event(buffer, event_no, name)` | Register event name (0-31) |
| `sm_profiler_init_buffer(buffer)` | Initialize/clear buffer before kernel launch |
| `sm_profiler_get_device_ptr(buffer)` | Get device pointer for kernel |
| `sm_profiler_export_to_file(buffer, filename)` | Export to JSON |
| `sm_profiler_destroy_buffer(buffer)` | Free buffer memory |

### CUDA Device API

```cpp
sm_profiler_event_start(buffer, event_no, predicate);   // Start range event
sm_profiler_event_end(buffer, event_no, predicate);     // End range event
sm_profiler_event_instant(buffer, event_no, predicate); // Instant event
```

### CUDA Helper Functions (Device)

```cpp
sm_profiler_get_lane_id()       // Lane ID within warp (0-31)
sm_profiler_get_warp_id()       // Warp ID within block (works for 1D/2D/3D blocks)
sm_profiler_get_block_idx()     // Linear block index
sm_profiler_get_smid()          // SM ID
sm_profiler_get_timestamp()     // Global timer (nanoseconds)
```

### Python Host API

| Method | Description |
|--------|-------------|
| `SmProfiler(num_blocks, num_groups, max_events_per_group, enabled=True)` | Create profiler |
| `profiler.register_event(event_no, name)` | Register event name |
| `profiler.init_buffer()` | Re-initialize buffer (called automatically on creation) |
| `profiler.get_buffer()` | Get buffer tensor for kernel |
| `profiler.export(filename)` | Export to JSON |
| `profiler.print_stats()` | Print profiling statistics |

### Triton Device API

```python
ctx = profiler_init(buffer_ptr)                                  # Initialize context
profiler_event_start(ctx, block_idx, group_idx, event_no)        # Start range event
profiler_event_end(ctx, block_idx, group_idx, event_no)          # End range event
profiler_event_instant(ctx, block_idx, group_idx, event_no)      # Instant event
```

## API Design Differences

The CUDA and Triton APIs differ intentionally:

| Aspect | CUDA | Triton |
|--------|------|--------|
| Block/Group indexing | Auto-computed internally | Explicitly passed by user |
| Thread control | Uses `predicate` (e.g., `is_lane_0`) | Each program is independent |
| Context | Pass buffer pointer directly | Call `profiler_init()` to get context |

**Why?**
- CUDA: Has fine-grained thread/warp control via inline assembly
- Triton: No warp/lane concept; each "program" processes a tile, user decides the mapping

## Building

### CUDA

```bash
cd cuda/
make clean && make && ./sm_profiler_demo
```

### Python

```bash
cd python/
pip install -e .
python demo.py
```

## Design Notes

- **Zero-sync**: Each (block, warp) has its own counter - no atomics needed
- **2D/3D block support**: `sm_profiler_get_warp_id()` works correctly for all block dimensions
- **Per-launch init**: Call `sm_profiler_init_buffer()` / `profiler.init_buffer()` before each kernel
- **Event IDs**: Must be 0-31
