# SM Profiler Library

A simplified C/CUDA library for SM-level profiling with export to Perfetto-compatible trace format.

## Overview

This library provides:
- **Zero-sync profiling** - no atomics needed, each warp has its own counter
- **Simple API** - just call `sm_profiler_event_start/end/instant`
- **Host-side setup** - register event type names and initialize buffer before kernel launch
- **SM-aware tracing** - each SM is a Process, each Block+Group is a Thread in Perfetto
- **Trace export** to Chrome Trace JSON format (compatible with Perfetto UI)

## Quick Start

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
    /*max_events_per_group=*/100  // per warp
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

### 6. View the Trace

Open `trace.json` in:
- Chrome: `chrome://tracing`
- Perfetto UI: https://ui.perfetto.dev/

## API Reference

### Host API

| Function | Description |
|----------|-------------|
| `sm_profiler_create_buffer(num_blocks, num_groups, max_events_per_group)` | Create a profiler buffer |
| `sm_profiler_register_event(buffer, event_no, name)` | Register event name (0-31) |
| `sm_profiler_init_buffer(buffer)` | Initialize/clear buffer before kernel launch |
| `sm_profiler_get_device_ptr(buffer)` | Get device pointer for kernel |
| `sm_profiler_export_to_file(buffer, filename)` | Export to JSON |
| `sm_profiler_destroy_buffer(buffer)` | Free buffer memory |

### Device API

```cpp
sm_profiler_event_start(buffer, event_no, predicate);   // Start range event
sm_profiler_event_end(buffer, event_no, predicate);     // End range event
sm_profiler_event_instant(buffer, event_no, predicate); // Instant event
```

### Helper Functions (Device)

```cpp
sm_profiler_get_lane_id()       // Lane ID within warp (0-31)
sm_profiler_get_warp_id()       // Warp ID within block (works for 1D/2D/3D blocks)
sm_profiler_get_block_idx()     // Linear block index
sm_profiler_get_smid()          // SM ID
sm_profiler_get_timestamp()     // Global timer (nanoseconds)
```

## Building

```bash
make clean && make && ./sm_profiler_demo
```

## Design Notes

- **Zero-sync**: Each (block, warp) has its own counter - no atomics needed
- **2D/3D block support**: `sm_profiler_get_warp_id()` works correctly for all block dimensions
- **Per-launch init**: Call `sm_profiler_init_buffer()` before each kernel
