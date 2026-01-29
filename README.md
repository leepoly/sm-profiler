# SM Profiler Library

A simplified C/CUDA library for SM-level profiling with export to Perfetto-compatible trace format.

## Overview

This library provides:
- **Device-side macros** for recording profiling events in CUDA kernels
- **Buffer management** for efficient GPU-side event storage
- **Trace export** to Chrome Trace JSON format (compatible with Perfetto UI)

## Quick Start

### 1. Include the Header

```cpp
#include "sm_profiler.h"
```

### 2. Create a Profiler Buffer (Host Side)

```cpp
// Create buffer for 4 blocks, 4 warps per block, 100 events per warp
sm_profiler_buffer_t profiler = sm_profiler_create_buffer(
    /*num_blocks=*/4, 
    /*num_groups=*/4,   // e.g., warps per block
    /*max_events_per_group=*/100
);

uint64_t* device_ptr = sm_profiler_get_device_ptr(profiler);
```

### 3. Use in Your Kernel

```cpp
__global__ void my_kernel(uint64_t* profiler_buffer) {
    // Each warp (32 threads) is a group, only thread 0 writes
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    
    // Declare profiler variables
    SM_PROFILER_DECL_VARS;
    
    // Initialize profiler
    SM_PROFILER_INIT(profiler_buffer, warp_id, num_warps, threadIdx.x == 0);
    
    // Record events
    SM_PROFILER_EVENT_START(0, 0);  // event_id=0, event_no=0
    
    // ... your code ...
    
    SM_PROFILER_EVENT_END(0, 0);
}
```

### 4. Export to Trace (Host Side)

```cpp
// Define event names (optional)
const char* event_names[] = {
    "compute",   // event_id = 0
    "memory",    // event_id = 1
};

// Export to Chrome Trace JSON
sm_profiler_export_to_file(profiler, "trace.json", event_names);

// Cleanup
sm_profiler_destroy_buffer(profiler);
```

### 5. View the Trace

Open `trace.json` in:
- Chrome: `chrome://tracing`
- Perfetto UI: https://ui.perfetto.dev/

## API Reference

### Host API

| Function | Description |
|----------|-------------|
| `sm_profiler_create_buffer(num_blocks, num_groups, max_events)` | Create a profiler buffer |
| `sm_profiler_destroy_buffer(buffer)` | Free buffer memory |
| `sm_profiler_get_device_ptr(buffer)` | Get device pointer for kernel |
| `sm_profiler_export_to_file(buffer, filename, event_names)` | Export to JSON |

### Device Macros

| Macro | Description |
|-------|-------------|
| `SM_PROFILER_DECL_VARS` | Declare required variables in kernel |
| `SM_PROFILER_INIT(buf, group_idx, num_groups, predicate)` | Initialize profiler |
| `SM_PROFILER_EVENT_START(event_id, event_no)` | Record event start |
| `SM_PROFILER_EVENT_END(event_id, event_no)` | Record event end |
| `SM_PROFILER_EVENT_INSTANT(event_id, event_no)` | Record instant event |

## Building

```bash
make        # Build demo
make run    # Build and run demo
```

## Example Output

```
Device: NVIDIA A100

Configuration:
  Blocks: 4
  Threads per block: 128
  Warps per block (profiling groups): 4
  Iterations: 3
  Max events per warp: 18

Creating profiler buffer...
Launching kernel...
Kernel completed

Exporting trace...
sm_profiler: Exported 144 events to sm_profile.json
Trace saved to sm_profile.json
```

## Buffer Format

The profiler buffer stores events in a structured format:

```
[0] Header: (num_blocks, num_groups) as two uint32s packed in uint64
[1...N] Events: (tag, timestamp) pairs
```

Tag encoding (32 bits):
```
[31-19] [18-11]      [10-2]       [1-0]
[seq]   [block_grp]  [event_id]   [type:0=begin,1=end,2=instant]
```

## Comparison with Original

| Feature | Original (profiler.h) | SM Profiler Library |
|---------|----------------------|---------------------|
| Buffer creation | Manual | `sm_profiler_create_buffer()` |
| Buffer format | Same structure | Same structure |
| Device macros | `PROFILER_INIT`, etc. | `SM_PROFILER_INIT`, etc. |
| Export | Python + tg4perfetto | C/C++ native (JSON) |
| Dependencies | Python, protobuf | CUDA runtime only |
