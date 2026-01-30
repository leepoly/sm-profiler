# SM Profiler Library

A simplified C/CUDA library for SM-level profiling with export to Perfetto-compatible trace format.

## Overview

This library provides:
- **Device-side inline functions** for recording profiling events in CUDA kernels
- **Named event types** - create events with human-readable names
- **Automatic event ID management** - internal per-block event ID generation
- **SM-aware tracing** - each SM is a Process, each Block is a Thread in Perfetto
- **Trace export** to Chrome Trace JSON format (compatible with Perfetto UI)

## Quick Start

### 1. Include the Header

```cpp
#include "sm_profiler.h"
```

### 2. Create a Profiler Buffer (Host Side)

```cpp
// Create buffer for 4 blocks, 100 events per block
sm_profiler_buffer_t profiler = sm_profiler_create_buffer(
    /*num_blocks=*/4, 
    /*max_events_per_block=*/100
);

uint64_t* device_ptr = sm_profiler_get_device_ptr(profiler);
```

### 3. Use in Your Kernel

```cpp
__global__ void my_kernel(uint64_t* profiler_buffer, int num_blocks, int max_events) {
    // Initialize profiler (called once per kernel)
    sm_profiler_init(profiler_buffer, num_blocks, max_events);
    
    // Predicate for lane 0
    bool is_lane_0 = (threadIdx.x % 32 == 0);
    
    // Create event types with names (only lane 0, once per block)
    uint32_t evt_compute = sm_profiler_create_event(profiler_buffer, "compute", is_lane_0);
    uint32_t evt_memory = sm_profiler_create_event(profiler_buffer, "memory", is_lane_0);
    
    // Record events
    sm_profiler_event_start(profiler_buffer, evt_compute, is_lane_0);
    // ... your computation ...
    sm_profiler_event_end(profiler_buffer, evt_compute, is_lane_0);
    
    sm_profiler_event_start(profiler_buffer, evt_memory);
    // ... memory operation ...
    sm_profiler_event_end(profiler_buffer, evt_memory);
    
    // Instant event (no end needed)
    sm_profiler_event_instant(profiler_buffer, evt_memory);
}
```

### 4. Export to Trace (Host Side)

```cpp
// Export to Chrome Trace JSON
sm_profiler_export_to_file(profiler, "trace.json");

// Cleanup
sm_profiler_destroy_buffer(profiler);
```

### 5. View the Trace

Open `trace.json` in:
- Chrome: `chrome://tracing`
- Perfetto UI: https://ui.perfetto.dev/

**Visualization**: 
- Each SM (Streaming Multiprocessor) appears as a **Process**
- Each CUDA Block appears as a **Thread** within its SM's process
- Event names are formatted as: `{event_name}_{event_id}` (e.g., "compute_0", "memory_1")

## API Reference

### Host API

| Function | Description |
|----------|-------------|
| `sm_profiler_create_buffer(num_blocks, max_events_per_block)` | Create a profiler buffer |
| `sm_profiler_destroy_buffer(buffer)` | Free buffer memory |
| `sm_profiler_get_device_ptr(buffer)` | Get device pointer for kernel |
| `sm_profiler_export_to_file(buffer, filename)` | Export to JSON |
| `sm_profiler_get_info(buffer, &num_blocks, &max_events)` | Get buffer info |

### Device Inline Functions

| Function | Description |
|----------|-------------|
| `sm_profiler_init(buffer, num_blocks, max_events_per_block)` | Initialize profiler buffer |
| `sm_profiler_create_event(buffer, name, predicate)` | Create named event type, returns `event_no` |
| `sm_profiler_event_start(buffer, event_no, predicate)` | Start a range event (auto event_id) |
| `sm_profiler_event_end(buffer, event_no, predicate)` | End a range event |
| `sm_profiler_event_instant(buffer, event_no, predicate)` | Record instant event (auto event_id) |

**Notes:**
- All functions take a `predicate` parameter for flexible thread selection (typically `threadIdx.x % 32 == 0`)
- Each block can create up to `SM_PROFILER_MAX_EVENT_TYPES` (32) different event types
- Event names can be up to `SM_PROFILER_MAX_EVENT_NAME_LEN` (128) characters
- `event_no` is returned by `create_event` and used to identify event types
- `event_id` is automatically generated per block for each event instance

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
  Iterations: 3
  Max events per block: 19

Creating profiler buffer...
Launching kernel...
Kernel completed

Exporting trace...
sm_profiler: Exported 36 events (69 trace entries) to sm_profile.json
Trace saved to sm_profile.json

To view: Open sm_profile.json in Chrome chrome://tracing or https://ui.perfetto.dev/
Note: Each SM is a Process, each Block is a Thread
```

## Buffer Format

The profiler buffer stores data in a structured format:

```
[0] Header: (num_blocks, max_events_per_block)
[1 ... num_blocks] Event ID counters per block (uint32_t array)
[...] Event Type Table: [num_blocks][SM_PROFILER_MAX_EVENT_TYPES] EventTypeEntry
      Each entry: {char name[128], uint32_t active_event_id, uint32_t padding}
[...] Event Data: [num_blocks][max_events_per_block] DeviceEvent
      Each event: {event_id, event_no, block_id, sm_id, st_ts, en_ts, type, reserved}
```

## Event Type Handling

- **Range Events**: Call `event_start` followed by `event_end`. Each pair gets a unique `event_id`.
- **Instant Events**: Call `event_instant`. Gets a unique `event_id` but no duration.
- **Event Names**: Stored per-block, per-event-type in the buffer. Used during export.
- **Active Tracking**: The library tracks active range events per event type to match start/end pairs.

## Design Notes

- **Per-block event counters**: Each block maintains its own event ID counter using atomic operations
- **Automatic lane selection**: Only lane 0 (threadIdx.x % 32 == 0) writes events
- **SM tracking**: Events record the SM ID (`%smid`) for process grouping in the trace
- **Event naming**: Event names are set at creation time and exported as `{name}_{event_id}`
