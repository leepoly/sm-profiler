/*
 * SM-level C Profiler Library
 * 
 * A simplified library for SM-level profiling in CUDA kernels.
 * Optimized for minimal overhead - no atomics needed.
 */

#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to profiler buffer */
typedef struct sm_profiler_buffer* sm_profiler_buffer_t;

/* Configuration constants */
#define SM_PROFILER_MAX_EVENT_TYPES 32
#define SM_PROFILER_MAX_EVENT_NAME_LEN 128

/*
 * Create a profiler buffer
 * 
 * @param num_blocks: Number of CUDA blocks expected
 * @param num_groups: Number of groups (warps) per block
 * @param max_events_per_group: Maximum events per group (warp)
 * @return: Handle to profiler buffer, or NULL on error
 */
sm_profiler_buffer_t sm_profiler_create_buffer(
    uint32_t num_blocks,
    uint32_t num_groups,
    uint32_t max_events_per_group,
    int enabled
);

/*
 * Destroy a profiler buffer and free associated memory
 */
void sm_profiler_destroy_buffer(sm_profiler_buffer_t buffer);

/*
 * Get the raw device pointer for use in kernels
 */
uint64_t* sm_profiler_get_device_ptr(sm_profiler_buffer_t buffer);

/*
 * Initialize buffer on host side (call before kernel launch)
 * Clears event counters and event data, preserves event type names
 */
void sm_profiler_init_buffer(sm_profiler_buffer_t buffer);

/*
 * Register an event type name (call once during setup)
 * The event_no is the user-defined ID (0 to SM_PROFILER_MAX_EVENT_TYPES-1)
 * This name will be used for all blocks and groups
 */
int sm_profiler_register_event(
    sm_profiler_buffer_t buffer,
    uint32_t event_no,
    const char* name
);

/*
 * Export profiler buffer to Chrome Trace JSON format (compatible with Perfetto UI)
 * 
 * @param buffer: Profiler buffer handle
 * @param filename: Output JSON file path
 * @return: 0 on success, non-zero on error
 */
int sm_profiler_export_to_file(
    sm_profiler_buffer_t buffer,
    const char* filename
);

/*
 * Get buffer info
 */
void sm_profiler_get_info(
    sm_profiler_buffer_t buffer,
    uint32_t* out_num_blocks,
    uint32_t* out_num_groups,
    uint32_t* out_max_events
);

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * Device-side inline functions for use in CUDA kernels
 * ============================================================================ */

#ifdef __CUDACC__

/* Event type constants */
#define SM_PROFILER_EVENT_TYPE_RANGE 0
#define SM_PROFILER_EVENT_TYPE_INSTANT 1

/* Active event tracking (per block, per group, per event type) */
struct SmProfilerActiveEventEntry {
    uint32_t active_event_id;  /* Current active range event id (0 if none) */
};

/* Device event structure - 40 bytes, packed to ensure consistent layout */
struct __attribute__((packed)) SmProfilerDeviceEvent {
    uint32_t event_id;      /* Unique id within group (auto-increment) */
    uint32_t event_no;      /* User-created event number (event type) */
    uint32_t block_id;
    uint32_t group_idx;     /* Group (warp) index within block */
    uint32_t sm_id;         /* SM where this event was recorded */
    uint32_t type;          /* 0=range, 1=instant */
    uint64_t st_timestamp_ns;
    uint64_t en_timestamp_ns;
};

/* ============================================================================
 * Helper functions
 * ============================================================================ */

__device__ __forceinline__ uint32_t sm_profiler_get_block_idx() {
    return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
}

__device__ __forceinline__ uint32_t sm_profiler_get_thread_idx_in_block() {
    return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ uint32_t sm_profiler_get_lane_id() {
    uint32_t lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__ __forceinline__ uint32_t sm_profiler_get_warp_id() {
    return sm_profiler_get_thread_idx_in_block() / 32;
}

__device__ __forceinline__ uint32_t sm_profiler_get_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ __forceinline__ uint64_t sm_profiler_get_timestamp() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));
    return ret;
}

/* ============================================================================
 * Internal buffer parsing (inlined for each call)
 * ============================================================================ */

struct __SmProfilerBufferInfo {
    uint32_t num_blocks;
    uint32_t num_groups;
    uint32_t max_events_per_group;
    uint32_t enabled;
    uint32_t* counters;
    char* active_table_base;
    char* event_data_base;
};

__device__ __forceinline__ __SmProfilerBufferInfo __sm_profiler_parse_buffer(uint64_t* buffer) {
    __SmProfilerBufferInfo info;
    
    /* Parse header */
    uint64_t header0 = buffer[0];
    uint64_t header1 = buffer[1];
    info.num_blocks = (uint32_t)(header0 >> 32);
    info.num_groups = (uint32_t)(header0 & 0xFFFFFFFF);
    info.max_events_per_group = (uint32_t)(header1 >> 32);
    info.enabled = (uint32_t)(header1 & 0xFFFFFFFF);
    
    /* Counters start after header */
    info.counters = (uint32_t*)(buffer + 2);
    
    /* Calculate offsets */
    size_t counters_bytes = (size_t)info.num_blocks * info.num_groups * sizeof(uint32_t);
    uint32_t counters_size_uint64 = (uint32_t)((counters_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    uint32_t active_offset = 2 + counters_size_uint64;
    
    size_t active_bytes = (size_t)info.num_blocks * info.num_groups * SM_PROFILER_MAX_EVENT_TYPES * sizeof(SmProfilerActiveEventEntry);
    uint32_t active_size_uint64 = (uint32_t)((active_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    uint32_t event_offset = active_offset + active_size_uint64;
    
    info.active_table_base = (char*)(buffer + active_offset);
    info.event_data_base = (char*)(buffer + event_offset);
    
    return info;
}

__device__ __forceinline__ SmProfilerActiveEventEntry* __sm_profiler_get_active_entry(
    __SmProfilerBufferInfo& info, uint32_t block_idx, uint32_t group_idx, uint32_t event_no) {
    size_t block_stride = (size_t)info.num_groups * SM_PROFILER_MAX_EVENT_TYPES * sizeof(SmProfilerActiveEventEntry);
    size_t group_stride = SM_PROFILER_MAX_EVENT_TYPES * sizeof(SmProfilerActiveEventEntry);
    return (SmProfilerActiveEventEntry*)(info.active_table_base + block_idx * block_stride + group_idx * group_stride + event_no * sizeof(SmProfilerActiveEventEntry));
}

__device__ __forceinline__ SmProfilerDeviceEvent* __sm_profiler_get_event(
    __SmProfilerBufferInfo& info, uint32_t block_idx, uint32_t group_idx, uint32_t event_idx) {
    size_t group_offset = ((size_t)block_idx * info.num_groups + group_idx) * info.max_events_per_group;
    return (SmProfilerDeviceEvent*)(info.event_data_base + (group_offset + event_idx) * sizeof(SmProfilerDeviceEvent));
}

/* ============================================================================
 * User-facing functions
 * ============================================================================ */

__device__ __forceinline__ void sm_profiler_event_start(
    uint64_t* buffer,
    uint32_t event_no,
    bool predicate)
{
    if (!predicate) return;
    if (event_no >= SM_PROFILER_MAX_EVENT_TYPES) return;
    
    __SmProfilerBufferInfo info = __sm_profiler_parse_buffer(buffer);
    if (!info.enabled) return;
    
    uint32_t block_idx = sm_profiler_get_block_idx();
    uint32_t group_idx = sm_profiler_get_warp_id();
    
    /* Get active event tracking for this group */
    SmProfilerActiveEventEntry* entry = __sm_profiler_get_active_entry(info, block_idx, group_idx, event_no);
    
    /* Check if already active */
    if (entry->active_event_id != 0) return;
    
    /* Allocate new event id - NO ATOMICS! Each (block, group) has its own counter */
    uint32_t* counter = &info.counters[block_idx * info.num_groups + group_idx];
    uint32_t event_id = *counter;
    *counter = event_id + 1;
    
    /* Check buffer full */
    if (event_id >= info.max_events_per_group) return;
    
    /* Record event */
    SmProfilerDeviceEvent* ev = __sm_profiler_get_event(info, block_idx, group_idx, event_id);
    ev->event_id = event_id;
    ev->event_no = event_no;
    ev->block_id = block_idx;
    ev->group_idx = group_idx;
    ev->sm_id = sm_profiler_get_smid();
    ev->type = SM_PROFILER_EVENT_TYPE_RANGE;
    ev->st_timestamp_ns = sm_profiler_get_timestamp();
    ev->en_timestamp_ns = 0;
    
    /* Mark as active (+1 to distinguish from 0) */
    entry->active_event_id = event_id + 1;
}

__device__ __forceinline__ void sm_profiler_event_end(
    uint64_t* buffer,
    uint32_t event_no,
    bool predicate)
{
    if (!predicate) return;
    if (event_no >= SM_PROFILER_MAX_EVENT_TYPES) return;
    
    __SmProfilerBufferInfo info = __sm_profiler_parse_buffer(buffer);
    if (!info.enabled) return;
    
    uint32_t block_idx = sm_profiler_get_block_idx();
    uint32_t group_idx = sm_profiler_get_warp_id();
    
    /* Get active event tracking for this group */
    SmProfilerActiveEventEntry* entry = __sm_profiler_get_active_entry(info, block_idx, group_idx, event_no);
    
    /* Check if active */
    if (entry->active_event_id == 0) return;
    
    uint32_t event_id = entry->active_event_id - 1;
    
    /* Record end timestamp */
    SmProfilerDeviceEvent* ev = __sm_profiler_get_event(info, block_idx, group_idx, event_id);
    ev->en_timestamp_ns = sm_profiler_get_timestamp();
    
    /* Mark as inactive */
    entry->active_event_id = 0;
}

__device__ __forceinline__ void sm_profiler_event_instant(
    uint64_t* buffer,
    uint32_t event_no,
    bool predicate)
{
    if (!predicate) return;
    if (event_no >= SM_PROFILER_MAX_EVENT_TYPES) return;
    
    __SmProfilerBufferInfo info = __sm_profiler_parse_buffer(buffer);
    if (!info.enabled) return;
    
    uint32_t block_idx = sm_profiler_get_block_idx();
    uint32_t group_idx = sm_profiler_get_warp_id();
    
    /* Allocate new event id - NO ATOMICS! */
    uint32_t* counter = &info.counters[block_idx * info.num_groups + group_idx];
    uint32_t event_id = *counter;
    *counter = event_id + 1;
    
    /* Check buffer full */
    if (event_id >= info.max_events_per_group) return;
    
    /* Record event */
    SmProfilerDeviceEvent* ev = __sm_profiler_get_event(info, block_idx, group_idx, event_id);
    ev->event_id = event_id;
    ev->event_no = event_no;
    ev->block_id = block_idx;
    ev->group_idx = group_idx;
    ev->sm_id = sm_profiler_get_smid();
    ev->type = SM_PROFILER_EVENT_TYPE_INSTANT;
    ev->st_timestamp_ns = sm_profiler_get_timestamp();
    ev->en_timestamp_ns = 0;
}

#endif /* __CUDACC__ */
