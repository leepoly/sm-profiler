/*
 * SM-level C Profiler Library
 * 
 * A simplified library for SM-level profiling in CUDA kernels.
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
 * @param max_events_per_block: Maximum events per block
 * @return: Handle to profiler buffer, or NULL on error
 */
sm_profiler_buffer_t sm_profiler_create_buffer(
    uint32_t num_blocks,
    uint32_t max_events_per_block
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
    uint32_t* out_max_events
);

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * Device-side inline functions for use in CUDA kernels
 * ============================================================================ */

#ifdef __CUDA_ARCH__

namespace sm_profiler {

__device__ __forceinline__ uint32_t get_block_idx() {
    return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
}

__device__ __forceinline__ uint32_t get_num_blocks() {
    return gridDim.x * gridDim.y * gridDim.z;
}

__device__ __forceinline__ uint32_t get_thread_idx() {
    return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ uint32_t get_lane_id() {
    return threadIdx.x % 32;
}

__device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ __forceinline__ uint64_t get_timestamp() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));
    return ret;
}

/* Event type constants */
constexpr uint32_t EVENT_TYPE_RANGE = 0;
constexpr uint32_t EVENT_TYPE_INSTANT = 1;

/* Event type entry (per block) */
struct EventTypeEntry {
    char name[SM_PROFILER_MAX_EVENT_NAME_LEN];
    uint32_t active_event_id;  // Current active range event id (0 if none)
    uint32_t padding;
};

/* Device event structure (matches trace_event_t) */
struct DeviceEvent {
    uint32_t event_id;      // Unique id within block (auto-increment)
    uint32_t event_no;      // User-created event number (event type)
    uint32_t block_id;
    uint32_t sm_id;         // SM where this event was recorded
    uint64_t st_timestamp_ns;
    uint64_t en_timestamp_ns;
    uint32_t type;          // 0=range, 1=instant
    uint32_t reserved;
};

/* Buffer layout manager */
struct ProfilerBuffer {
    uint64_t* data;
    uint32_t num_blocks;
    uint32_t max_events_per_block;
    uint32_t event_type_table_offset;  // In uint64_t units
    uint32_t event_data_offset;        // In uint64_t units
    
    __device__ __forceinline__ static ProfilerBuffer from_ptr(uint64_t* ptr) {
        ProfilerBuffer buf;
        buf.data = ptr;
        uint64_t header = ptr[0];
        buf.num_blocks = (uint32_t)(header >> 32);
        buf.max_events_per_block = (uint32_t)(header & 0xFFFFFFFF);
        
        // Calculate offsets
        // [0] header (1 uint64)
        // [1 .. num_blocks] event id counters
        buf.event_type_table_offset = 1 + buf.num_blocks;
        // Event type table: num_blocks * SM_PROFILER_MAX_EVENT_TYPES * sizeof(EventTypeEntry)
        size_t type_table_size = buf.num_blocks * SM_PROFILER_MAX_EVENT_TYPES * sizeof(EventTypeEntry);
        buf.event_data_offset = buf.event_type_table_offset + (type_table_size + sizeof(uint64_t) - 1) / sizeof(uint64_t);
        
        return buf;
    }
    
    __device__ __forceinline__ uint32_t* get_event_id_counters() {
        return (uint32_t*)(data + 1);
    }
    
    __device__ __forceinline__ EventTypeEntry* get_event_type_table(uint32_t block_idx) {
        char* base = (char*)(data + event_type_table_offset);
        return (EventTypeEntry*)(base + block_idx * SM_PROFILER_MAX_EVENT_TYPES * sizeof(EventTypeEntry));
    }
    
    __device__ __forceinline__ DeviceEvent* get_event_data(uint32_t block_idx, uint32_t event_idx) {
        char* base = (char*)(data + event_data_offset);
        return (DeviceEvent*)(base + (block_idx * max_events_per_block + event_idx) * sizeof(DeviceEvent));
    }
};

} // namespace sm_profiler

/* ============================================================================
 * User-facing inline functions
 * ============================================================================ */

/* Initialize profiler buffer (call once per kernel, before any events) */
__device__ __forceinline__ void sm_profiler_init(uint64_t* buffer,
                                                  uint32_t num_blocks,
                                                  uint32_t max_events_per_block) {
    using namespace sm_profiler;
    
    uint32_t block_idx = get_block_idx();
    uint32_t thread_idx = get_thread_idx();
    
    // Thread 0 of block 0 writes header
    if (block_idx == 0 && thread_idx == 0) {
        buffer[0] = ((uint64_t)num_blocks << 32) | (uint64_t)max_events_per_block;
        
        // Clear event id counters
        uint32_t* counters = (uint32_t*)(buffer + 1);
        for (uint32_t i = 0; i < num_blocks; i++) {
            counters[i] = 0;
        }
        
        // Clear event type table
        ProfilerBuffer buf = ProfilerBuffer::from_ptr(buffer);
        for (uint32_t b = 0; b < num_blocks; b++) {
            EventTypeEntry* table = buf.get_event_type_table(b);
            for (uint32_t e = 0; e < SM_PROFILER_MAX_EVENT_TYPES; e++) {
                table[e].name[0] = '\0';
                table[e].active_event_id = 0;
            }
        }
    }
    
    __threadfence();
    __syncthreads();
}

/* 
 * Create an event type with a name
 * Returns event_no (0 to SM_PROFILER_MAX_EVENT_TYPES-1), or -1 if failed
 * 
 * @param predicate: Only execute if true (typically: threadIdx.x % 32 == 0)
 */
__device__ __forceinline__ uint32_t sm_profiler_create_event(uint64_t* buffer,
                                                                const char* name,
                                                                bool predicate) {
    using namespace sm_profiler;
    
    if (!predicate) return (uint32_t)-1;
    
    uint32_t block_idx = get_block_idx();
    ProfilerBuffer buf = ProfilerBuffer::from_ptr(buffer);
    EventTypeEntry* table = buf.get_event_type_table(block_idx);
    
    // Find first empty slot
    for (uint32_t i = 0; i < SM_PROFILER_MAX_EVENT_TYPES; i++) {
        if (table[i].name[0] == '\0') {
            // Copy name
            for (int j = 0; j < SM_PROFILER_MAX_EVENT_NAME_LEN - 1 && name[j] != '\0'; j++) {
                table[i].name[j] = name[j];
                table[i].name[j + 1] = '\0';
            }
            table[i].active_event_id = 0;
            return i;
        }
    }
    
    return (uint32_t)-1;  // No slot available
}

/* Record range event start
 * 
 * @param predicate: Only execute if true (typically: threadIdx.x % 32 == 0)
 */
__device__ __forceinline__ void sm_profiler_event_start(uint64_t* buffer,
                                                         uint32_t event_no,
                                                         bool predicate) {
    using namespace sm_profiler;
    
    if (!predicate) return;
    if (event_no >= SM_PROFILER_MAX_EVENT_TYPES) return;
    
    uint32_t block_idx = get_block_idx();
    ProfilerBuffer buf = ProfilerBuffer::from_ptr(buffer);
    
    // Get event type entry
    EventTypeEntry* table = buf.get_event_type_table(block_idx);
    EventTypeEntry* entry = &table[event_no];
    
    // Check if already active
    if (entry->active_event_id != 0) return;  // Already have an active event
    
    // Allocate new event id
    uint32_t* counters = buf.get_event_id_counters();
    uint32_t event_id = atomicAdd(&counters[block_idx], 1);
    
    if (event_id >= buf.max_events_per_block) return;  // Buffer full
    
    // Record event
    DeviceEvent* ev = buf.get_event_data(block_idx, event_id);
    ev->event_id = event_id;
    ev->event_no = event_no;
    ev->block_id = block_idx;
    ev->sm_id = get_smid();
    ev->st_timestamp_ns = get_timestamp();
    ev->en_timestamp_ns = 0;
    ev->type = EVENT_TYPE_RANGE;
    ev->reserved = 0;
    
    // Mark as active
    entry->active_event_id = event_id + 1;  // +1 to distinguish from 0
}

/* Record range event end
 * 
 * @param predicate: Only execute if true (typically: threadIdx.x % 32 == 0)
 */
__device__ __forceinline__ void sm_profiler_event_end(uint64_t* buffer,
                                                       uint32_t event_no,
                                                       bool predicate) {
    using namespace sm_profiler;
    
    if (!predicate) return;
    if (event_no >= SM_PROFILER_MAX_EVENT_TYPES) return;
    
    uint32_t block_idx = get_block_idx();
    ProfilerBuffer buf = ProfilerBuffer::from_ptr(buffer);
    
    // Get event type entry
    EventTypeEntry* table = buf.get_event_type_table(block_idx);
    EventTypeEntry* entry = &table[event_no];
    
    // Check if active
    if (entry->active_event_id == 0) return;  // No active event
    
    uint32_t event_id = entry->active_event_id - 1;  // Convert back to 0-based
    
    // Record end timestamp
    DeviceEvent* ev = buf.get_event_data(block_idx, event_id);
    ev->en_timestamp_ns = get_timestamp();
    
    // Mark as inactive
    entry->active_event_id = 0;
}

/* Record instant event
 * 
 * @param predicate: Only execute if true (typically: threadIdx.x % 32 == 0)
 */
__device__ __forceinline__ void sm_profiler_event_instant(uint64_t* buffer,
                                                           uint32_t event_no,
                                                           bool predicate) {
    using namespace sm_profiler;
    
    if (!predicate) return;
    if (event_no >= SM_PROFILER_MAX_EVENT_TYPES) return;
    
    uint32_t block_idx = get_block_idx();
    ProfilerBuffer buf = ProfilerBuffer::from_ptr(buffer);
    
    // Allocate new event id
    uint32_t* counters = buf.get_event_id_counters();
    uint32_t event_id = atomicAdd(&counters[block_idx], 1);
    
    if (event_id >= buf.max_events_per_block) return;  // Buffer full
    
    // Record event
    DeviceEvent* ev = buf.get_event_data(block_idx, event_id);
    ev->event_id = event_id;
    ev->event_no = event_no;
    ev->block_id = block_idx;
    ev->sm_id = get_smid();
    ev->st_timestamp_ns = get_timestamp();
    ev->en_timestamp_ns = 0;
    ev->type = EVENT_TYPE_INSTANT;
    ev->reserved = 0;
}

#else /* !__CUDA_ARCH__ */

/* Host-side stubs */
inline void sm_profiler_init(uint64_t* buffer, uint32_t num_blocks, uint32_t max_events_per_block) {}
inline uint32_t sm_profiler_create_event(uint64_t* buffer, const char* name, bool predicate) { return 0; }
inline void sm_profiler_event_start(uint64_t* buffer, uint32_t event_no, bool predicate) {}
inline void sm_profiler_event_end(uint64_t* buffer, uint32_t event_no, bool predicate) {}
inline void sm_profiler_event_instant(uint64_t* buffer, uint32_t event_no, bool predicate) {}

#endif /* __CUDA_ARCH__ */
