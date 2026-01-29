/*
 * SM-level C Profiler Library
 * 
 * A simplified library for SM-level profiling in CUDA kernels.
 * 
 * Usage:
 *   1. Call sm_profiler_create_buffer(num_blocks, num_groups, max_events_per_group) 
 *      to create a profiler buffer
 *   2. In your kernel, use PROFILER_INIT_CUDA, PROFILER_EVENT_START, etc.
 *   3. After kernel execution, call sm_profiler_export_to_file(buffer, "trace.json")
 *      to export results
 *   4. Call sm_profiler_destroy_buffer(buffer) to cleanup
 */

#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to profiler buffer */
typedef struct sm_profiler_buffer* sm_profiler_buffer_t;

/*
 * Create a profiler buffer
 * 
 * @param num_blocks: Number of CUDA blocks expected
 * @param num_groups: Number of groups per block (e.g., warps, thread groups)
 * @param max_events_per_group: Maximum events per group
 * @return: Handle to profiler buffer, or NULL on error
 */
sm_profiler_buffer_t sm_profiler_create_buffer(
    uint32_t num_blocks,
    uint32_t num_groups,
    uint32_t max_events_per_group
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
 * @param event_names: Optional array of event names (indexed by event_id). 
 *                     If NULL, uses "event_X" format.
 * @return: 0 on success, non-zero on error
 */
int sm_profiler_export_to_file(
    sm_profiler_buffer_t buffer,
    const char* filename,
    const char** event_names
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
 * Device-side macros for use in CUDA kernels
 * Include this header in your .cu files and use these macros inside kernels
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

__device__ __forceinline__ uint32_t get_timestamp() {
    uint32_t volatile ret;
    asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(ret));
    return ret;
}

/* Tag encoding:
 *   [31-19]      [18-11]       [10-2]         [1-0]
 *  [event no] [block group] [event type] [begin/end/instant]
 */
constexpr uint32_t EVENT_IDX_SHIFT = 2;
constexpr uint32_t BLOCK_GROUP_IDX_SHIFT = 11;
constexpr uint32_t EVENT_NO_SHIFT = 19;

constexpr uint32_t EVENT_BEGIN = 0x0;
constexpr uint32_t EVENT_END = 0x1;
constexpr uint32_t EVENT_INSTANT = 0x2;

__device__ __forceinline__ uint32_t encode_tag(uint32_t block_group_idx,
                                                uint32_t event_idx,
                                                uint32_t event_type) {
    return (block_group_idx << BLOCK_GROUP_IDX_SHIFT) |
           (event_idx << EVENT_IDX_SHIFT) | event_type;
}

__device__ __forceinline__ uint32_t make_event_tag_start(uint32_t base_tag,
                                                          uint32_t event_id,
                                                          uint32_t event_no) {
    return base_tag | (event_id << EVENT_IDX_SHIFT) |
           (event_no << EVENT_NO_SHIFT) | EVENT_BEGIN;
}

__device__ __forceinline__ uint32_t make_event_tag_end(uint32_t base_tag,
                                                        uint32_t event_id,
                                                        uint32_t event_no) {
    return base_tag | (event_id << EVENT_IDX_SHIFT) |
           (event_no << EVENT_NO_SHIFT) | EVENT_END;
}

__device__ __forceinline__ uint32_t make_event_tag_instant(uint32_t base_tag,
                                                            uint32_t event_id,
                                                            uint32_t event_no) {
    return base_tag | (event_id << EVENT_IDX_SHIFT) |
           (event_no << EVENT_NO_SHIFT) | EVENT_INSTANT;
}

struct ProfilerEntry {
    union {
        struct {
            uint32_t nblocks;
            uint32_t ngroups;
        };
        struct {
            uint32_t tag;
            uint32_t delta_time;
        };
        uint64_t raw;
    };
};

} // namespace sm_profiler

/* Kernel-side variable declarations */
#define SM_PROFILER_DECL_VARS                                                  \
    volatile sm_profiler::ProfilerEntry sm_prof_entry;                          \
    uint64_t* sm_prof_write_ptr;                                               \
    uint32_t sm_prof_write_stride;                                             \
    uint32_t sm_prof_entry_tag_base;                                           \
    bool sm_prof_write_predicate;

/* Initialize profiler in kernel
 * 
 * @param buffer: Device pointer from sm_profiler_get_device_ptr()
 * @param group_idx: Group index within block (0 to num_groups-1)
 * @param num_groups: Total number of groups per block
 * @param write_predicate: Condition for this thread to write events
 *                         (typically: threadIdx.x == 0 for one writer per group)
 */
#define SM_PROFILER_INIT(buffer, group_idx, num_groups, write_predicate)       \
    do {                                                                       \
        if (sm_profiler::get_block_idx() == 0 && sm_profiler::get_thread_idx() == 0) { \
            sm_prof_entry.nblocks = sm_profiler::get_num_blocks();             \
            sm_prof_entry.ngroups = (num_groups);                              \
            (buffer)[0] = sm_prof_entry.raw;                                   \
        }                                                                      \
        sm_prof_write_ptr =                                                    \
            (buffer) + 1 + sm_profiler::get_block_idx() * (num_groups) + (group_idx); \
        sm_prof_write_stride = sm_profiler::get_num_blocks() * (num_groups);   \
        sm_prof_entry_tag_base =                                               \
            sm_profiler::encode_tag(sm_profiler::get_block_idx() * (num_groups) + (group_idx), 0, 0); \
        sm_prof_write_predicate = (write_predicate);                           \
    } while (0)

/* Record event start */
#define SM_PROFILER_EVENT_START(event_id, event_no)                            \
    do {                                                                       \
        if (sm_prof_write_predicate) {                                         \
            sm_prof_entry.tag = sm_profiler::make_event_tag_start(             \
                sm_prof_entry_tag_base, (event_id), (event_no));               \
            sm_prof_entry.delta_time = sm_profiler::get_timestamp();           \
            *sm_prof_write_ptr = sm_prof_entry.raw;                            \
            sm_prof_write_ptr += sm_prof_write_stride;                         \
        }                                                                      \
        __threadfence_block();                                                 \
    } while (0)

/* Record event end */
#define SM_PROFILER_EVENT_END(event_id, event_no)                              \
    do {                                                                       \
        __threadfence_block();                                                 \
        if (sm_prof_write_predicate) {                                         \
            sm_prof_entry.tag = sm_profiler::make_event_tag_end(               \
                sm_prof_entry_tag_base, (event_id), (event_no));               \
            sm_prof_entry.delta_time = sm_profiler::get_timestamp();           \
            *sm_prof_write_ptr = sm_prof_entry.raw;                            \
            sm_prof_write_ptr += sm_prof_write_stride;                         \
        }                                                                      \
    } while (0)

/* Record instant event */
#define SM_PROFILER_EVENT_INSTANT(event_id, event_no)                          \
    do {                                                                       \
        __threadfence_block();                                                 \
        if (sm_prof_write_predicate) {                                         \
            sm_prof_entry.tag = sm_profiler::make_event_tag_instant(           \
                sm_prof_entry_tag_base, (event_id), (event_no));               \
            sm_prof_entry.delta_time = sm_profiler::get_timestamp();           \
            *sm_prof_write_ptr = sm_prof_entry.raw;                            \
        }                                                                      \
        __threadfence_block();                                                 \
    } while (0)

#else /* !__CUDA_ARCH__ */

/* Host-side stubs */
#define SM_PROFILER_DECL_VARS
#define SM_PROFILER_INIT(buffer, group_idx, num_groups, write_predicate)
#define SM_PROFILER_EVENT_START(event_id, event_no)
#define SM_PROFILER_EVENT_END(event_id, event_no)
#define SM_PROFILER_EVENT_INSTANT(event_id, event_no)

#endif /* __CUDA_ARCH__ */
