/*
 * SM-level C Profiler Library - Implementation
 */

#include "sm_profiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct sm_profiler_buffer {
    uint32_t num_blocks;
    uint32_t num_groups;
    uint32_t max_events_per_group;
    int enabled;
    uint64_t* device_ptr;
    uint64_t* host_copy;
    size_t buffer_size;
    
    /* Event type names stored on host side for export */
    char event_names[SM_PROFILER_MAX_EVENT_TYPES][SM_PROFILER_MAX_EVENT_NAME_LEN];
};

/* Structure sizes (must match device side, defined in sm_profiler.h) */
#define ACTIVE_EVENT_ENTRY_SIZE 4  /* sizeof(SmProfilerActiveEventEntry) = 4 */
#define DEVICE_EVENT_SIZE 40       /* sizeof(SmProfilerDeviceEvent) = 40 */

/* Compile-time size checks */
typedef char static_assert_active_entry_size[(sizeof(SmProfilerActiveEventEntry) == ACTIVE_EVENT_ENTRY_SIZE) ? 1 : -1];

/* Buffer layout offsets (in uint64_t units) - must be consistent across all functions */
#define HEADER_SIZE_UINT64 2  /* 2 x uint64_t for header */

/* Calculate counters size in uint64_t units */
static uint32_t get_counters_size_uint64(uint32_t num_blocks, uint32_t num_groups) {
    size_t counters_bytes = (size_t)num_blocks * num_groups * sizeof(uint32_t);
    return (uint32_t)((counters_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
}

/* Calculate offset to active event table (in uint64_t units) */
static uint32_t get_active_event_offset(uint32_t num_blocks, uint32_t num_groups) {
    return HEADER_SIZE_UINT64 + get_counters_size_uint64(num_blocks, num_groups);
}

/* Calculate active table size in uint64_t units */
static uint32_t get_active_table_size_uint64(uint32_t num_blocks, uint32_t num_groups) {
    size_t active_bytes = (size_t)num_blocks * num_groups * SM_PROFILER_MAX_EVENT_TYPES * ACTIVE_EVENT_ENTRY_SIZE;
    return (uint32_t)((active_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
}

/* Calculate offset to event data (in uint64_t units) */
static uint32_t get_event_data_offset(uint32_t num_blocks, uint32_t num_groups) {
    return get_active_event_offset(num_blocks, num_groups) + get_active_table_size_uint64(num_blocks, num_groups);
}

sm_profiler_buffer_t sm_profiler_create_buffer(
    uint32_t num_blocks,
    uint32_t num_groups,
    uint32_t max_events_per_group,
    int enabled
) {
    if (num_blocks == 0 || num_groups == 0 || max_events_per_group == 0) {
        fprintf(stderr, "sm_profiler: Invalid parameters\n");
        return NULL;
    }

    sm_profiler_buffer_t buffer = (sm_profiler_buffer_t)malloc(sizeof(struct sm_profiler_buffer));
    if (!buffer) {
        fprintf(stderr, "sm_profiler: Failed to allocate buffer struct\n");
        return NULL;
    }

    buffer->num_blocks = num_blocks;
    buffer->num_groups = num_groups;
    buffer->max_events_per_group = max_events_per_group;
    buffer->enabled = enabled;

    /* Initialize event names to empty */
    for (int i = 0; i < SM_PROFILER_MAX_EVENT_TYPES; i++) {
        buffer->event_names[i][0] = '\0';
    }

    /* Calculate buffer size */
    if (enabled) {
        uint32_t event_data_offset_uint64 = get_event_data_offset(num_blocks, num_groups);
        /* Event data: [num_blocks][num_groups][max_events_per_group] */
        size_t event_data_size = (size_t)num_blocks * num_groups * max_events_per_group * DEVICE_EVENT_SIZE;
        size_t event_data_size_uint64 = (event_data_size + sizeof(uint64_t) - 1) / sizeof(uint64_t);
        buffer->buffer_size = (event_data_offset_uint64 + event_data_size_uint64) * sizeof(uint64_t);
    } else {
        /* Minimal buffer when disabled: just header */
        buffer->buffer_size = HEADER_SIZE_UINT64 * sizeof(uint64_t);
    }

    /* Allocate device memory */
    cudaError_t err = cudaMalloc(&buffer->device_ptr, buffer->buffer_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "sm_profiler: cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(buffer);
        return NULL;
    }

    /* Zero-initialize device memory */
    err = cudaMemset(buffer->device_ptr, 0, buffer->buffer_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "sm_profiler: cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(buffer->device_ptr);
        free(buffer);
        return NULL;
    }

    /* Allocate host copy buffer */
    buffer->host_copy = (uint64_t*)malloc(buffer->buffer_size);
    if (!buffer->host_copy) {
        fprintf(stderr, "sm_profiler: Failed to allocate host copy buffer\n");
        cudaFree(buffer->device_ptr);
        free(buffer);
        return NULL;
    }

    return buffer;
}

void sm_profiler_destroy_buffer(sm_profiler_buffer_t buffer) {
    if (!buffer) return;
    
    if (buffer->device_ptr) {
        cudaFree(buffer->device_ptr);
    }
    if (buffer->host_copy) {
        free(buffer->host_copy);
    }
    free(buffer);
}

uint64_t* sm_profiler_get_device_ptr(sm_profiler_buffer_t buffer) {
    if (!buffer) return NULL;
    return buffer->device_ptr;
}

/*
 * Initialize buffer on host side (call before kernel launch)
 * Sets header and clears counters and active event states
 */
void sm_profiler_init_buffer(sm_profiler_buffer_t buffer) {
    if (!buffer) return;
    
    /* Write header */
    /* header0: [num_blocks (32-bit) | num_groups (32-bit)] */
    /* header1: [max_events_per_group (32-bit) | enabled (32-bit)] */
    buffer->host_copy[0] = ((uint64_t)buffer->num_blocks << 32) | (uint64_t)buffer->num_groups;
    buffer->host_copy[1] = ((uint64_t)buffer->max_events_per_group << 32) | (uint64_t)(buffer->enabled ? 1 : 0);
    
    if (buffer->enabled) {
        /* Clear event id counters: [num_blocks * num_groups] */
        uint32_t* counters = (uint32_t*)(buffer->host_copy + HEADER_SIZE_UINT64);
        size_t num_counters = (size_t)buffer->num_blocks * buffer->num_groups;
        for (size_t i = 0; i < num_counters; i++) {
            counters[i] = 0;
        }
        
        /* Clear active event table */
        uint32_t active_event_offset = get_active_event_offset(buffer->num_blocks, buffer->num_groups);
        size_t active_table_size = (size_t)buffer->num_blocks * buffer->num_groups * SM_PROFILER_MAX_EVENT_TYPES * ACTIVE_EVENT_ENTRY_SIZE;
        char* active_table = (char*)(buffer->host_copy + active_event_offset);
        memset(active_table, 0, active_table_size);
        
        /* Clear event data */
        uint32_t event_data_offset = get_event_data_offset(buffer->num_blocks, buffer->num_groups);
        size_t event_data_size = (size_t)buffer->num_blocks * buffer->num_groups * buffer->max_events_per_group * DEVICE_EVENT_SIZE;
        char* event_data = (char*)(buffer->host_copy + event_data_offset);
        memset(event_data, 0, event_data_size);
    }
    
    /* Copy to device */
    cudaError_t err = cudaMemcpy(buffer->device_ptr, buffer->host_copy, buffer->buffer_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "sm_profiler: cudaMemcpy failed in init_buffer: %s\n", cudaGetErrorString(err));
    }
}

/*
 * Register an event type name (call once during setup)
 */
int sm_profiler_register_event(
    sm_profiler_buffer_t buffer,
    uint32_t event_no,
    const char* name
) {
    if (!buffer || !name) return -1;
    if (event_no >= SM_PROFILER_MAX_EVENT_TYPES) {
        fprintf(stderr, "sm_profiler: event_no %u out of range (max %d)\n", event_no, SM_PROFILER_MAX_EVENT_TYPES);
        return -1;
    }

    strncpy(buffer->event_names[event_no], name, SM_PROFILER_MAX_EVENT_NAME_LEN - 1);
    buffer->event_names[event_no][SM_PROFILER_MAX_EVENT_NAME_LEN - 1] = '\0';

    return 0;
}

void sm_profiler_get_info(
    sm_profiler_buffer_t buffer,
    uint32_t* out_num_blocks,
    uint32_t* out_num_groups,
    uint32_t* out_max_events
) {
    if (!buffer) return;
    if (out_num_blocks) *out_num_blocks = buffer->num_blocks;
    if (out_num_groups) *out_num_groups = buffer->num_groups;
    if (out_max_events) *out_max_events = buffer->max_events_per_group;
}

/* ============================================================================
 * Export to Chrome Trace JSON format
 * ============================================================================ */

/* Compile-time size check for SmProfilerDeviceEvent (defined in sm_profiler.h) */
typedef char static_assert_host_event_size[(sizeof(SmProfilerDeviceEvent) == DEVICE_EVENT_SIZE) ? 1 : -1];

static void parse_header(uint64_t* host_copy, uint32_t* num_blocks, uint32_t* num_groups, uint32_t* max_events_per_group) {
    uint64_t header0 = host_copy[0];
    uint64_t header1 = host_copy[1];
    *num_blocks = (uint32_t)(header0 >> 32);
    *num_groups = (uint32_t)(header0 & 0xFFFFFFFF);
    *max_events_per_group = (uint32_t)(header1 >> 32);
}

static SmProfilerDeviceEvent* get_device_event(void* data_ptr, uint32_t num_groups, uint32_t max_events_per_group,
                                              uint32_t block_idx, uint32_t group_idx, uint32_t event_idx) {
    char* base = (char*)data_ptr;
    size_t group_offset = ((size_t)block_idx * num_groups + group_idx) * max_events_per_group;
    return (SmProfilerDeviceEvent*)(base + (group_offset + event_idx) * DEVICE_EVENT_SIZE);
}

/* Comparator for qsort */
static int compare_events_by_timestamp(const void* a, const void* b) {
    const SmProfilerDeviceEvent* ea = (const SmProfilerDeviceEvent*)a;
    const SmProfilerDeviceEvent* eb = (const SmProfilerDeviceEvent*)b;
    if (ea->st_timestamp_ns < eb->st_timestamp_ns) return -1;
    if (ea->st_timestamp_ns > eb->st_timestamp_ns) return 1;
    return 0;
}

int sm_profiler_export_to_file(
    sm_profiler_buffer_t buffer,
    const char* filename
) {
    if (!buffer || !filename) {
        fprintf(stderr, "sm_profiler_export_to_file: Invalid arguments\n");
        return -1;
    }

    /* Copy data from device to host */
    cudaError_t err = cudaMemcpy(buffer->host_copy, buffer->device_ptr, 
                                  buffer->buffer_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "sm_profiler: cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    /* Parse header */
    uint32_t num_blocks, num_groups, max_events_per_group;
    parse_header(buffer->host_copy, &num_blocks, &num_groups, &max_events_per_group);

    if (num_blocks == 0 || num_groups == 0 || max_events_per_group == 0) {
        fprintf(stderr, "sm_profiler: Invalid header in buffer\n");
        return -1;
    }

    if (!buffer->enabled) {
        fprintf(stderr, "sm_profiler: Profiling is disabled, no events to export\n");
        return -1;
    }

    /* Get pointer to counters and event data */
    uint32_t* counters = (uint32_t*)(buffer->host_copy + HEADER_SIZE_UINT64);
    uint32_t event_data_offset = get_event_data_offset(num_blocks, num_groups);
    void* event_data_ptr = (char*)buffer->host_copy + event_data_offset * sizeof(uint64_t);

    /* Count total events */
    size_t total_events = 0;
    size_t num_counters = (size_t)num_blocks * num_groups;
    for (size_t i = 0; i < num_counters; i++) {
        total_events += counters[i];
    }

    if (total_events == 0) {
        fprintf(stderr, "sm_profiler: No events recorded\n");
        return -1;
    }

    /* Collect all events */
    SmProfilerDeviceEvent* events = (SmProfilerDeviceEvent*)malloc(total_events * sizeof(SmProfilerDeviceEvent));
    if (!events) {
        fprintf(stderr, "sm_profiler: Failed to allocate events array\n");
        return -1;
    }

    size_t event_count = 0;
    for (uint32_t b = 0; b < num_blocks; b++) {
        for (uint32_t g = 0; g < num_groups; g++) {
            uint32_t counter_idx = b * num_groups + g;
            uint32_t group_event_count = counters[counter_idx];
            
            for (uint32_t e = 0; e < group_event_count && e < max_events_per_group; e++) {
                SmProfilerDeviceEvent* dev_ev = get_device_event(event_data_ptr, num_groups, max_events_per_group, b, g, e);
                events[event_count++] = *dev_ev;
            }
        }
    }

    /* Sort events by start timestamp using qsort (O(n log n) instead of O(n²)) */
    qsort(events, event_count, sizeof(SmProfilerDeviceEvent), compare_events_by_timestamp);

    /* Find min timestamp for relative time calculation */
    uint64_t min_timestamp = event_count > 0 ? events[0].st_timestamp_ns : 0;

    /* Open output file */
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "sm_profiler: Failed to open %s for writing\n", filename);
        free(events);
        return -1;
    }

    /* Write Chrome Trace JSON */
    fprintf(fp, "{\n");
    fprintf(fp, "  \"traceEvents\": [\n");

    size_t output_count = 0;
    int first_entry = 1;
    
    for (size_t i = 0; i < event_count; i++) {
        SmProfilerDeviceEvent* ev = &events[i];
        
        /* Get event name from host-side registered names */
        const char* event_name = buffer->event_names[ev->event_no];
        if (event_name[0] == '\0') {
            event_name = "unknown";
        }
        char event_name_id[128];
        snprintf(event_name_id, sizeof(event_name_id), "%s_%u", event_name, ev->event_id);

        /* Calculate relative timestamp in microseconds (Chrome trace format uses us) */
        double ts_us = (double)(ev->st_timestamp_ns - min_timestamp) / 1000.0;

        /* Generate thread name */
        char tid_str[64];
        snprintf(tid_str, sizeof(tid_str), "block_%u_group_%u", ev->block_id, ev->group_idx);
        
        if (ev->type == 0) {
            /* Range event - Begin */
            if (!first_entry) fprintf(fp, ",\n");
            first_entry = 0;
            
            fprintf(fp, "    {\"name\": \"%s\", \"ph\": \"B\", \"ts\": %.3f, \"pid\": %u, \"tid\": \"%s\", \"cat\": \"gpu\"}",
                    event_name_id, ts_us, ev->sm_id, tid_str);
            output_count++;
            
            /* Range event - End */
            double end_ts_us = (double)(ev->en_timestamp_ns - min_timestamp) / 1000.0;
            fprintf(fp, ",\n");
            fprintf(fp, "    {\"name\": \"%s\", \"ph\": \"E\", \"ts\": %.3f, \"pid\": %u, \"tid\": \"%s\", \"cat\": \"gpu\"}",
                    event_name_id, end_ts_us, ev->sm_id, tid_str);
            output_count++;
        } else {
            /* Instant event */
            if (!first_entry) fprintf(fp, ",\n");
            first_entry = 0;
            
            fprintf(fp, "    {\"name\": \"%s\", \"ph\": \"i\", \"ts\": %.3f, \"pid\": %u, \"tid\": \"%s\", \"cat\": \"gpu\", \"s\": \"t\"}",
                    event_name_id, ts_us, ev->sm_id, tid_str);
            output_count++;
        }
    }

    fprintf(fp, "\n  ],\n");
    fprintf(fp, "  \"displayTimeUnit\": \"ns\",\n");
    fprintf(fp, "  \"metadata\": {\n");
    fprintf(fp, "    \"num_blocks\": %u,\n", num_blocks);
    fprintf(fp, "    \"num_groups\": %u,\n", num_groups);
    fprintf(fp, "    \"max_events_per_group\": %u,\n", max_events_per_group);
    fprintf(fp, "    \"total_events\": %zu\n", event_count);
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");

    free(events);
    fclose(fp);

    printf("sm_profiler: Exported %zu events (%zu trace entries) to %s\n", 
           event_count, output_count, filename);
    return 0;
}
