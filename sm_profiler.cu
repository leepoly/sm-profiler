/*
 * SM-level C Profiler Library - Implementation
 */

#include "sm_profiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct sm_profiler_buffer {
    uint32_t num_blocks;
    uint32_t max_events_per_block;
    uint64_t* device_ptr;
    uint64_t* host_copy;
    size_t buffer_size;
};

/* Structure sizes (must match device side) */
#define EVENT_TYPE_ENTRY_SIZE (SM_PROFILER_MAX_EVENT_NAME_LEN + 8)  /* name + active + padding */
#define DEVICE_EVENT_SIZE 32  /* sizeof(DeviceEvent) */

sm_profiler_buffer_t sm_profiler_create_buffer(
    uint32_t num_blocks,
    uint32_t max_events_per_block
) {
    if (num_blocks == 0 || max_events_per_block == 0) {
        fprintf(stderr, "sm_profiler: Invalid parameters\n");
        return NULL;
    }

    sm_profiler_buffer_t buffer = (sm_profiler_buffer_t)malloc(sizeof(struct sm_profiler_buffer));
    if (!buffer) {
        fprintf(stderr, "sm_profiler: Failed to allocate buffer struct\n");
        return NULL;
    }

    buffer->num_blocks = num_blocks;
    buffer->max_events_per_block = max_events_per_block;

    /* Calculate buffer size:
     * [0] Header (1 x uint64)
     * [1 .. num_blocks] Event id counters (uint32 per block, packed in uint64)
     * Event type table: num_blocks * SM_PROFILER_MAX_EVENT_TYPES * sizeof(EventTypeEntry)
     * Event data: num_blocks * max_events_per_block * sizeof(DeviceEvent)
     */
    size_t header_size = sizeof(uint64_t);
    size_t counters_size = num_blocks * sizeof(uint32_t);
    size_t type_table_size = num_blocks * SM_PROFILER_MAX_EVENT_TYPES * EVENT_TYPE_ENTRY_SIZE;
    size_t event_data_size = num_blocks * max_events_per_block * DEVICE_EVENT_SIZE;
    
    buffer->buffer_size = header_size + counters_size + type_table_size + event_data_size;
    
    /* Round up to uint64 alignment */
    buffer->buffer_size = (buffer->buffer_size + sizeof(uint64_t) - 1) & ~(sizeof(uint64_t) - 1);

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

void sm_profiler_get_info(
    sm_profiler_buffer_t buffer,
    uint32_t* out_num_blocks,
    uint32_t* out_max_events
) {
    if (!buffer) return;
    if (out_num_blocks) *out_num_blocks = buffer->num_blocks;
    if (out_max_events) *out_max_events = buffer->max_events_per_block;
}

/* ============================================================================
 * Export to Chrome Trace JSON format
 * ============================================================================ */

/* Trace event structure (host side) */
typedef struct {
    uint32_t event_id;
    uint32_t event_no;
    uint32_t block_id;
    uint32_t sm_id;
    uint64_t st_timestamp_ns;
    uint64_t en_timestamp_ns;
    uint32_t type;  /* 0=range, 1=instant */
} trace_event_t;

/* Event type entry (host side, packed) */
typedef struct {
    char name[SM_PROFILER_MAX_EVENT_NAME_LEN];
    uint32_t active_event_id;
    uint32_t padding;
} host_event_type_entry_t;

/* Device event structure (host side, packed) */
typedef struct {
    uint32_t event_id;
    uint32_t event_no;
    uint32_t block_id;
    uint32_t sm_id;
    uint64_t st_timestamp_ns;
    uint64_t en_timestamp_ns;
    uint32_t type;
    uint32_t reserved;
} host_device_event_t;

static void* get_event_type_table_ptr(uint64_t* host_copy, uint32_t num_blocks) {
    char* base = (char*)host_copy;
    return base + sizeof(uint64_t) + num_blocks * sizeof(uint32_t);
}

static void* get_event_data_ptr(uint64_t* host_copy, uint32_t num_blocks) {
    char* base = (char*)host_copy;
    size_t type_table_size = num_blocks * SM_PROFILER_MAX_EVENT_TYPES * EVENT_TYPE_ENTRY_SIZE;
    return base + sizeof(uint64_t) + num_blocks * sizeof(uint32_t) + type_table_size;
}

static host_event_type_entry_t* get_event_type_entry(void* table_ptr, uint32_t block_idx, uint32_t event_no) {
    char* base = (char*)table_ptr;
    host_event_type_entry_t* block_table = (host_event_type_entry_t*)(base + block_idx * SM_PROFILER_MAX_EVENT_TYPES * EVENT_TYPE_ENTRY_SIZE);
    return &block_table[event_no];
}

static host_device_event_t* get_device_event(void* data_ptr, uint32_t max_events_per_block, 
                                              uint32_t block_idx, uint32_t event_idx) {
    char* base = (char*)data_ptr;
    return (host_device_event_t*)(base + (block_idx * max_events_per_block + event_idx) * DEVICE_EVENT_SIZE);
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
    uint64_t header = buffer->host_copy[0];
    uint32_t num_blocks = (uint32_t)(header >> 32);
    uint32_t max_events_per_block = (uint32_t)(header & 0xFFFFFFFF);

    if (num_blocks == 0 || max_events_per_block == 0) {
        fprintf(stderr, "sm_profiler: Invalid header in buffer\n");
        return -1;
    }

    /* Get pointers to data regions */
    void* type_table_ptr = get_event_type_table_ptr(buffer->host_copy, num_blocks);
    void* event_data_ptr = get_event_data_ptr(buffer->host_copy, num_blocks);

    /* Count total events */
    uint32_t* counters = (uint32_t*)(buffer->host_copy + 1);
    size_t total_events = 0;
    for (uint32_t b = 0; b < num_blocks; b++) {
        total_events += counters[b];
    }

    if (total_events == 0) {
        fprintf(stderr, "sm_profiler: No events recorded\n");
        return -1;
    }

    /* Collect all events */
    trace_event_t* events = (trace_event_t*)malloc(total_events * sizeof(trace_event_t));
    if (!events) {
        fprintf(stderr, "sm_profiler: Failed to allocate events array\n");
        return -1;
    }

    size_t event_count = 0;
    for (uint32_t b = 0; b < num_blocks; b++) {
        uint32_t block_event_count = counters[b];
        for (uint32_t e = 0; e < block_event_count; e++) {
            host_device_event_t* dev_ev = get_device_event(event_data_ptr, max_events_per_block, b, e);
            
            if (dev_ev->event_id != e) continue;  // Sanity check
            
            events[event_count].event_id = dev_ev->event_id;
            events[event_count].event_no = dev_ev->event_no;
            events[event_count].block_id = dev_ev->block_id;
            events[event_count].sm_id = dev_ev->sm_id;
            events[event_count].st_timestamp_ns = dev_ev->st_timestamp_ns;
            events[event_count].en_timestamp_ns = dev_ev->en_timestamp_ns;
            events[event_count].type = dev_ev->type;
            event_count++;
        }
    }

    /* Sort events by start timestamp */
    for (size_t i = 0; i < event_count; i++) {
        for (size_t j = i + 1; j < event_count; j++) {
            if (events[j].st_timestamp_ns < events[i].st_timestamp_ns) {
                trace_event_t tmp = events[i];
                events[i] = events[j];
                events[j] = tmp;
            }
        }
    }

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
    for (size_t i = 0; i < event_count; i++) {
        trace_event_t* ev = &events[i];
        
        /* Get event name from event type table */
        host_event_type_entry_t* type_entry = get_event_type_entry(type_table_ptr, ev->block_id, ev->event_no);
        char event_name[SM_PROFILER_MAX_EVENT_NAME_LEN];
        if (type_entry->name[0] != '\0') {
            strncpy(event_name, type_entry->name, SM_PROFILER_MAX_EVENT_NAME_LEN - 1);
            event_name[SM_PROFILER_MAX_EVENT_NAME_LEN - 1] = '\0';
        } else {
            snprintf(event_name, sizeof(event_name), "event_%u", ev->event_no);
        }

        /* Caption = event_name + "_" + event_id */
        char caption[SM_PROFILER_MAX_EVENT_NAME_LEN + 16];
        snprintf(caption, sizeof(caption), "%s_%u", event_name, ev->event_id);

        /* Calculate relative timestamp in microseconds (Chrome trace format uses us) */
        double ts_us = (double)(ev->st_timestamp_ns - min_timestamp) / 1000.0;

        /* Output format:
         * - Range: Output B and E events
         * - Instant: Output i event
         * pid = sm_id, tid = block_id
         */
        
        if (ev->type == 0) {
            /* Range event - Begin */
            fprintf(fp, "    {\n");
            fprintf(fp, "      \"name\": \"%s\",\n", caption);
            fprintf(fp, "      \"ph\": \"B\",\n");
            fprintf(fp, "      \"ts\": %.3f,\n", ts_us);
            fprintf(fp, "      \"pid\": %u,\n", ev->sm_id);
            fprintf(fp, "      \"tid\": %u,\n", ev->block_id);
            fprintf(fp, "      \"cat\": \"gpu\"\n");
            fprintf(fp, "    },\n");
            output_count++;
            
            /* Range event - End */
            double end_ts_us = (double)(ev->en_timestamp_ns - min_timestamp) / 1000.0;
            fprintf(fp, "    {\n");
            fprintf(fp, "      \"name\": \"%s\",\n", caption);
            fprintf(fp, "      \"ph\": \"E\",\n");
            fprintf(fp, "      \"ts\": %.3f,\n", end_ts_us);
            fprintf(fp, "      \"pid\": %u,\n", ev->sm_id);
            fprintf(fp, "      \"tid\": %u,\n", ev->block_id);
            fprintf(fp, "      \"cat\": \"gpu\"\n");
            fprintf(fp, "    }%s\n", (i < event_count - 1 || output_count < total_events * 2 - 1) ? "," : "");
            output_count++;
        } else {
            /* Instant event */
            fprintf(fp, "    {\n");
            fprintf(fp, "      \"name\": \"%s\",\n", caption);
            fprintf(fp, "      \"ph\": \"i\",\n");
            fprintf(fp, "      \"ts\": %.3f,\n", ts_us);
            fprintf(fp, "      \"pid\": %u,\n", ev->sm_id);
            fprintf(fp, "      \"tid\": %u,\n", ev->block_id);
            fprintf(fp, "      \"cat\": \"gpu\"\n");
            fprintf(fp, "    }%s\n", (i < event_count - 1) ? "," : "");
            output_count++;
        }
    }

    fprintf(fp, "  ],\n");
    fprintf(fp, "  \"displayTimeUnit\": \"ns\",\n");
    fprintf(fp, "  \"metadata\": {\n");
    fprintf(fp, "    \"num_blocks\": %u,\n", num_blocks);
    fprintf(fp, "    \"max_events_per_block\": %u,\n", max_events_per_block);
    fprintf(fp, "    \"total_events\": %zu\n", event_count);
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");

    free(events);
    fclose(fp);

    printf("sm_profiler: Exported %zu events (%zu trace entries) to %s\n", 
           event_count, output_count, filename);
    return 0;
}
