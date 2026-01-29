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
    uint64_t* device_ptr;
    uint64_t* host_copy;      // For copying data back to host
    size_t buffer_size;       // Total size in bytes
};

sm_profiler_buffer_t sm_profiler_create_buffer(
    uint32_t num_blocks,
    uint32_t num_groups,
    uint32_t max_events_per_group
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

    /* Calculate buffer size:
     * - First entry: header (nblocks, ngroups)
     * - Remaining: one entry per block per group per max_events
     */
    size_t num_entries = 1 + (size_t)num_blocks * num_groups * max_events_per_group;
    buffer->buffer_size = num_entries * sizeof(uint64_t);

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

/* Decode tag structure (must match device-side encoding) */
typedef struct {
    uint32_t event_no;
    uint32_t block_idx;
    uint32_t group_idx;
    uint32_t event_id;
    uint32_t event_type;  /* 0=begin, 1=end, 2=instant */
} decoded_tag_t;

static void decode_tag(uint32_t tag, uint32_t num_blocks, uint32_t num_groups, decoded_tag_t* out) {
    out->event_no = tag >> 19;
    uint32_t block_group_tag = (tag >> 11) & 0xFF;
    out->block_idx = block_group_tag / num_groups;
    out->group_idx = block_group_tag % num_groups;
    out->event_id = (tag >> 2) & 0x1FF;
    out->event_type = tag & 0x3;
}

/* Chrome Trace event types */
typedef struct {
    uint32_t event_id;
    uint32_t event_no;
    uint32_t block_idx;
    uint32_t group_idx;
    uint64_t timestamp_ns;  /* GPU timestamps are in ns */
    int is_begin;           /* 1=begin, 0=end, 2=instant */
} trace_event_t;

int sm_profiler_export_to_file(
    sm_profiler_buffer_t buffer,
    const char* filename,
    const char** event_names
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
    uint32_t num_blocks, num_groups;
    memcpy(&num_blocks, &buffer->host_copy[0], sizeof(uint32_t));
    memcpy(&num_groups, (char*)&buffer->host_copy[0] + sizeof(uint32_t), sizeof(uint32_t));

    if (num_blocks == 0 || num_groups == 0) {
        fprintf(stderr, "sm_profiler: Invalid header in buffer\n");
        return -1;
    }

    /* Open output file */
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "sm_profiler: Failed to open %s for writing\n", filename);
        return -1;
    }

    /* Count events and collect them */
    size_t max_events = (size_t)num_blocks * num_groups * buffer->max_events_per_group;
    trace_event_t* events = (trace_event_t*)malloc(max_events * sizeof(trace_event_t));
    if (!events) {
        fprintf(stderr, "sm_profiler: Failed to allocate events array\n");
        fclose(fp);
        return -1;
    }

    size_t event_count = 0;
    for (size_t i = 1; i < 1 + max_events; i++) {
        if (buffer->host_copy[i] == 0) continue;

        uint32_t tag, timestamp;
        memcpy(&tag, &buffer->host_copy[i], sizeof(uint32_t));
        memcpy(&timestamp, (char*)&buffer->host_copy[i] + sizeof(uint32_t), sizeof(uint32_t));

        if (tag == 0) continue;

        decoded_tag_t decoded;
        decode_tag(tag, num_blocks, num_groups, &decoded);

        events[event_count].event_id = decoded.event_id;
        events[event_count].event_no = decoded.event_no;
        events[event_count].block_idx = decoded.block_idx;
        events[event_count].group_idx = decoded.group_idx;
        events[event_count].timestamp_ns = timestamp;
        events[event_count].is_begin = decoded.event_type;
        event_count++;
    }

    /* Sort events by timestamp for proper trace visualization */
    /* Simple bubble sort for small event counts */
    for (size_t i = 0; i < event_count; i++) {
        for (size_t j = i + 1; j < event_count; j++) {
            if (events[j].timestamp_ns < events[i].timestamp_ns) {
                trace_event_t tmp = events[i];
                events[i] = events[j];
                events[j] = tmp;
            }
        }
    }

    /* Find min timestamp for relative time calculation */
    uint64_t min_timestamp = event_count > 0 ? events[0].timestamp_ns : 0;
    for (size_t i = 1; i < event_count; i++) {
        if (events[i].timestamp_ns < min_timestamp) {
            min_timestamp = events[i].timestamp_ns;
        }
    }

    /* Write Chrome Trace JSON */
    fprintf(fp, "{\n");
    fprintf(fp, "  \"traceEvents\": [\n");

    for (size_t i = 0; i < event_count; i++) {
        trace_event_t* ev = &events[i];
        
        /* Get event name */
        char event_name[64];
        if (event_names && event_names[ev->event_id]) {
            snprintf(event_name, sizeof(event_name), "%s_%u", 
                     event_names[ev->event_id], ev->event_no);
        } else {
            snprintf(event_name, sizeof(event_name), "event_%u_%u", 
                     ev->event_id, ev->event_no);
        }

        /* Calculate relative timestamp in microseconds (Chrome trace format uses us) */
        double ts_us = (double)(ev->timestamp_ns - min_timestamp) / 1000.0;

        /* Determine phase */
        const char* phase;
        if (ev->is_begin == 0) phase = "B";      /* Begin */
        else if (ev->is_begin == 1) phase = "E"; /* End */
        else phase = "i";                         /* Instant */

        /* Chrome trace event format:
         * {
         *   "name": "...",
         *   "ph": "B|E|i",
         *   "ts": timestamp_us,
         *   "pid": process_id (used for block),
         *   "tid": thread_id (used for group),
         *   "cat": "..."
         * }
         */
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"name\": \"%s\",\n", event_name);
        fprintf(fp, "      \"ph\": \"%s\",\n", phase);
        fprintf(fp, "      \"ts\": %.3f,\n", ts_us);
        fprintf(fp, "      \"pid\": %u,\n", ev->block_idx);
        fprintf(fp, "      \"tid\": %u,\n", ev->group_idx);
        fprintf(fp, "      \"cat\": \"gpu\"\n");
        fprintf(fp, "    }%s\n", (i < event_count - 1) ? "," : "");
    }

    fprintf(fp, "  ],\n");
    fprintf(fp, "  \"displayTimeUnit\": \"ns\",\n");
    fprintf(fp, "  \"metadata\": {\n");
    fprintf(fp, "    \"num_blocks\": %u,\n", num_blocks);
    fprintf(fp, "    \"num_groups\": %u,\n", num_groups);
    fprintf(fp, "    \"total_events\": %zu\n", event_count);
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");

    free(events);
    fclose(fp);

    printf("sm_profiler: Exported %zu events to %s\n", event_count, filename);
    return 0;
}
