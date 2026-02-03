/*
 * SM Profiler Library Demo
 * 
 * This demonstrates how to use the sm_profiler library in a CUDA application.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "sm_profiler.h"

/* Define event type constants - global across all blocks and groups */
#define EVT_COMPUTE  0
#define EVT_MEMORY   1
#define EVT_SYNC     2

/* Example kernel that uses the profiler */
__global__ void demo_kernel(uint64_t* profiler_buffer, int num_iterations) {
    /* Lane id for predicate - only lane 0 records events */
    bool is_lane_0 = (sm_profiler_get_lane_id() == 0);

    float sum = 0.0f;
    for (int iter = 0; iter < num_iterations; iter++) {
        /* Start of computation phase */
        sm_profiler_event_start(profiler_buffer, EVT_COMPUTE, is_lane_0);

        /* Simulate some work */
        for (int i = 0; i < 1000; i++) {
            sum += threadIdx.x * 0.001f;
        }

        sm_profiler_event_end(profiler_buffer, EVT_COMPUTE, is_lane_0);

        sm_profiler_event_start(profiler_buffer, EVT_MEMORY, is_lane_0);

        __syncthreads();

        sm_profiler_event_end(profiler_buffer, EVT_MEMORY, is_lane_0);

        sm_profiler_event_instant(profiler_buffer, EVT_SYNC, is_lane_0);

        __syncthreads();
    }
    
    /* Prevent compiler from optimizing away the sum */
    if (sum < 0) {
        profiler_buffer[0] = 0;
    }
}

int main() {
    /* Check CUDA */
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA not available\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s\n", prop.name);
    
    /* Configuration */
    const int num_blocks = 4;
    const int num_threads = 128;
    const int num_warps = num_threads / 32;  /* 4 warps per block */
    const int max_events_per_group = 100;    /* Per warp, not per block */
    
    printf("\nConfiguration:\n");
    printf("  Blocks: %d\n", num_blocks);
    printf("  Threads per block: %d\n", num_threads);
    printf("  Warps per block (groups): %d\n", num_warps);
    printf("  Max events per group: %d\n", max_events_per_group);
    
    /* Create profiler buffer (enabled=1) */
    printf("\nCreating profiler buffer (enabled=1)...\n");
    sm_profiler_buffer_t profiler = sm_profiler_create_buffer(
        num_blocks,
        num_warps,
        max_events_per_group,
        1  /* enabled */
    );
    
    if (!profiler) {
        fprintf(stderr, "Failed to create profiler buffer\n");
        return 1;
    }
    
    /* Register event type names (once during setup) */
    printf("Registering event types...\n");
    sm_profiler_register_event(profiler, EVT_COMPUTE, "compute");
    sm_profiler_register_event(profiler, EVT_MEMORY, "memory");
    sm_profiler_register_event(profiler, EVT_SYNC, "sync");
    
    /* Initialize buffer before kernel launch (clears counters and states) */
    sm_profiler_init_buffer(profiler);
    
    /* Get device pointer for kernel */
    uint64_t* device_ptr = sm_profiler_get_device_ptr(profiler);
    
    /* Launch kernel */
    printf("Launching kernel...\n");
    const int num_iterations = 3;
    demo_kernel<<<num_blocks, num_threads>>>(device_ptr, num_iterations);
    
    /* Check for errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        sm_profiler_destroy_buffer(profiler);
        return 1;
    }
    
    /* Synchronize */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        sm_profiler_destroy_buffer(profiler);
        return 1;
    }
    
    printf("Kernel completed\n");
    
    /* Export trace to JSON */
    printf("\nExporting trace...\n");
    
    int ret = sm_profiler_export_to_file(profiler, "sm_profile.json");
    if (ret != 0) {
        fprintf(stderr, "Failed to export trace\n");
        sm_profiler_destroy_buffer(profiler);
        return 1;
    }
    
    printf("Trace saved to sm_profile.json\n");
    printf("\nTo view: Open sm_profile.json in Chrome chrome://tracing or https://ui.perfetto.dev/\n");
    
    /* Cleanup */
    sm_profiler_destroy_buffer(profiler);
    
    return 0;
}
