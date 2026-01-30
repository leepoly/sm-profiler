/*
 * SM Profiler Library Demo
 * 
 * This demonstrates how to use the sm_profiler library in a CUDA application.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "sm_profiler.h"

/* Example kernel that uses the profiler */
__global__ void demo_kernel(uint64_t* profiler_buffer, int num_blocks, int max_events_per_block, int num_iterations) {
    /* Lane id for predicate */
    int lane_id = threadIdx.x % 32;
    bool is_lane_0 = (lane_id == 0);
    
    /* Initialize profiler */
    sm_profiler_init(profiler_buffer, num_blocks, max_events_per_block);
    
    /* Create event types (only lane 0, once per block) */
    uint32_t evt_compute = sm_profiler_create_event(profiler_buffer, "compute", is_lane_0);
    uint32_t evt_memory = sm_profiler_create_event(profiler_buffer, "memory", is_lane_0);
    uint32_t evt_sync = sm_profiler_create_event(profiler_buffer, "sync", is_lane_0);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        /* Start of computation phase */
        sm_profiler_event_start(profiler_buffer, evt_compute, is_lane_0);
        
        /* Simulate some work */
        float sum = 0.0f;
        for (int i = 0; i < 1000; i++) {
            sum += threadIdx.x * 0.001f;
        }
        
        sm_profiler_event_end(profiler_buffer, evt_compute, is_lane_0);
        
        /* Memory phase */
        sm_profiler_event_start(profiler_buffer, evt_memory, is_lane_0);
        
        __syncthreads();
        
        sm_profiler_event_end(profiler_buffer, evt_memory, is_lane_0);
        
        /* Sync point */
        sm_profiler_event_instant(profiler_buffer, evt_sync, is_lane_0);
        
        __syncthreads();
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
    const int num_iterations = 3;
    const int events_per_iteration = 3;
    const int max_events_per_block = num_iterations * events_per_iteration + 10;

    printf("\nConfiguration:\n");
    printf("  Blocks: %d\n", num_blocks);
    printf("  Threads per block: %d\n", num_threads);
    printf("  Iterations: %d\n", num_iterations);
    printf("  Max events per block: %d\n", max_events_per_block);

    /* Create profiler buffer */
    printf("\nCreating profiler buffer...\n");
    sm_profiler_buffer_t profiler = sm_profiler_create_buffer(
        num_blocks, 
        max_events_per_block
    );

    if (!profiler) {
        fprintf(stderr, "Failed to create profiler buffer\n");
        return 1;
    }

    /* Get device pointer for kernel */
    uint64_t* device_ptr = sm_profiler_get_device_ptr(profiler);

    /* Launch kernel */
    printf("Launching kernel...\n");
    demo_kernel<<<num_blocks, num_threads>>>(device_ptr, num_blocks, max_events_per_block, num_iterations);

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
