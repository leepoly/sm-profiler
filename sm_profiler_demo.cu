/*
 * SM Profiler Library Demo
 * 
 * This demonstrates how to use the sm_profiler library in a CUDA application.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "sm_profiler.h"

/* Example kernel that uses the profiler */
__global__ void demo_kernel(uint64_t* profiler_buffer, int num_iterations) {
    /* Each warp is a profiling group, only lane 0 writes events */
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    int lane_id = threadIdx.x % 32;
    
    /* Declare profiler variables */
    SM_PROFILER_DECL_VARS;
    
    /* Initialize profiler for this block/group */
    SM_PROFILER_INIT(profiler_buffer, warp_id, num_warps, lane_id == 0);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        /* Start of computation phase */
        SM_PROFILER_EVENT_START(0, iter);  /* event_id=0, event_no=iter */
        
        /* Simulate some work */
        float sum = 0.0f;
        for (int i = 0; i < 1000; i++) {
            sum += threadIdx.x * 0.001f;
        }
        
        SM_PROFILER_EVENT_END(0, iter);
        
        /* Memory phase */
        SM_PROFILER_EVENT_START(1, iter);  /* event_id=1 */
        
        __syncthreads();
        
        SM_PROFILER_EVENT_END(1, iter);
        
        /* Sync point */
        SM_PROFILER_EVENT_INSTANT(2, iter);  /* event_id=2, instant */
        
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
    const int num_warps = num_threads / 32;  /* 4 warps per block */
    const int num_iterations = 3;
    const int max_events_per_warp = num_iterations * 6;  /* 3 events per iteration, begin+end each */
    
    printf("\nConfiguration:\n");
    printf("  Blocks: %d\n", num_blocks);
    printf("  Threads per block: %d\n", num_threads);
    printf("  Warps per block (profiling groups): %d\n", num_warps);
    printf("  Iterations: %d\n", num_iterations);
    printf("  Max events per warp: %d\n", max_events_per_warp);
    
    /* Create profiler buffer */
    printf("\nCreating profiler buffer...\n");
    sm_profiler_buffer_t profiler = sm_profiler_create_buffer(
        num_blocks, 
        num_warps, 
        max_events_per_warp
    );
    
    if (!profiler) {
        fprintf(stderr, "Failed to create profiler buffer\n");
        return 1;
    }
    
    /* Get device pointer for kernel */
    uint64_t* device_ptr = sm_profiler_get_device_ptr(profiler);
    
    /* Launch kernel */
    printf("Launching kernel...\n");
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
    
    /* Define event names (optional) */
    const char* event_names[] = {
        "compute",   /* event_id = 0 */
        "memory",    /* event_id = 1 */
        "sync",      /* event_id = 2 */
    };
    
    int ret = sm_profiler_export_to_file(profiler, "sm_profile.json", event_names);
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
