"""
SM Profiler Triton Demo

Demonstrates how to use sm_profiler with Triton kernels.
"""

import torch
import triton
import triton.language as tl

from sm_profiler import SmProfiler
from sm_profiler.triton_ops import (
    profiler_init,
    profiler_event_start,
    profiler_event_end,
    profiler_event_instant,
)

# Event type constants
EVT_COMPUTE = tl.constexpr(0)
EVT_MEMORY = tl.constexpr(1)
EVT_SYNC = tl.constexpr(2)


@triton.jit
def demo_kernel(
    profiler_buffer_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    """Demo kernel with profiling."""
    # Initialize profiler context once at start
    ctx = profiler_init(profiler_buffer_ptr)
    
    # Get block ID
    block_idx = tl.program_id(0)
    
    # For this demo, we treat each program as one group (group_idx = 0)
    # In a real kernel with multiple warps, you'd compute group_idx differently
    group_idx = 0
    
    # --- Compute phase (using start/end) ---
    profiler_event_start(ctx, block_idx, group_idx, EVT_COMPUTE)
    
    # Simulate computation
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Some dummy computation
    x = tl.load(output_ptr + offsets, mask=mask, other=0.0)
    for _ in range(10):
        x = x * 1.01 + 0.001
    
    profiler_event_end(ctx, block_idx, group_idx, EVT_COMPUTE)
    
    # --- Memory phase ---
    profiler_event_start(ctx, block_idx, group_idx, EVT_MEMORY)
    tl.store(output_ptr + offsets, x, mask=mask)
    profiler_event_end(ctx, block_idx, group_idx, EVT_MEMORY)
    
    # --- Sync point (instant event) ---
    profiler_event_instant(ctx, block_idx, group_idx, EVT_SYNC)


def run_demo():
    """Run demo with specified enabled state."""
    
    # Configuration
    n_elements = 4096
    BLOCK_SIZE = 256
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_groups = 1  # One group per block for this simple demo
    max_events_per_group = 100
    
    print("Configuration:")
    print(f"  Elements: {n_elements}")
    print(f"  Block size: {BLOCK_SIZE}")
    print(f"  Num blocks: {num_blocks}")
    print(f"  Num groups: {num_groups}")
    print(f"  Max events per group: {max_events_per_group}")
    print()
    
    # Create profiler
    print("Creating profiler...")
    profiler = SmProfiler(
        num_blocks=num_blocks,
        num_groups=num_groups,
        max_events_per_group=max_events_per_group,
        enabled=True,
    )
    
    # Register event types
    profiler.register_event(EVT_COMPUTE, "compute")
    profiler.register_event(EVT_MEMORY, "memory")
    profiler.register_event(EVT_SYNC, "sync")
    
    # Get buffer
    buffer = profiler.get_buffer()
    
    # Create output tensor
    output = torch.zeros(n_elements, dtype=torch.float32, device="cuda")
    
    # Launch kernel
    print("Launching kernel...")
    demo_kernel[(num_blocks,)](
        buffer,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_GROUPS=num_groups,
    )
    
    # Synchronize
    torch.cuda.synchronize()
    print("Kernel completed")
    print()
    
    # Print stats
    profiler.print_stats()
    print()
    
    # Export trace
    profiler.export("triton_profile.json")


def main():
    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    run_demo()
    print("To view: Open triton_profile.json in Chrome chrome://tracing or https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
