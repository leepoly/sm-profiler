"""
SM Profiler for Triton - A simple profiling library for Triton kernels.

Usage:
    from sm_profiler import SmProfiler
    
    # Host side setup
    profiler = SmProfiler(num_blocks=4, num_groups=4, max_events_per_group=100)
    profiler.register_event(0, "compute")
    profiler.register_event(1, "memory")
    buffer = profiler.get_buffer()
    
    # In Triton kernel - use the inline functions
    from sm_profiler.triton_ops import profiler_event_start, profiler_event_end
    
    # After kernel
    profiler.export("trace.json")
"""

from .profiler import SmProfiler
from .triton_ops import (
    profiler_parse_buffer,
    profiler_event_start,
    profiler_event_end,
    profiler_event_instant,
)

__version__ = "0.1.0"
__all__ = [
    "SmProfiler",
    "profiler_parse_buffer",
    "profiler_event_start",
    "profiler_event_end",
    "profiler_event_instant",
]
