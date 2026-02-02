"""
SM Profiler - Host-side Python API for Triton kernel profiling.
"""

import torch
import json
from typing import Optional, Dict, List
from dataclasses import dataclass

# Constants - must match C version
SM_PROFILER_MAX_EVENT_TYPES = 32
ACTIVE_EVENT_ENTRY_SIZE = 4   # sizeof(SmProfilerActiveEventEntry) = 4 bytes
DEVICE_EVENT_SIZE = 40        # sizeof(SmProfilerDeviceEvent) = 40 bytes
HEADER_SIZE_UINT64 = 2        # 2 x uint64_t for header


@dataclass
class ProfilerEvent:
    """Parsed event from device buffer."""
    event_id: int
    event_no: int
    block_id: int
    group_idx: int
    sm_id: int
    event_type: int  # 0=range, 1=instant
    st_timestamp_ns: int
    en_timestamp_ns: int


class SmProfiler:
    """
    SM-level profiler for Triton kernels.
    
    Usage:
        profiler = SmProfiler(num_blocks=4, num_groups=4, max_events_per_group=100)
        profiler.register_event(0, "compute")
        profiler.register_event(1, "memory")
        buffer = profiler.get_buffer()
        
        # Pass buffer to kernel...
        
        profiler.export("trace.json")
    """
    
    def __init__(
        self,
        num_blocks: int,
        num_groups: int,
        max_events_per_group: int,
        device: str = "cuda"
    ):
        """
        Create a profiler buffer.
        
        Args:
            num_blocks: Number of CUDA blocks expected
            num_groups: Number of groups (warps) per block
            max_events_per_group: Maximum events per group (warp)
            device: PyTorch device (default: "cuda")
        """
        if num_blocks <= 0 or num_groups <= 0 or max_events_per_group <= 0:
            raise ValueError("All parameters must be positive")
        
        self.num_blocks = num_blocks
        self.num_groups = num_groups
        self.max_events_per_group = max_events_per_group
        self.device = device
        
        # Event names (host-side only)
        self.event_names: Dict[int, str] = {}
        
        # Calculate buffer layout
        self._calc_layout()
        
        # Allocate buffer
        self._allocate_buffer()
        
        # Initialize buffer
        self.init_buffer()
    
    def _calc_layout(self):
        """Calculate buffer layout offsets."""
        # Counters: uint32_t[num_blocks * num_groups]
        counters_bytes = self.num_blocks * self.num_groups * 4
        self._counters_size_uint64 = (counters_bytes + 7) // 8
        
        # Active event offset (in uint64_t units)
        self._active_event_offset = HEADER_SIZE_UINT64 + self._counters_size_uint64
        
        # Active table: [num_blocks][num_groups][MAX_EVENT_TYPES] * 4 bytes
        active_bytes = self.num_blocks * self.num_groups * SM_PROFILER_MAX_EVENT_TYPES * ACTIVE_EVENT_ENTRY_SIZE
        self._active_table_size_uint64 = (active_bytes + 7) // 8
        
        # Event data offset (in uint64_t units)
        self._event_data_offset = self._active_event_offset + self._active_table_size_uint64
        
        # Event data: [num_blocks][num_groups][max_events_per_group] * 40 bytes
        event_data_bytes = self.num_blocks * self.num_groups * self.max_events_per_group * DEVICE_EVENT_SIZE
        self._event_data_size_uint64 = (event_data_bytes + 7) // 8
        
        # Total size in uint64_t
        self._total_size_uint64 = self._event_data_offset + self._event_data_size_uint64
    
    def _allocate_buffer(self):
        """Allocate device buffer."""
        self._buffer = torch.zeros(self._total_size_uint64, dtype=torch.int64, device=self.device)
    
    def init_buffer(self):
        """Initialize/clear buffer before kernel launch."""
        # Clear everything
        self._buffer.zero_()
        
        # Write header
        # header0: [num_blocks (32-bit) | num_groups (32-bit)]
        # header1: [max_events_per_group (32-bit) | reserved (32-bit)]
        header0 = (self.num_blocks << 32) | self.num_groups
        header1 = (self.max_events_per_group << 32)
        
        self._buffer[0] = header0
        self._buffer[1] = header1
    
    def register_event(self, event_no: int, name: str):
        """
        Register an event type name.
        
        Args:
            event_no: Event type ID (0 to SM_PROFILER_MAX_EVENT_TYPES-1)
            name: Human-readable name for this event type
        """
        if event_no < 0 or event_no >= SM_PROFILER_MAX_EVENT_TYPES:
            raise ValueError(f"event_no must be 0-{SM_PROFILER_MAX_EVENT_TYPES-1}")
        self.event_names[event_no] = name
    
    def get_buffer(self) -> torch.Tensor:
        """Get the device buffer tensor to pass to kernels."""
        return self._buffer
    
    def get_buffer_ptr(self):
        """Get raw pointer for use in kernels (for tl.load/tl.store)."""
        return self._buffer.data_ptr()
    
    def _parse_events(self) -> List[ProfilerEvent]:
        """Parse events from device buffer."""
        # Copy to CPU
        buffer_cpu = self._buffer.cpu()
        buffer_bytes = buffer_cpu.numpy().view('uint8')
        
        # Parse header
        header0 = int(buffer_cpu[0].item())
        header1 = int(buffer_cpu[1].item())
        num_blocks = (header0 >> 32) & 0xFFFFFFFF
        num_groups = header0 & 0xFFFFFFFF
        max_events_per_group = (header1 >> 32) & 0xFFFFFFFF
        
        # Get counters
        counters_offset = HEADER_SIZE_UINT64 * 8  # bytes
        counters = buffer_bytes[counters_offset:counters_offset + num_blocks * num_groups * 4].view('uint32')
        
        # Get event data base offset
        event_data_offset_bytes = self._event_data_offset * 8
        
        events = []
        for b in range(num_blocks):
            for g in range(num_groups):
                counter_idx = b * num_groups + g
                event_count = min(int(counters[counter_idx]), max_events_per_group)
                
                for e in range(event_count):
                    # Calculate event offset
                    group_offset = (b * num_groups + g) * max_events_per_group
                    event_offset = event_data_offset_bytes + (group_offset + e) * DEVICE_EVENT_SIZE
                    
                    # Parse event (40 bytes packed)
                    ev_bytes = buffer_bytes[event_offset:event_offset + DEVICE_EVENT_SIZE]
                    
                    # Parse fields
                    event_id = int.from_bytes(ev_bytes[0:4], 'little')
                    event_no = int.from_bytes(ev_bytes[4:8], 'little')
                    block_id = int.from_bytes(ev_bytes[8:12], 'little')
                    group_idx = int.from_bytes(ev_bytes[12:16], 'little')
                    sm_id = int.from_bytes(ev_bytes[16:20], 'little')
                    event_type = int.from_bytes(ev_bytes[20:24], 'little')
                    st_timestamp = int.from_bytes(ev_bytes[24:32], 'little')
                    en_timestamp = int.from_bytes(ev_bytes[32:40], 'little')
                    
                    events.append(ProfilerEvent(
                        event_id=event_id,
                        event_no=event_no,
                        block_id=block_id,
                        group_idx=group_idx,
                        sm_id=sm_id,
                        event_type=event_type,
                        st_timestamp_ns=st_timestamp,
                        en_timestamp_ns=en_timestamp
                    ))
        
        return events
    
    def export(self, filename: str):
        """
        Export profiler data to Chrome Trace JSON format.
        
        Args:
            filename: Output JSON file path
        """
        events = self._parse_events()
        
        if not events:
            print("sm_profiler: No events recorded")
            return
        
        # Sort by start timestamp
        events.sort(key=lambda e: e.st_timestamp_ns)
        
        # Find min timestamp
        min_ts = events[0].st_timestamp_ns if events else 0
        
        # Build trace events
        trace_events = []
        for ev in events:
            event_name = self.event_names.get(ev.event_no, "unknown")
            ts_us = (ev.st_timestamp_ns - min_ts) / 1000.0
            tid = f"block_{ev.block_id}_group_{ev.group_idx}"
            
            if ev.event_type == 0:  # Range event
                # Begin
                trace_events.append({
                    "name": f"{event_name}_{ev.event_id}",
                    "ph": "B",
                    "ts": ts_us,
                    "pid": ev.sm_id,
                    "tid": tid,
                    "cat": "gpu"
                })
                # End
                end_ts_us = (ev.en_timestamp_ns - min_ts) / 1000.0
                trace_events.append({
                    "name": f"{event_name}_{ev.event_id}",
                    "ph": "E",
                    "ts": end_ts_us,
                    "pid": ev.sm_id,
                    "tid": tid,
                    "cat": "gpu"
                })
            else:  # Instant event
                trace_events.append({
                    "name": f"{event_name}_{ev.event_id}",
                    "ph": "i",
                    "ts": ts_us,
                    "pid": ev.sm_id,
                    "tid": tid,
                    "cat": "gpu",
                    "s": "t"
                })
        
        # Write JSON
        output = {
            "traceEvents": trace_events,
            "displayTimeUnit": "ns",
            "metadata": {
                "num_blocks": self.num_blocks,
                "num_groups": self.num_groups,
                "max_events_per_group": self.max_events_per_group,
                "total_events": len(events)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"sm_profiler: Exported {len(events)} events ({len(trace_events)} trace entries) to {filename}")
    
    def print_stats(self):
        """Print profiling statistics."""
        events = self._parse_events()
        
        if not events:
            print("No events recorded")
            return
        
        print(f"Total events: {len(events)}")
        
        # Count by event type
        from collections import Counter
        event_counts = Counter(self.event_names.get(e.event_no, f"event_{e.event_no}") for e in events)
        print("Events by type:")
        for name, count in event_counts.most_common():
            print(f"  {name}: {count}")
        
        # SM distribution
        sm_counts = Counter(e.sm_id for e in events)
        print(f"SMs used: {sorted(sm_counts.keys())}")
