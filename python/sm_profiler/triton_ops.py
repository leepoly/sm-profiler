"""
SM Profiler - Triton device-side operations.

All operations use int64 arithmetic to avoid overflow issues.
"""

import triton
import triton.language as tl

__all__ = [
    "profiler_parse_buffer",
    "profiler_init",
    "profiler_event_start",
    "profiler_event_end",
    "profiler_event_instant",
]


@triton.jit
def _as_i64(x):
    """Promote any value to int64 by adding to zero."""
    return tl.zeros([], dtype=tl.int64) + x


@triton.jit
def profiler_parse_buffer(buffer_ptr):
    """
    Parse profiler buffer header and return pointers/info.
    """
    header0 = tl.load(buffer_ptr)
    header1 = tl.load(buffer_ptr + 1)
    
    num_blocks = (header0 >> 32) & 0x7FFFFFFF  # Use smaller mask to stay in int32 range
    num_groups = header0 & 0x7FFFFFFF
    max_events_per_group = (header1 >> 32) & 0x7FFFFFFF
    
    counters_ptr = buffer_ptr + 2
    
    counters_bytes = num_blocks * num_groups * 4
    counters_size_uint64 = (counters_bytes + 7) // 8
    active_offset = _as_i64(2) + counters_size_uint64
    active_ptr = buffer_ptr + active_offset
    
    active_bytes = num_blocks * num_groups * 32 * 4  # 32 = MAX_EVENT_TYPES
    active_size_uint64 = (active_bytes + 7) // 8
    event_offset = active_offset + active_size_uint64
    events_ptr = buffer_ptr + event_offset
    
    return num_blocks, num_groups, max_events_per_group, counters_ptr, active_ptr, events_ptr


@triton.jit
def _read_int32_lo(base_ptr, idx):
    """Read lower 32 bits of int64 at index."""
    val = tl.load(base_ptr + idx)
    return val & 0x7FFFFFFF  # Mask to avoid overflow


@triton.jit
def _read_int32_hi(base_ptr, idx):
    """Read upper 32 bits of int64 at index."""
    val = tl.load(base_ptr + idx)
    return (val >> 32) & 0x7FFFFFFF


@triton.jit
def _write_int32_lo(base_ptr, idx, value):
    """Write to lower 32 bits of int64 at index."""
    old_val = tl.load(base_ptr + idx)
    upper = (old_val >> 32) << 32
    new_val = upper + _as_i64(value & 0x7FFFFFFF)
    tl.store(base_ptr + idx, new_val)


@triton.jit
def _write_int32_hi(base_ptr, idx, value):
    """Write to upper 32 bits of int64 at index."""
    old_val = tl.load(base_ptr + idx)
    lower = old_val & 0x7FFFFFFF
    new_val = (_as_i64(value & 0x7FFFFFFF) << 32) + lower
    tl.store(base_ptr + idx, new_val)


@triton.jit
def _get_counter(counters_ptr, num_groups, block_idx, group_idx):
    """Get counter value for (block, group)."""
    counter_idx = _as_i64(block_idx) * num_groups + _as_i64(group_idx)
    int64_idx = counter_idx // 2
    is_upper = counter_idx % 2
    
    if is_upper:
        return _read_int32_hi(counters_ptr, int64_idx)
    else:
        return _read_int32_lo(counters_ptr, int64_idx)


@triton.jit
def _set_counter(counters_ptr, num_groups, block_idx, group_idx, value):
    """Set counter value for (block, group)."""
    counter_idx = _as_i64(block_idx) * num_groups + _as_i64(group_idx)
    int64_idx = counter_idx // 2
    is_upper = counter_idx % 2
    
    if is_upper:
        _write_int32_hi(counters_ptr, int64_idx, value)
    else:
        _write_int32_lo(counters_ptr, int64_idx, value)


@triton.jit
def _get_active_entry(active_ptr, num_groups, block_idx, group_idx, event_no):
    """Get active event entry."""
    entry_idx = (_as_i64(block_idx) * num_groups + _as_i64(group_idx)) * 32 + _as_i64(event_no)
    int64_idx = entry_idx // 2
    is_upper = entry_idx % 2
    
    if is_upper:
        return _read_int32_hi(active_ptr, int64_idx)
    else:
        return _read_int32_lo(active_ptr, int64_idx)


@triton.jit
def _set_active_entry(active_ptr, num_groups, block_idx, group_idx, event_no, value):
    """Set active event entry."""
    entry_idx = (_as_i64(block_idx) * num_groups + _as_i64(group_idx)) * 32 + _as_i64(event_no)
    int64_idx = entry_idx // 2
    is_upper = entry_idx % 2
    
    if is_upper:
        _write_int32_hi(active_ptr, int64_idx, value)
    else:
        _write_int32_lo(active_ptr, int64_idx, value)


@triton.jit
def _get_event_ptr(events_ptr, num_groups, max_events, block_idx, group_idx, event_id):
    """Get int64 pointer to event data. Event is 5 x int64 = 40 bytes."""
    group_offset = (_as_i64(block_idx) * num_groups + _as_i64(group_idx)) * max_events
    event_int64_offset = (group_offset + _as_i64(event_id)) * 5
    return events_ptr + event_int64_offset


@triton.jit
def _store_event(ev_ptr, event_id, event_no, block_idx, group_idx, sm_id, event_type, timestamp):
    """Store event data as 5 x int64."""
    # [0]: event_id (lo) | event_no (hi)
    val0 = _as_i64(event_id) + (_as_i64(event_no) << 32)
    tl.store(ev_ptr + 0, val0)
    
    # [1]: block_id (lo) | group_idx (hi)
    val1 = _as_i64(block_idx) + (_as_i64(group_idx) << 32)
    tl.store(ev_ptr + 1, val1)
    
    # [2]: sm_id (lo) | type (hi)
    val2 = _as_i64(sm_id) + (_as_i64(event_type) << 32)
    tl.store(ev_ptr + 2, val2)
    
    # [3]: st_timestamp
    tl.store(ev_ptr + 3, timestamp)
    
    # [4]: en_timestamp (0 initially)
    tl.store(ev_ptr + 4, tl.zeros([], dtype=tl.int64))


@triton.jit
def _profiler_event_start_impl(
    counters_ptr, active_ptr, events_ptr,
    num_groups, max_events,
    block_idx, group_idx, event_no
):
    """Internal implementation: Start a range event."""
    # Check if already active
    active_event_id = _get_active_entry(active_ptr, num_groups, block_idx, group_idx, event_no)
    if active_event_id != 0:
        return
    
    # Get and increment counter
    event_id = _get_counter(counters_ptr, num_groups, block_idx, group_idx)
    _set_counter(counters_ptr, num_groups, block_idx, group_idx, event_id + 1)
    
    # Check buffer full
    if event_id >= max_events:
        return
    
    # Get event pointer
    ev_ptr = _get_event_ptr(events_ptr, num_groups, max_events, block_idx, group_idx, event_id)
    
    # Get timestamp and SM ID
    timestamp = tl.extra.cuda.globaltimer()
    sm_id = tl.extra.cuda.smid()
    
    # Write event
    _store_event(ev_ptr, event_id, event_no, block_idx, group_idx, sm_id, 0, timestamp)
    
    # Mark as active (event_id + 1)
    _set_active_entry(active_ptr, num_groups, block_idx, group_idx, event_no, event_id + 1)


@triton.jit
def _profiler_event_end_impl(
    active_ptr, events_ptr,
    num_groups, max_events,
    block_idx, group_idx, event_no
):
    """Internal implementation: End a range event."""
    # Check if active
    active_event_id = _get_active_entry(active_ptr, num_groups, block_idx, group_idx, event_no)
    if active_event_id == 0:
        return
    
    event_id = active_event_id - 1
    
    # Get event pointer
    ev_ptr = _get_event_ptr(events_ptr, num_groups, max_events, block_idx, group_idx, event_id)
    
    # Write end timestamp
    timestamp = tl.extra.cuda.globaltimer()
    tl.store(ev_ptr + 4, timestamp)
    
    # Mark as inactive
    _set_active_entry(active_ptr, num_groups, block_idx, group_idx, event_no, 0)


@triton.jit
def _profiler_event_instant_impl(
    counters_ptr, active_ptr, events_ptr,
    num_groups, max_events,
    block_idx, group_idx, event_no
):
    """Internal implementation: Record an instant event."""
    # Get and increment counter
    event_id = _get_counter(counters_ptr, num_groups, block_idx, group_idx)
    _set_counter(counters_ptr, num_groups, block_idx, group_idx, event_id + 1)
    
    # Check buffer full
    if event_id >= max_events:
        return
    
    # Get event pointer
    ev_ptr = _get_event_ptr(events_ptr, num_groups, max_events, block_idx, group_idx, event_id)
    
    # Get timestamp and SM ID
    timestamp = tl.extra.cuda.globaltimer()
    sm_id = tl.extra.cuda.smid()
    
    # Write event - type=1 for instant
    _store_event(ev_ptr, event_id, event_no, block_idx, group_idx, sm_id, 1, timestamp)


@triton.jit
def profiler_init(buffer_ptr):
    """
    Initialize profiler context. Call once at kernel start.
    Returns a tuple (counters_ptr, active_ptr, events_ptr, num_groups, max_events, enabled).
    
    Usage:
        ctx = profiler_init(buffer_ptr)
        if ctx[5]:  # check enabled
            profiler_event_start(ctx, block_idx, group_idx, event_no)
    """
    header1 = tl.load(buffer_ptr + 1)
    enabled = header1 & 0x7FFFFFFF
    
    num_blocks, num_groups, max_events_per_group, counters_ptr, active_ptr, events_ptr = \
        profiler_parse_buffer(buffer_ptr)
    
    return counters_ptr, active_ptr, events_ptr, num_groups, max_events_per_group, enabled


@triton.jit
def profiler_event_start(ctx, block_idx, group_idx, event_no):
    """
    Start a range event using context tuple.
    
    Args:
        ctx: Context tuple from profiler_init()
        block_idx: Block index (usually tl.program_id(0))
        group_idx: Group index (usually 0 or warp_id)
        event_no: Event type number
    """
    counters_ptr, active_ptr, events_ptr, num_groups, max_events, enabled = ctx
    if enabled == 0:
        return
    _profiler_event_start_impl(counters_ptr, active_ptr, events_ptr,
                              num_groups, max_events, block_idx, group_idx, event_no)


@triton.jit
def profiler_event_end(ctx, block_idx, group_idx, event_no):
    """
    End a range event using context tuple.
    
    Args:
        ctx: Context tuple from profiler_init()
        block_idx: Block index (usually tl.program_id(0))
        group_idx: Group index (usually 0 or warp_id)
        event_no: Event type number
    """
    counters_ptr, active_ptr, events_ptr, num_groups, max_events, enabled = ctx
    if enabled == 0:
        return
    _profiler_event_end_impl(active_ptr, events_ptr,
                            num_groups, max_events, block_idx, group_idx, event_no)


@triton.jit
def profiler_event_instant(ctx, block_idx, group_idx, event_no):
    """
    Record an instant event using context tuple.
    
    Args:
        ctx: Context tuple from profiler_init()
        block_idx: Block index (usually tl.program_id(0))
        group_idx: Group index (usually 0 or warp_id)
        event_no: Event type number
    """
    counters_ptr, active_ptr, events_ptr, num_groups, max_events, enabled = ctx
    if enabled == 0:
        return
    _profiler_event_instant_impl(counters_ptr, active_ptr, events_ptr,
                                num_groups, max_events, block_idx, group_idx, event_no)
