# CXL Memory Pool Backend for HiCache

This document describes how to use CXL Memory Pool (DAX device) as the L3 KV cache backend for SGLang.

## Overview

The CXL Memory Pool backend uses a DAX (Direct Access) device (typically `/dev/dax1.0`) as persistent memory storage for KV cache. This backend implements a simple hash-based key-value store on top of the DAX device using memory-mapped I/O.

## Prerequisites

1. **DAX Device**: A DAX device must be available on the system. Typically this is `/dev/dax1.0` for CXL memory pools.
2. **Permissions**: The process must have read/write access to the DAX device.

## Configuration

### Environment Variables

- `SGLANG_CXL_DAX_DEVICE`: Path to the DAX device (default: `/dev/dax1.0`)
- `SGLANG_CXL_DAX_DEVICE_SIZE`: Size of the DAX device in bytes (required if auto-detection fails)

Example:
```bash
export SGLANG_CXL_DAX_DEVICE=/dev/dax1.0
export SGLANG_CXL_DAX_DEVICE_SIZE=8589934592  # 8GB in bytes
```

**Note**: DAX devices are character devices and don't report their size via standard file operations. 
The backend will try to auto-detect the size using `ndctl`, but if that fails, you must set 
`SGLANG_CXL_DAX_DEVICE_SIZE` explicitly.

### Usage in SGLang

The backend can be enabled by specifying `cxl_memory_pool` as the storage backend when configuring HiCache.

Example configuration:
```python
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory

storage_config = HiCacheStorageConfig(
    tp_rank=0,
    tp_size=1,
    is_mla_model=False,
    is_page_first_layout=True,
    model_name="your-model-name",
)

# Create CXL Memory Pool backend
backend = StorageBackendFactory.create_backend(
    "cxl_memory_pool",
    storage_config,
    mem_pool_host=your_mem_pool_host,
)
```

## Implementation Details

### Storage Layout

The DAX device is organized as follows:

```
[Header: 1KB]
  - Magic number (4 bytes)
  - Version (4 bytes)
  - Number of entries (8 bytes)
  - Free offset pointer (8 bytes)
  - Reserved space

[Entry Table: MAX_ENTRIES * 52 bytes]
  Each entry contains:
  - Key hash (SHA256, 32 bytes)
  - Key length (4 bytes)
  - Value offset (8 bytes)
  - Value size (8 bytes)

[Data Area: Remaining space]
  - Actual KV cache data
```

### Features

- **Hash-based lookup**: Uses SHA256 hash of keys for fast lookup
- **Linear entry table**: Simple linear search through entry table (suitable for moderate number of entries)
- **Automatic initialization**: Initializes header on first use
- **Key suffixing**: Automatically adds model-specific suffix to keys for multi-model support

### Limitations

- Maximum entries: 1,000,000 (configurable via `MAX_ENTRIES`)
- Linear search: Entry lookup uses linear search, which may be slow for large numbers of entries
- No eviction: The current implementation does not implement eviction policies
- No concurrent access protection: Not thread-safe for concurrent writes

## Performance Considerations

- The DAX device provides byte-addressable persistent memory access
- Memory-mapped I/O allows direct access without system call overhead
- For better performance with large numbers of entries, consider implementing a hash table or B-tree structure

## Troubleshooting

### Device Not Found

If you see `FileNotFoundError: DAX device /dev/dax1.0 not found`:

1. Check if the DAX device exists:
   ```bash
   ls -l /dev/dax*
   ```

2. Verify device permissions:
   ```bash
   ls -l /dev/dax1.0
   ```

3. Check if the device is properly configured in the system

### Insufficient Space

If you see `ValueError: Device size is too small`:

- The device must be at least `HEADER_SIZE + MAX_ENTRIES * ENTRY_SIZE` bytes
- Current minimum: 1KB + 1,000,000 * 52 bytes ≈ 52 GB

### Permission Denied

If you see permission errors:

- Ensure the process has read/write access to the DAX device
- You may need to run with appropriate permissions or add the user to the device group

