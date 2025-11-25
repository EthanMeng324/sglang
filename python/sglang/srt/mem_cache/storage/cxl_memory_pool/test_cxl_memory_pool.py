"""
Simple test for CXL Memory Pool backend.

Note: This test requires a DAX device to be available.
Set SGLANG_CXL_DAX_DEVICE environment variable to specify the device path.

To run this test, simply execute from this directory:
    python test_cxl_memory_pool.py
"""

import os
import sys

# Add the python directory to the path to enable imports
# Find the python directory relative to this file
# Path: python/sglang/srt/mem_cache/storage/cxl_memory_pool/test_cxl_memory_pool.py
# Need to go up 5 levels to reach python/
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

# Go up: cxl_memory_pool -> storage -> mem_cache -> srt -> sglang -> python
python_dir = os.path.abspath(os.path.join(
    current_dir,  # cxl_memory_pool
    "..",  # storage
    "..",  # mem_cache
    "..",  # srt
    "..",  # sglang
    "..",  # python
))

if os.path.exists(python_dir) and python_dir not in sys.path:
    sys.path.insert(0, python_dir)

import logging
import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.cxl_memory_pool.cxl_memory_pool import (
    CXLMemoryPool,
)

# Optional: Enable debug logging for troubleshooting
# logging.basicConfig(level=logging.DEBUG)

# Check if DAX device is available
dax_device = os.getenv("SGLANG_CXL_DAX_DEVICE", "/dev/dax1.0")
if not os.path.exists(dax_device):
    print(f"DAX device {dax_device} not found. Skipping test.")
    print("To run this test, ensure a DAX device is available or set SGLANG_CXL_DAX_DEVICE.")
    sys.exit(0)


def test_basic_operations():
    """Test basic get/set operations."""
    print("Testing basic operations...")

    # Create storage config
    storage_config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        is_mla_model=False,
        is_page_first_layout=True,
        model_name="test-model",
    )

    # Create backend
    backend = CXLMemoryPool(storage_config, dax_device=dax_device)

    # Test set and get
    key = "test_key_1"
    value = torch.randn(10, 20, dtype=torch.float16)
    
    # Clear before test to ensure fresh start
    backend.clear()

    # Set value
    result = backend.set(key, value)
    assert result, "Set operation failed"
    print(f"✓ Set operation successful for key: {key}")

    # Check exists
    exists = backend.exists(key)
    assert exists, "Exists check failed"
    print(f"✓ Exists check successful for key: {key}")

    # Get value
    target = torch.zeros_like(value)
    retrieved = backend.get(key, target)
    assert retrieved is not None, "Get operation failed"
    assert torch.allclose(value, retrieved, atol=1e-3), "Retrieved value mismatch"
    print(f"✓ Get operation successful for key: {key}")

    # Test batch operations
    keys = [f"test_key_{i}" for i in range(2, 5)]
    values = [torch.randn(10, 20, dtype=torch.float16) for _ in keys]

    # Batch set
    batch_set_result = backend.batch_set(keys, values)
    assert batch_set_result, "Batch set operation failed"
    print(f"✓ Batch set operation successful for {len(keys)} keys")

    # Batch exists
    batch_exists_count = backend.batch_exists(keys)
    assert batch_exists_count == len(keys), f"Batch exists failed: {batch_exists_count}/{len(keys)}"
    print(f"✓ Batch exists check successful: {batch_exists_count}/{len(keys)}")

    # Batch get
    targets = [torch.zeros_like(v) for v in values]
    batch_get_results = backend.batch_get(keys, targets)
    assert all(r is not None for r in batch_get_results), "Batch get operation failed"
    for i, (orig, ret) in enumerate(zip(values, batch_get_results)):
        assert torch.allclose(orig, ret, atol=1e-3), f"Retrieved value mismatch for key {keys[i]}"
    print(f"✓ Batch get operation successful for {len(keys)} keys")

    # Test clear
    backend.clear()
    exists_after_clear = backend.exists(key)
    assert not exists_after_clear, "Clear operation failed"
    print("✓ Clear operation successful")

    print("\nAll basic operations tests passed!")


def test_key_suffixing():
    """Test that keys are properly suffixed."""
    print("\nTesting key suffixing...")

    storage_config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=2,
        is_mla_model=False,
        is_page_first_layout=True,
        model_name="test-model",
    )

    backend = CXLMemoryPool(storage_config, dax_device=dax_device)

    key = "test_key"
    value = torch.randn(5, 10, dtype=torch.float16)

    backend.set(key, value)
    exists = backend.exists(key)
    assert exists, "Key with suffix should exist"
    print("✓ Key suffixing works correctly")

    # Test with different rank
    storage_config2 = HiCacheStorageConfig(
        tp_rank=1,
        tp_size=2,
        is_mla_model=False,
        is_page_first_layout=True,
        model_name="test-model",
    )
    backend2 = CXLMemoryPool(storage_config2, dax_device=dax_device)
    exists2 = backend2.exists(key)
    # Should not exist for different rank (different suffix)
    assert not exists2, "Key with different suffix should not exist"
    print("✓ Key isolation between different ranks works correctly")


if __name__ == "__main__":
    try:
        test_basic_operations()
        test_key_suffixing()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

