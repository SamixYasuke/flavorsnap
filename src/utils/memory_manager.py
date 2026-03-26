import gc
import torch
import psutil
import os
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Utilities for monitoring and managing system and GPU memory.
    Ensures PyTorch tensors and system caches are cleaned up efficiently.
    """
    @staticmethod
    def get_memory_info():
        """Returns current system memory and GPU memory usage (MB)."""
        process = psutil.Process(os.getpid())
        system_mb = process.memory_info().rss / (1024 * 1024)
        
        gpu_mb = 0
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            
        return {
            "system_mb": round(system_mb, 2),
            "gpu_mb": round(gpu_mb, 2),
        }

    @staticmethod
    def cleanup():
        """Force triggers garbage collection and clears unused GPU memory."""
        # 1. Clear Python objects
        gc.collect()
        
        # 2. Clear PyTorch CUDA Cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        logger.info(f"Memory cleanup performed: {MemoryManager.get_memory_info()}")

    @staticmethod
    def log_usage(tag=""):
        """Logs current memory usage with an optional tag."""
        usage = MemoryManager.get_memory_info()
        logger.info(f"Memory Usage [{tag}]: System {usage['system_mb']}MB | GPU {usage['gpu_mb']}MB")
