#ifndef MEMORY_POOL_HPP
#define MEMORY_POOL_HPP

#include "pch.hpp"

class SimpleMemoryPool {
private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
        
        Block(size_t s) : size(s), in_use(false) {
            ptr = malloc(s);
        }
        
        ~Block() {
            if (ptr) free(ptr);
        }
    };
    
    std::vector<Block> blocks_;
    std::mutex mutex_;
    size_t default_block_size_;
    
public:
    SimpleMemoryPool(size_t block_size = 1024 * 1024 * 10) 
        : default_block_size_(block_size) {
        // Pre-allocate some blocks
        blocks_.reserve(16);
        for (int i = 0; i < 4; ++i) {
            blocks_.emplace_back(default_block_size_);
        }
    }
    
    ~SimpleMemoryPool() {
        // Destructor will clean up all blocks
    }
    
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find available block
        for (auto& block : blocks_) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }
        
        // Create new block
        if (blocks_.size() < 32) {
            size_t new_size = std::max(size, default_block_size_);
            blocks_.emplace_back(new_size);
            blocks_.back().in_use = true;
            return blocks_.back().ptr;
        }
        
        // Fallback to regular allocation
        return malloc(size);
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& block : blocks_) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
        
        // Fallback to regular deallocation
        free(ptr);
    }
    
    static SimpleMemoryPool& getInstance() {
        static SimpleMemoryPool instance;
        return instance;
    }
};

#endif // MEMORY_POOL_HPP 