#ifndef VM_FRAME_H
#define VM_FRAME_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <filesys/file.h>

void frame_init(size_t user_page_limit);
void *frame_get_page(bool *need_eviction);
void frame_free_page(void *kpage,bool write_back);
bool frame_load_page(void *kpage,bool in_swap,struct file* f,size_t position);
void frame_set_page(void *kpage,uint32_t *pd, void *upage);
void frame_evict_page(void* kpage);
#endif /**< vm/frame.h */