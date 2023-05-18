#ifndef VM_FRAME_H
#define VM_FRAME_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <filesys/file.h>

void f_init(size_t user_page_limit);
void *f_get_p(bool *need_eviction);
void f_free_p(void *kpage,bool write_back);
bool f_load_p(void *kpage,bool in_swap,struct file* f,size_t position);
void f_set_p(void *kpage,uint32_t *pd, void *upage);
void f_evict_p(void* kpage);
#endif /**< vm/frame.h */