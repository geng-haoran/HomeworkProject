#ifndef VM_SWAP_H
#define VM_SWAP_H

#include <stddef.h>

void swap_init(void);
size_t swap_get_slot(void);
void swap_out(size_t slot,void *page);
void swap_in(size_t slot,void *page);
void swap_free_slot(size_t slot);
#endif /**< vm/swap.h */