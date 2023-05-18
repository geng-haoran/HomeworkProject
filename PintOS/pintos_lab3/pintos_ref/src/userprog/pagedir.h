#ifndef USERPROG_PAGEDIR_H
#define USERPROG_PAGEDIR_H

#include <stdbool.h>
#include <stdint.h>
void pagedir_lock_init(void);
uint32_t *pagedir_create (void);
void pagedir_destroy (uint32_t *pd);
bool pagedir_set_page (uint32_t *pd, void *upage, void *kpage, bool rw);
void *pagedir_get_page (uint32_t *pd, const void *upage);
bool pagedir_is_unmapped(uint32_t *pd, const void *upage);
bool pagedir_set_SPTE (uint32_t *pd, void *upage, void *SPTE, 
                                        bool rw,bool in_swap);
bool pagedir_demand_page (uint32_t *pd, void *upage);
bool pagedir_map_page(uint32_t *pd,void *addr,void *file,uint32_t length);
void pagedir_unmap_page(uint32_t *pd,void *addr,uint32_t length);
void pagedir_clear_page (uint32_t *pd, void *upage);
bool pagedir_is_dirty (uint32_t *pd, const void *upage);
void pagedir_set_dirty (uint32_t *pd, const void *upage, bool dirty);
bool pagedir_is_accessed (uint32_t *pd, const void *upage);
void pagedir_set_accessed (uint32_t *pd, const void *upage, bool accessed);
void pagedir_activate (uint32_t *pd);
bool pagedir_is_writable(uint32_t *pd,const void *upage);
#endif /**< userprog/pagedir.h */
