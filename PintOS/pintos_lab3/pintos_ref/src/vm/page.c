
#include "vm/page.h"
#include <list.h>
#include <stdint.h>
#include <round.h>
#include <stddef.h>
#include "threads/palloc.h"
#include "threads/vaddr.h"
#include "threads/synch.h"
#include "vm/frame.h"
#include "vm/swap.h"

/* Align SPTE to 8 bytes. */
union SPTE {
    struct list_elem elem;              // List elem in empty_SPTE.
    struct {
        struct file *file;               // Which file to load the page.
        size_t position;                 // Position in file.
        struct semaphore ready_sema;     // Synch loading and evicting.
    } content;
};

// To make sure pointers to SPTEs are aligned. Its size must be rounded up to multiple of SPTE_ALIGN = 8.
#define SPTE_SIZE ROUND_UP(sizeof(union SPTE), SPTE_ALIGN)

static struct list empty_SPTE;          // List for empty STPEs.
static struct lock empty_SPTE_lock;     // Lock to protect empty_SPTE.

static union SPTE *get_empty_SPTE(void);

void supplemental_paging_init(void) {
    list_init(&empty_SPTE);
    lock_init(&empty_SPTE_lock);
}

/* Get an empty SPTE from empty_SPTE_list. Return the pointer to the new SPTE. */
void *create_SPTE_file(struct file *f, off_t ofs, uint32_t read_bytes) {
    union SPTE *new_SPTE = get_empty_SPTE();
    ASSERT(new_SPTE != NULL);
    // Initialize SPTE for a zero page.
    if (f == NULL || read_bytes == 0) {
        new_SPTE->content.file = NULL;
        new_SPTE->content.position = 0;
    }
    // Initialize SPTE for a file position. Make sure read_bytes > 0.
    else {
        new_SPTE->content.file = f;
        new_SPTE->content.position = ofs | (read_bytes - 1);
    }
    sema_init(&new_SPTE->content.ready_sema, 0);
    return new_SPTE;
}

/* Get an empty SPTE from empty_SPTE_list. Return the pointer to the new SPTE. */
void *create_SPTE_swap(size_t slot) {
    union SPTE *new_SPTE = get_empty_SPTE();
    ASSERT(new_SPTE != NULL);
    new_SPTE->content.file = NULL;
    new_SPTE->content.position = slot;
    sema_init(&new_SPTE->content.ready_sema, 0);
    return new_SPTE;
}

/* Package a pointer to SPTE, writable flag, and swap flag in a PTE format. */
uint32_t pack_SPTE(void *SPTE, bool rw, bool in_swap) {
    return (uint32_t)SPTE | (rw ? SPTE_W : 0) | (in_swap ? SPTE_S : 0);
}

/* Make SPTE ready for loading. */
void ready_SPTE(void *SPTE) {
    union SPTE *s = (union SPTE *)SPTE;
    sema_up(&s->content.ready_sema);
}

/* Free an occupied SPTE. Wait until this SPTE is ready. */
void free_SPTE(void *SPTE, bool in_swap) {
    union SPTE *s = (union SPTE *)SPTE;
    sema_down(&s->content.ready_sema);
    if (in_swap) {
        swap_free_slot(s->content.position);
    }
    destory_SPTE(s);
}

/* Load an initialized SPTE. Wait until this SPTE is ready. Return loading result. */
bool SPTE_load_p(void *SPTE, void *kpage, bool in_swap) {
    union SPTE *s = (union SPTE *)SPTE;
    bool success = true;
    sema_down(&s->content.ready_sema);
    if (!f_load_p(kpage, in_swap, s->content.file, s->content.position)) {
        success = false;
    }
    destory_SPTE(s);
    return success;
}

/* Destroy an SPTE. Put it into empty_SPTE_list. */
void destory_SPTE(void *s) {
    lock_acquire(&empty_SPTE_lock);
    list_push_back(&empty_SPTE, &((union SPTE *)s)->elem);
    lock_release(&empty_SPTE_lock);
}

/* Get a new SPTE out of the list. If the list is empty, request a new kernel page and cut it into SPTE_SIZE. */
static union SPTE *get_empty_SPTE() {
    lock_acquire(&empty_SPTE_lock);
    if (list_empty(&empty_SPTE)) {
        uint32_t kpage = (uint32_t)palloc_get_page(PAL_ASSERT);
        uint32_t bound = kpage + PGSIZE;
        for (uint32_t p = kpage; p < bound; p += SPTE_SIZE) {
            list_push_back(&empty_SPTE, &((union SPTE *)p)->elem);
        }
    }
    struct list_elem *new_elem = list_pop_front(&empty_SPTE);
    lock_release(&empty_SPTE_lock);
    return list_entry(new_elem, union SPTE, elem);
}
