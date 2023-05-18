#include "vm/frame.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <round.h>
#include <list.h>
#include <string.h>
#include "threads/loader.h"
#include "threads/vaddr.h"
#include "threads/palloc.h"
#include "threads/thread.h"
#include "threads/synch.h"
#include "userprog/pagedir.h"
#include "filesys/file.h"
#include "filesys/filesys_lock.h"
#include "vm/page.h"
#include "vm/swap.h"

static size_t user_page_limit;          // User page limit.
static void *find_evict(void);          // Select a frame to evict.
static void clock_next(void);
static void clock_prev(void);
/* Convert a kernel virtual address to its FTE. */
static struct list frame_list;          // List for all active FTEs.
static struct lock frame_list_lock;     // Lock to protect frame_list.
static struct list_elem *clock_ptr;     // For clock algorithm.

/* Frame table entry. */
struct FTE 
{
    struct list_elem elem;              // List element in frame_list.
    uint32_t *pdir;                      // Owner's page directory.
    void *upage;                         // Mapped user virtual address.
    struct file* file;                   // Loaded from which file.
    size_t position;                     // Loaded from which position.
    void *evict_SPTE;                    // Synch loading and evicting.
};

static struct FTE* KP_FTE(void *kpage); // kpage to frame table entry.
static struct FTE *frame_table;         // Array contains all FTEs.
/** Return KAPGE's corresponding FTE.*/
static struct FTE *KP_FTE(void *kpage)
{
    return &frame_table[user_page_no(kpage)];
}

/** Return the user page address of FTE E.*/
static void *find_evict(void)
{
    struct FTE *evic;
    lock_acquire(&frame_list_lock);
    if(clock_ptr==NULL)
        clock_ptr=list_begin(&frame_list);
    while(true)
    {
        evic=list_entry(clock_ptr,struct FTE,elem);
        clock_next();
        if(pagedir_is_accessed(evic->pdir,evic->upage))
            pagedir_set_accessed(evic->pdir,evic->upage,false);
        else
            break;
    }
    list_remove(&evic->elem);
    lock_release(&frame_list_lock);
    return user_page_addr(evic-frame_table);
}

/** Move CLOCK_PTR to the next element.*/
static void clock_next(void)
{
    ASSERT(lock_held_by_current_thread(&frame_list_lock));
    if(clock_ptr!=list_rbegin(&frame_list))
        clock_ptr=list_next(clock_ptr);
    else
        clock_ptr=list_begin(&frame_list);
}

/* Convert a kernel virtual address to its FTE. */
void f_init(size_t user_page_limit) {
    uint8_t *fstart = ptov(1024 * 1024);
    uint8_t *fend = ptov(init_ram_pages * PGSIZE);
    size_t free_pages = (fend - fstart) / PGSIZE;
    size_t user_pages = (fend - fstart) / PGSIZE / 2;
    if (user_page_limit < user_pages) {
        user_pages = user_page_limit;
    }
    size_t page_cnt = DIV_ROUND_UP(user_pages * sizeof(struct FTE), PGSIZE);
    frame_table = (struct FTE*) palloc_get_multiple(PAL_ASSERT, page_cnt);
    list_init(&frame_list);
    lock_init(&frame_list_lock);
    clock_ptr = NULL;
}

/* Free a user page. */
bool f_load_p(void *kpage, bool in_swap, struct file* f, size_t position) {
    struct FTE *e = KP_FTE(kpage);
    if (!in_swap) {
        // A zero page or a page from filesys.
        e->file = f;
        e->position = position;
        off_t offset = position & ~PGMASK;
        size_t read_bytes = f == NULL ? 0 : (position & PGMASK) + 1;
        size_t zero_bytes = PGSIZE - read_bytes;
        if (f != NULL) {
            lock_acquire(&filesys_lock);
            bool success = file_read_at(f, kpage, read_bytes, offset) == (int) read_bytes;
            lock_release(&filesys_lock);
            if (!success) {
                return false;
            }
        }
        memset((void *)((uint32_t) kpage + read_bytes), 0, zero_bytes);
    }
    else {
        // A page from swap file.
        e->file = NULL;
        e->position = 0;
        swap_in(position, kpage);
    }
    return true;
}

/* Request a user page. Select a page to evict if all user pages are occupied. */
void *f_get_p(bool *need_eviction) {
    *need_eviction = false;
    void *kpage = palloc_get_page(PAL_USER);
    if (kpage == NULL) {
        kpage = find_evict(); // Find a frame to evict.
        struct FTE* evic = KP_FTE(kpage);
        bool rw = pagedir_is_writable(evic->pdir, evic->upage);
        if (evic->file != NULL && !pagedir_is_dirty(evic->pdir, evic->upage)) {
            void *new_SPTE = create_SPTE_file(evic->file, evic->position & ~PGMASK, (evic->position & PGMASK) + 1);
            pagedir_set_SPTE(evic->pdir, evic->upage, new_SPTE, rw, false);
            *need_eviction = false;
            ready_SPTE(new_SPTE);
        }
        else if (evic->file != NULL && file_can_write(evic->file)) {
            void *new_SPTE = create_SPTE_file(evic->file, evic->position & ~PGMASK, (evic->position & PGMASK) + 1);
            pagedir_set_SPTE(evic->pdir, evic->upage, new_SPTE, rw, false);
            *need_eviction = true;
            evic->evict_SPTE = new_SPTE;
        }
        else {
            size_t slot = swap_get_slot();
            void *new_SPTE = create_SPTE_swap(slot);
            pagedir_set_SPTE(evic->pdir, evic->upage, new_SPTE, rw, true);
            *need_eviction = true;
            evic->file = NULL;
            evic->position = slot;
            evic->evict_SPTE = new_SPTE;
        }
    }
    return kpage;
}

/* Free a user page. Write back if necessary. */
void f_free_p(void *kpage, bool write_back) {
    struct FTE *e = KP_FTE(kpage);

    lock_acquire(&filesys_lock);
    if (write_back && e->file != NULL && pagedir_is_dirty(e->pdir, e->upage) && file_can_write(e->file)) {
        off_t offset = e->position & ~PGMASK;
        size_t read_bytes = (e->position & PGMASK) + 1;
        file_write_at(e->file, kpage, read_bytes, offset);
    }
    lock_release(&filesys_lock);

    pagedir_clear_page(e->pdir, e->upage);
    lock_acquire(&frame_list_lock);
    list_remove(&e->elem);
    lock_release(&frame_list_lock);
    palloc_free_page(kpage);
}

/* Write KPAGE to filesys or swap file. */
void f_evict_p(void *kpage) {
    struct FTE *evic = KP_FTE(kpage);
    if (evic->file == NULL) {
        // Write KPAGE to swap file.
        size_t slot = evic->position;
        swap_out(slot, kpage);
    }
    else {
        // Write KPAGE to filesys.
        off_t offset = evic->position & ~PGMASK;
        size_t read_bytes = (evic->position & PGMASK) + 1;
        lock_acquire(&filesys_lock);
        file_write_at(evic->file, kpage, read_bytes, offset);
        lock_release(&filesys_lock);
    }
    // Make corresponding SPTE ready to load.
    ready_SPTE(evic->evict_SPTE);
}

/* Set the user page KPAGE to UPAGE in PDIR. Must call frame_free_page() on KPAGE before. */
void f_set_p(void *kpage, uint32_t *pdir, void *upage) {
    struct FTE *e = KP_FTE(kpage);
    e->pdir = pdir;
    e->upage = upage;
    lock_acquire(&frame_list_lock);
    list_push_back(&frame_list, &e->elem);
    lock_release(&frame_list_lock);
}

/* Find a frame to evict. */
void *find_frame_evict(void) {
    lock_acquire(&frame_list_lock);
    struct list_elem *e = clock_ptr == NULL ? list_begin(&frame_list) : clock_ptr;
    while (true) {
        struct FTE *f = list_entry(e, struct FTE, elem);
        if (!pagedir_is_accessed(f->pdir, f->upage)) {
            clock_ptr = list_next(e);
            lock_release(&frame_list_lock);
        }
        pagedir_set_accessed(f->pdir, f->upage, false);
        e = list_next(e);
        if (e == list_end(&frame_list)) {
            e = list_begin(&frame_list);
        }
    }
}