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

/** A frame table entry(FTE) records infos about a user page.
  All FTEs are arranged in an array and correspond to user 
  pages one by one. E.g. FTE that indexs N always contains infos
  for user page N, which is the Nth page from user_pool.base. 
  FTEs are allocated when Pintos boots and won't be freed until
  Pintos shuts down. 

  If a user page is in use, its corresponding FTE will be put in 
  a list called frame_list. Otherwise, the FTE only exists in array.

  Most of time, an active FTE saves virtual page in PD and UPAGE.
  FILE and POSITION save file infos and EVICT_SPTE is unused.
    FILE == NULL if a zero page or a page from swap file.
    POSITION is ignored in this case.

    FILE != NULL if a nonzero page from a filesys file. POSITION 
    holds which position in FILE to get the page.
    POSITION is formated as offset|(read_bytes-1)
  
  When the user page is about to evict, FTE is temporarily used to 
  contain infos on later eviction.
    FILE == NULL indicates this page will be evicted to filesys.
    POSITION holds which position in file to write.

    FILE != NULL indicates this page will be evicted to swap file.
    POSITION holds which swap slot to write.

    EVICT_SPTE is a pointer to SPTE. Must call ready_spte() when eviction
    finishes.
*/
struct fte{
    struct list_elem elem;              /**< List element in frame_list. */
    uint32_t *pd;                       /**< Mapped pagedir. */
    void *upage;                        /**< Mapped user virtual page. */
    struct file* file;                  /**< Loaded from which file. */
    size_t position;                    /**< Position in file. */
    void *evict_spte;                   /**< Synch loading and evicting. */
};

static struct list frame_list;          /**< List for all active FTEs. */
static struct lock frame_list_lock;     /**< Lock to protect frame_list. */
static struct fte *frame_table;         /**< Array contains all FTEs. */
static struct list_elem *clock_ptr;     /**< For clock algorithm. */

static struct fte* kpage_to_fte(void *kpage);
static void *select_frame_to_evict(void);
static void clock_next(void);

/** Allocate all FTEs when Pintos boots. 
*/
void 
frame_init(size_t user_page_limit){
    uint8_t *free_start = ptov (1024 * 1024);
    uint8_t *free_end = ptov (init_ram_pages * PGSIZE);
    size_t free_pages = (free_end - free_start) / PGSIZE;
    size_t user_pages = free_pages / 2;
    if(user_page_limit<user_pages)
        user_pages=user_page_limit;
    size_t page_cnt= DIV_ROUND_UP(user_pages*sizeof(struct fte),PGSIZE);
    frame_table=(struct fte*)palloc_get_multiple(PAL_ASSERT,page_cnt);
    list_init(&frame_list);
    lock_init(&frame_list_lock);
    clock_ptr=NULL;
}

/** Request a user page. Select a page to evict if all user pages are 
  occupied. Set NEED_EVICTION to true if it should make I/Os later. 
  Return the page's kernel virtual address.
*/
void *
frame_get_page(bool *need_eviction)
{
    *need_eviction=false;
    void *kpage=palloc_get_page(PAL_USER);
    if(kpage==NULL)
    {   
        /* Select a frame to evict. */
        kpage=select_frame_to_evict();
        /* Notify the frame's old owner that its page has been evicted. */
        struct fte* evic=kpage_to_fte(kpage);
        bool rw=pagedir_is_writable(evic->pd,evic->upage);
        if(evic->file!=NULL&&!pagedir_is_dirty(evic->pd,evic->upage))
        {
            /* Unmodified pages are evicted to filesys. */
            void *new_spte=create_spte_file(evic->file,
                        evic->position&~PGMASK,
                        (evic->position&PGMASK)+1);
            /* Reset old owner's pte, later access from old owner will 
            cause a page fault. */
            pagedir_set_spte(evic->pd,evic->upage,new_spte,rw,false);
            /* Needn't to make addition I/Os. */
            *need_eviction=false;
            /* SPTE is a ready spte. */
            ready_spte(new_spte);
        }
        else if(evic->file!=NULL&&file_can_write(evic->file))
        {
            /* If a modified page can write back, evict it to filesys. */
            void *new_spte=create_spte_file(evic->file,
                        evic->position&~PGMASK,
                        (evic->position&PGMASK)+1);
            /* Reset old owner's pte, later access from old owner will 
            cause a page fault. */
            pagedir_set_spte(evic->pd,evic->upage,new_spte,rw,false);
            /* Need to make I/Os later. SPTE isn't ready. */
            *need_eviction=true;
            /* Save eviction infos in FTE. */
            evic->evict_spte=new_spte;
        }
        else
        {
            /* Other pages are evicted to swap file. */
            size_t slot=swap_get_slot();
            void *new_spte=create_spte_swap(slot);
            /* Reset old owner's pte, later access from old owner will 
            cause a page fault. */
            pagedir_set_spte(evic->pd,evic->upage,new_spte,rw,true);
            /* Need to make I/Os later. SPTE isn't ready. */
            *need_eviction=true;
            /* Save eviction infos in FTE. */
            evic->file=NULL;
            evic->position=slot;
            evic->evict_spte=new_spte;
        }
    }
    return kpage;
}

/** Load a user page and save file infos in FTE. Return loading result. 
*/
bool
frame_load_page(void *kpage,bool in_swap,struct file* f,size_t position)
{
    struct fte *e=kpage_to_fte(kpage);
    if(!in_swap)
    {
        /* A zero page or a page from filesys. */
        e->file=f;
        e->position=position;
        off_t offset=position&~PGMASK;
        size_t read_bytes=f==NULL?0:(position&PGMASK)+1;
        size_t zero_bytes=PGSIZE-read_bytes;
        if(f!=NULL)
        {
            lock_acquire(&filesys_lock);
            bool success=file_read_at(f,kpage,read_bytes,offset)
                                                        ==(int)read_bytes;
            lock_release(&filesys_lock);
            if(!success)
                return false;
        }
        memset((void *)((uint32_t)kpage+read_bytes),0,zero_bytes);
    }
    else
    {
        /* A page from swap file. */
        e->file=NULL;
        e->position=0;
        swap_in(position,kpage);
    }
    return true;
}

/** Free the user page KPAGE. Must call frame_set_page() on KPAGE before.
   If WRITE_BACK is true, do necessary writings for writable files.
*/
void 
frame_free_page(void *kpage,bool write_back)
{
    struct fte *e=kpage_to_fte(kpage);

    lock_acquire(&filesys_lock);
    if(write_back&&e->file!=NULL&&pagedir_is_dirty(e->pd,e->upage)
        &&file_can_write(e->file))
    {
        off_t offset=e->position&~PGMASK;
        size_t read_bytes=(e->position&PGMASK)+1;
        file_write_at(e->file,kpage,read_bytes,offset);
    }
    lock_release(&filesys_lock);

    pagedir_clear_page(e->pd,e->upage);
    lock_acquire(&frame_list_lock);
    list_remove(&e->elem);
    lock_release(&frame_list_lock);
    palloc_free_page(kpage);
}

/** Set PD and UPAGE field in KPAGE's FTE. Mark KPAGE occupied and ready for
  eviction by inserting its FTE into list.
*/
void 
frame_set_page(void *kpage,uint32_t *pd, void *upage)
{
    struct fte *e=kpage_to_fte(kpage);
    e->pd=pd;
    e->upage=upage;
    lock_acquire(&frame_list_lock);
    list_push_back(&frame_list,&e->elem);
    lock_release(&frame_list_lock);
}

/** Write KPAGE to filesys or swap file.
*/
void
frame_evict_page(void *kpage)
{
    struct fte *evic=kpage_to_fte(kpage);
    if(evic->file==NULL)
    {
        /* Write KPAGE to swap file. */
        size_t slot=evic->position;
        swap_out(slot,kpage);
    }
    else
    {
        /* Write KAPGE to filesys. */
        off_t offset=evic->position&~PGMASK;
        size_t read_bytes=(evic->position&PGMASK)+1;
        lock_acquire(&filesys_lock);
        file_write_at(evic->file,kpage,read_bytes,offset);
        lock_release(&filesys_lock);
    }
    /* Make corresponding spte ready to load. */
    ready_spte(evic->evict_spte);
}

/** Return KAPGE's corresponding FTE.
*/
static struct fte *
kpage_to_fte(void *kpage)
{
    return &frame_table[user_page_no(kpage)];
}

/** Select a occupied user page for eviction. Clock algorithm is implemented.
*/
static void *
select_frame_to_evict(void)
{
    struct fte *evic;
    lock_acquire(&frame_list_lock);
    if(clock_ptr==NULL)
        clock_ptr=list_begin(&frame_list);
    while(true)
    {
        evic=list_entry(clock_ptr,struct fte,elem);
        clock_next();
        if(pagedir_is_accessed(evic->pd,evic->upage))
            pagedir_set_accessed(evic->pd,evic->upage,false);
        else
            break;
    }
    list_remove(&evic->elem);
    lock_release(&frame_list_lock);
    return user_page_addr(evic-frame_table);
}

/** Move CLOCK_PTR to the next element.
*/
static void 
clock_next(void)
{
    ASSERT(lock_held_by_current_thread(&frame_list_lock));
    if(clock_ptr!=list_rbegin(&frame_list))
        clock_ptr=list_next(clock_ptr);
    else
        clock_ptr=list_begin(&frame_list);
}