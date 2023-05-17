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

/** Struct of a SPTE.
   All free SPTEs are arranged in a list called empty_spte_list.
   An occupied SPTE contain FILE and POSITION infos to load a page.
   FILE : 
        FILE == NULL if a zero page or a page from swap file; 
        FILE != NULL if a nonzero page from a filesys file.
   POSITION :
        if FILE == NULL, POSITION is ignored for a zero page and is set to 
        the slot contains the page for a page from swap file.
        if FILE !=NULL, POSITION is in format offset|(read_bytes-1), which
        contains both offset and read_bytes. 
   READY_SEMA :
        A thread can wait on this semaphore until the page is ready for 
        loading.
 */
union spte
{
    struct list_elem elem;              /**< List elem in empty_spte. */
    struct{
        struct file *file;              /**< Which file to load the page. */
        size_t position;                /**< Position in file. */
        struct semaphore ready_sema;    /**< Synch loading and evicting. */
    }content;
};
/* To make sure pointers to SPTEs are aligned. Its size must be rounded up to
  multiple of SPTE_ALIGN = 8.*/
#define SPTE_SIZE ROUND_UP(sizeof(union spte),SPTE_ALIGN)

static struct list empty_spte;          /**< List for empty STPEs. */
static struct lock empty_spte_lock;     /**< Lock to protect empty_spte. */

static union spte *get_empty_spte(void);

void 
supplemental_paging_init(void)
{
    list_init(&empty_spte);
    lock_init(&empty_spte_lock);
}

/** Create a new SPTE for a page from filesys or an zero page. A page from 
  filesys which loads zero byte from a file will be treated as a zero page
  from nowhere, since it won't write back and only can go into swap file.
  New SPTE isn't ready for loading. Caller must call ready_spte() later to
  make this SPTE ready for loading.
  Return the pointer to the new SPTE.
*/
void *
create_spte_file(struct file* f,off_t ofs,uint32_t read_bytes)
{
    union spte* new_spte=get_empty_spte();
    ASSERT(new_spte!=NULL);
    /* Initalize spte for a zero page. */
    if(f==NULL||read_bytes==0)
    {
        new_spte->content.file=NULL;
        new_spte->content.position=0;
    }
    /* Initalize spte for a file position. 
        Make sure read_bytes > 0.*/
    else
    {
        new_spte->content.file=f;
        new_spte->content.position=ofs|(read_bytes-1);
    }
    sema_init(&new_spte->content.ready_sema,0);
    return new_spte;
}

/** Create a new SPTE for a page from swap file. New SPTE isn't ready for 
  loading. Caller must call ready_spte() later to make this SPTE ready for
  loading.
  Return the pointer to the new SPTE.
*/
void *
create_spte_swap(size_t slot)
{
    union spte *new_spte=get_empty_spte();
    ASSERT(new_spte!=NULL);
    new_spte->content.file=NULL;
    new_spte->content.position=slot;
    sema_init(&new_spte->content.ready_sema,0);
    return new_spte;
}
/** Packge a pointer to SPTE, writable flag and swap flag in a PTE format.
*/
uint32_t 
pack_spte(void *spte,bool rw,bool in_swap)
{
    return (uint32_t)spte|(rw?SPTE_W:0)|(in_swap?SPTE_S:0);
}

/** Make SPTE ready for loading.
*/
void 
ready_spte(void *spte)
{
    union spte *s=(union spte *)spte;
    sema_up(&s->content.ready_sema);
}

/** Free an occupied SPTE. Wait until this SPTE is ready.
*/
void
free_spte(void *spte,bool in_swap)
{
    union spte *s=(union spte *)spte;
    sema_down(&s->content.ready_sema);
    if(in_swap)
        swap_free_slot(s->content.position);
    destory_spte(s);
}

/** Load an initialized SPTE. Wait until this SPTE is ready. Return loading
  result.
*/
bool
spte_load_page(void *spte,void *kpage,bool in_swap)
{
    union spte *s=(union spte *)spte;
    bool success=true;
    sema_down(&s->content.ready_sema);
    if(!frame_load_page(kpage,in_swap,s->content.file,s->content.position))
        success=false;
    destory_spte(s);
    return success;
}

/** Recycle an occupied SPTE. Caller must make sure this SPTE contains no
  useful infos.
*/
void 
destory_spte(void *s)
{
    lock_acquire(&empty_spte_lock);
    list_push_back(&empty_spte,&((union spte*)s)->elem);
    lock_release(&empty_spte_lock);
}

/** Get a new SPTE out of list. If the list is empty, request a new kernel
  page and cut it into SPTE_SIZE.
*/
static union spte * 
get_empty_spte()
{    
    lock_acquire(&empty_spte_lock);
    if(list_empty(&empty_spte))
    {
        uint32_t kpage=(uint32_t)palloc_get_page(PAL_ASSERT);
        uint32_t bound=kpage+PGSIZE;
        for(uint32_t p=kpage;p<bound;p+=SPTE_SIZE)
            list_push_back(&empty_spte,&((union spte*)p)->elem);
    }
    struct list_elem *new_elem=list_pop_front(&empty_spte);
    lock_release(&empty_spte_lock);
    return list_entry(new_elem,union spte,elem);
}