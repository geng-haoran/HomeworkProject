#include "vm/swap.h"
#include <stddef.h>
#include <bitmap.h>
#include "devices/block.h"
#include "threads/vaddr.h"
#include "threads/synch.h"

/** Number of sectors in a slot. */
#define SECTOR_PER_SLOT (PGSIZE/BLOCK_SECTOR_SIZE)  

static struct block *swap_block;        /**< Block device for swap. */
static struct bitmap *swap_slots;       /**< Manage all slots. */
static struct lock swap_slots_lock;     /**< Lock to protect swap_slots. */

void swap_init(void)
{
    swap_block=block_get_role(BLOCK_SWAP);
    block_sector_t sectors=block_size(swap_block);
    swap_slots=bitmap_create(sectors/SECTOR_PER_SLOT);
    lock_init(&swap_slots_lock);
}

/** Request a empty slot for later writing. */
size_t swap_get_slot(void)
{
    lock_acquire(&swap_slots_lock);
    size_t slot=bitmap_scan_and_flip(swap_slots,0,1,false);
    lock_release(&swap_slots_lock);

    if(slot==BITMAP_ERROR)
        PANIC("swap_out: out of slots.");

    return slot;
}

/** Write PAGE into SLOT. SLOT must be a slot got through swap_get_slot().
*/
void swap_out(size_t slot,void *page)
{
    ASSERT(pg_ofs(page)==0);

    size_t start_sec=slot*SECTOR_PER_SLOT;
    for(size_t sec_idx=0;sec_idx<SECTOR_PER_SLOT;sec_idx++)
    {
        block_write(swap_block,start_sec+sec_idx,
                    (void*)((uint32_t)page+sec_idx*BLOCK_SECTOR_SIZE));
    }
}

/** Read from SLOT and write to PAGE. This function will free the slot when 
  writing finishes.*/
void swap_in(size_t slot,void *page)
{
    ASSERT(pg_ofs(page)==0);
    size_t start_sec=slot*SECTOR_PER_SLOT;
    for(size_t sec_idx=0;sec_idx<SECTOR_PER_SLOT;sec_idx++)
        block_read(swap_block,start_sec+sec_idx,
                    (void*)((uint32_t)page+sec_idx*BLOCK_SECTOR_SIZE));
    swap_free_slot(slot);
}

/** Free swap slot SLOT.
*/
void swap_free_slot(size_t slot)
{
    lock_acquire(&swap_slots_lock);
    bitmap_set(swap_slots,slot,false);
    lock_release(&swap_slots_lock);
}