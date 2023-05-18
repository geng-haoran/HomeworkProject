#include "userprog/pagedir.h"
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <round.h>
#include "threads/init.h"
#include "threads/pte.h"
#include "threads/palloc.h"
#include "threads/synch.h"
#include "vm/page.h"
#include "vm/frame.h"
static uint32_t *active_pd (void);
static void invalidate_pagedir (uint32_t *);

/**< Lock for demanding pages. */
static struct lock pagedir_lock;   

/** Returns the current thread's page directory. */
void pagedir_lock_init(void)
{
  lock_init(&pagedir_lock);
}

/** Sets up the CPU's page directory for paged kernel virtual addresses. */
uint32_t *
pagedir_create (void) 
{
  uint32_t *pd = palloc_get_page (0);
  if (pd != NULL)
    memcpy (pd, init_page_dir, PGSIZE);
  return pd;
}

/** Installs page directory PD into the CPU. */
void pagedir_destroy (uint32_t *pd) 
{
  uint32_t *pde;

  if (pd == NULL)
    return;
  lock_acquire(&pagedir_lock);
  ASSERT (pd != init_page_dir);
  for (pde = pd; pde < pd + pd_no (PHYS_BASE); pde++)
    if (*pde & PTE_P) 
      {
        uint32_t *pt = pde_get_pt (*pde);
        uint32_t *pte;
        
        for (pte = pt; pte < pt + PGSIZE / sizeof *pte; pte++)
          if (*pte)
          { 
            if(*pte&PTE_P)
              f_free_p (pte_get_page (*pte),false);
            else
              free_SPTE((void *)(*pte&SPTE_MASK),*pte&SPTE_S);
          }
        palloc_free_page (pt);
      }
  palloc_free_page (pd);
  lock_release(&pagedir_lock);
}

/** Returns true if the PTE for virtual address VADDR in page
    directory PD exists, false otherwise. */
static uint32_t *lookup_page (uint32_t *pd, const void *vaddr, bool create)
{
  uint32_t *pt, *pde;

  ASSERT (pd != NULL);

  /* Shouldn't create new kernel virtual mappings. */
  ASSERT (!create || is_user_vaddr (vaddr));

  /* Check for a page table for VADDR.
     If one is missing, create one if requested. */
  pde = pd + pd_no (vaddr);
  if (*pde == 0) 
    {
      if (create)
        {
          pt = palloc_get_page (PAL_ZERO);
          if (pt == NULL) 
            return NULL; 
      
          *pde = pde_create (pt);
        }
      else
        return NULL;
    }

  /* Return the page table entry. */
  pt = pde_get_pt (*pde);
  return &pt[pt_no (vaddr)];
}

/** Returns the address of the page table entry for virtual address
    VADDR in page directory PD. */
bool pagedir_set_page (uint32_t *pd, void *upage, void *kpage, bool writable)
{
  uint32_t *pte;

  ASSERT (pg_ofs (upage) == 0);
  ASSERT (pg_ofs (kpage) == 0);
  ASSERT (is_user_vaddr (upage));
  ASSERT (vtop (kpage) >> PTSHIFT < init_ram_pages);
  ASSERT (pd != init_page_dir);

  pte = lookup_page (pd, upage, true);

  if (pte != NULL) 
    {
      ASSERT ((*pte & PTE_P) == 0);
      *pte = pte_create_user (kpage, writable);
      return true;
    }
  else
    return false;
}

/** Returns the address of the page table entry for virtual address
    VADDR in page directory PD. */
void *pagedir_get_page (uint32_t *pd, const void *uaddr) 
{
  uint32_t *pte;

  ASSERT (is_user_vaddr (uaddr));
  
  pte = lookup_page (pd, uaddr, false);
  if (pte != NULL && (*pte & PTE_P) != 0)
    return pte_get_page (*pte) + pg_ofs (uaddr);
  else
    return NULL;
}

/** Return true if UADDR in PD hasn't been mapped.*/
bool pagedir_is_unmapped(uint32_t *pd, const void *uaddr)
{
  uint32_t *pte;

  ASSERT (is_user_vaddr (uaddr));
  
  pte = lookup_page (pd, uaddr, false);
  return pte==NULL||*pte==0;
}

/** Add a mapping in page directory PD from user virtual page
  UPAGE to a SPTE with flags RW and IN_SWAP. Return true if success.
*/
bool pagedir_set_SPTE (uint32_t *pd,void *upage,void *SPTE,bool rw,bool in_swap)
{
  uint32_t *pte;

  ASSERT(pg_ofs(upage)==0);
  ASSERT(is_user_vaddr(upage));
  ASSERT(((uint32_t)SPTE&~SPTE_MASK)==0);

  pte=lookup_page(pd,upage,true);

  if(pte!=NULL)
  {
    *pte=pack_SPTE(SPTE,rw,in_swap);
    return true;
  }
  else
    return false;
}

/** Demand an unpresent page UPAGE in PD. Return true if success. */
bool pagedir_demand_page (uint32_t *pd, void *upage)
{
  uint32_t *pte;
  bool success=false;
  void *SPTE;
  bool rw,in_swap;
  void *kpage;
  bool need_eviction;
  /* Must be a user page. */
  if(!is_user_vaddr(upage))
    goto demand_done;

  lock_acquire(&pagedir_lock);
  pte=lookup_page(pd,upage,false);
  /* Check PTE is an entry for SPTE. */
  if(!(pte!=NULL&&*pte&&(*pte&PTE_P)==0))
  {
    lock_release(&pagedir_lock);
    goto demand_done;
  }
    
  /* Get each field out of PTE. */
  SPTE=(void *)(*pte&SPTE_MASK);
  rw=*pte&SPTE_W;
  in_swap=*pte&SPTE_S;

  /* Get a frame to contain user virtual page. */
  kpage=f_get_p(&need_eviction);
  ASSERT(kpage!=NULL);
  lock_release(&pagedir_lock);
  /* Do evictions. */
  if(need_eviction)
    f_evict_p(kpage);
  /* Load KPAGE. */
  if(!SPTE_load_p(SPTE,kpage,in_swap))
  {
    palloc_free_page(kpage);
    goto demand_done;
  }
  /* Modify our page table and global frame table */
  pagedir_set_page(pd,upage,kpage,rw);
  f_set_p(kpage,pd,upage);
  success=true;

demand_done:

  return success;
}

/** Map a file FILE contains LENGTH bytes from ADDR in PD.
  Return true if success. Mappings will fail if user memory has already
  been mapped.
*/
bool pagedir_map_page(uint32_t *pd,void *addr,void *file,uint32_t length)
{
  ASSERT(pg_ofs(addr)==0);

  uint32_t page_cnt=DIV_ROUND_UP(length,PGSIZE);
  bool success=true;
  off_t ofs=0;
  lock_acquire(&pagedir_lock);
  /* Check if user memory has been mapped. */
  for(uint32_t pg=0;pg<page_cnt;pg++)
  {
    if(!pagedir_is_unmapped(pd,(void *)((uint32_t)addr+pg*PGSIZE)))
    {
      success=false;
      break;
    }
  }
  if(!success)
    goto map_done;
  /* Make mappings through setting some SPTEs in page table. */
  while(length>0)
  {
    size_t read_bytes=length<PGSIZE?length:PGSIZE;
    void *new_SPTE=create_SPTE_file((struct file *) file,ofs,read_bytes);
    ready_SPTE(new_SPTE);
    pagedir_set_SPTE(pd,addr,new_SPTE,true,false);
    length-=read_bytes;
    ofs+=read_bytes;
    addr=(void *)((uint32_t)addr+PGSIZE);
  }
map_done:
  lock_release(&pagedir_lock);
  return success;
}

/** Unmap LENGTH bytes form ADDR in PD.
  Write back any modified page to file.
*/
void pagedir_unmap_page(uint32_t *pd,void *addr,uint32_t length)
{
  ASSERT(pg_ofs(addr)==0);

  uint32_t page_cnt=DIV_ROUND_UP(length,PGSIZE);
  lock_acquire(&pagedir_lock);
  for(uint32_t pg=0;pg<page_cnt;pg++)
  {
    uint32_t *pte=lookup_page(pd,(void *)((uint32_t)addr+pg*PGSIZE),false);
    ASSERT(*pte);
    if(*pte&PTE_P)
      f_free_p (pte_get_page (*pte),true);
    else
      free_SPTE((void *)(*pte&SPTE_MASK),*pte&SPTE_S);
    *pte=0;
  }
  lock_release(&pagedir_lock);
}

/** Marks user virtual page UPAGE "not present" in page
   directory PD.  Later accesses to the page will fault.  Other
   bits in the page table entry are preserved.
   UPAGE need not be mapped. */
void
pagedir_clear_page (uint32_t *pd, void *upage) 
{
  uint32_t *pte;

  ASSERT (pg_ofs (upage) == 0);
  ASSERT (is_user_vaddr (upage));

  pte = lookup_page (pd, upage, false);
  if (pte != NULL && (*pte & PTE_P) != 0)
    {
      *pte &= ~PTE_P;
      invalidate_pagedir (pd);
    }
}

/** Returns true if the PTE for virtual page VPAGE in PD is dirty,
   that is, if the page has been modified since the PTE was
   installed.
   Returns false if PD contains no PTE for VPAGE. */
bool
pagedir_is_dirty (uint32_t *pd, const void *vpage) 
{
  uint32_t *pte = lookup_page (pd, vpage, false);
  return pte != NULL && (*pte & PTE_D) != 0;
}

/** Set the dirty bit to DIRTY in the PTE for virtual page VPAGE
   in PD. */
void
pagedir_set_dirty (uint32_t *pd, const void *vpage, bool dirty) 
{
  uint32_t *pte = lookup_page (pd, vpage, false);
  if (pte != NULL) 
    {
      if (dirty)
        *pte |= PTE_D;
      else 
        {
          *pte &= ~(uint32_t) PTE_D;
          invalidate_pagedir (pd);
        }
    }
}

/** Returns true if the PTE for virtual page VPAGE in PD has been
   accessed recently, that is, between the time the PTE was
   installed and the last time it was cleared.  Returns false if
   PD contains no PTE for VPAGE. */
bool
pagedir_is_accessed (uint32_t *pd, const void *vpage) 
{
  uint32_t *pte = lookup_page (pd, vpage, false);
  return pte != NULL && (*pte & PTE_A) != 0;
}

/** Sets the accessed bit to ACCESSED in the PTE for virtual page
   VPAGE in PD. */
void
pagedir_set_accessed (uint32_t *pd, const void *vpage, bool accessed) 
{
  uint32_t *pte = lookup_page (pd, vpage, false);
  if (pte != NULL) 
    {
      if (accessed)
        *pte |= PTE_A;
      else 
        {
          *pte &= ~(uint32_t) PTE_A; 
          invalidate_pagedir (pd);
        }
    }
}

/** Loads page directory PD into the CPU's page directory base
   register. */
void
pagedir_activate (uint32_t *pd) 
{
  if (pd == NULL)
    pd = init_page_dir;

  /* Store the physical address of the page directory into CR3
     aka PDBR (page directory base register).  This activates our
     new page tables immediately.  See [IA32-v2a] "MOV--Move
     to/from Control Registers" and [IA32-v3a] 3.7.5 "Base
     Address of the Page Directory". */
  asm volatile ("movl %0, %%cr3" : : "r" (vtop (pd)) : "memory");
}

/** Return true if the PTE for virtual page VPAGE in PD is writable.
*/
bool 
pagedir_is_writable(uint32_t *pd,const void *vpage)
{
    uint32_t *pte = lookup_page (pd, vpage, false);
    return pte != NULL && (*pte & PTE_W) != 0;
}


/** Returns the currently active page directory. */
static uint32_t *
active_pd (void) 
{
  /* Copy CR3, the page directory base register (PDBR), into
     `pd'.
     See [IA32-v2a] "MOV--Move to/from Control Registers" and
     [IA32-v3a] 3.7.5 "Base Address of the Page Directory". */
  uintptr_t pd;
  asm volatile ("movl %%cr3, %0" : "=r" (pd));
  return ptov (pd);
}

/** Seom page table changes can cause the CPU's translation
   lookaside buffer (TLB) to become out-of-sync with the page
   table.  When this happens, we have to "invalidate" the TLB by
   re-activating it.

   This function invalidates the TLB if PD is the active page
   directory.  (If PD is not active then its entries are not in
   the TLB, so there is no need to invalidate anything.) */
static void
invalidate_pagedir (uint32_t *pd) 
{
  if (active_pd () == pd) 
    {
      /* Re-activating PD clears the TLB.  See [IA32-v3a] 3.12
         "Translation Lookaside Buffers (TLBs)". */
      pagedir_activate (pd);
    } 
}
