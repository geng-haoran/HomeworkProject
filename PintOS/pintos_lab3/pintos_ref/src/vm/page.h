#ifndef VM_PAGE_H
#define VM_PAGE_H
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <filesys/file.h>

/** 
   PTEs for present pages are structured as follows:
    31                                    12 11            1    0
   +----------------------------------------+----------------+-----+
   |           Physical  Address            |      Flags     | P=1 |
   +----------------------------------------+----------------+-----+

    PTEs for unpresent pages are structured as follows:
    (unpresent page = pages that exists but does not present)
    31                                         3    2     1     0
   +---------------------------------------------+-----+-----+-----+
   |              Pointer To SPTE                |  W  |  S  | P=0 |
   +---------------------------------------------+-----+-----+-----+
    (SPTE stands for supplemental page table entry)
   It is obvious that if a virtual page exists, its PTE can't be zero.
   Zero PTEs indicates pages that does not exist.
   Nonzero PTEs are devided into two groups by P flag.

   Here's the meanings for each field in a PTE as an index for an SPTE :
   Pointer To SPTE : a pointer points to an union SPTE, declared in page.c.
   W : W = 1 indicates this unpresent page is writable, otherwise it's not.
   S : S = 1 indicates this unpresent page is in swap file, otherwise it's 
       in file system.
*/

/* Some basic macros to get each field. */
#define SPTE_ALIGN 8                    /**< Alignment. */
#define SPTE_MASK ~(SPTE_ALIGN-1)       /**< SPTE Pointer. */
#define SPTE_W 0x4                      /**< Writable flag. */
#define SPTE_S 0x2                      /**< Swap flag. */


void supplemental_paging_init(void);
void *create_SPTE_file(struct file* f, off_t ofs, uint32_t read_bytes);
void *create_SPTE_swap(size_t slot);
uint32_t pack_SPTE(void *SPTE, bool rw, bool in_swap);
void ready_SPTE(void *SPTE);
void free_SPTE(void *SPTE, bool in_swap);
bool SPTE_load_page(void *SPTE, void *kpage, bool in_swap);
void destory_SPTE(void *SPTE);
#endif /**< vm/page.h */