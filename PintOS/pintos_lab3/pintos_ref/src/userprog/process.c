#include "userprog/process.h"
#include <debug.h>
#include <inttypes.h>
#include <round.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "userprog/gdt.h"
#include "userprog/pagedir.h"
#include "userprog/tss.h"
#include "filesys/directory.h"
#include "filesys/file.h"
#include "filesys/filesys.h"
#include "filesys/filesys_lock.h"
#include "threads/flags.h"
#include "threads/init.h"
#include "threads/interrupt.h"
#include "threads/palloc.h"
#include "threads/thread.h"
#include "threads/vaddr.h"
#include "threads/synch.h"
#include "threads/malloc.h"
#include "vm/page.h"
#include "vm/frame.h"
#define ALIGNMENT 4
/* Decrease PTR by BYTES. */
#define DECREASE_PTR(ptr,bytes)   ((ptr)=(void*)((char*)(ptr)-(bytes)))
/* Put a 4 bytes VALUE at PTR. */
#define PUT_4BYTES(ptr,value)     (*(uint32_t*)(ptr)=(uint32_t)(value))
/* Round PTR down to a multiple of ALIGNMENT. */ 
#define ALIGN(ptr)          DECREASE_PTR(ptr,(uint32_t)(ptr)%ALIGNMENT)
/* Stack overflow? */
#define OVERFLOW(cur_esp,init_esp)  \
                           ((char*)(init_esp)-(char*)(cur_esp)>=PGSIZE)
/* Push a 4 bytes VALUE on stack PTR,but check before push. 
  Return whether push succeeds. */
#define TEST_AND_PUSH(ptr,value,init_esp)          \
        ({                                         \
          bool success=true;                       \
          DECREASE_PTR(ptr,4);                     \
          if(OVERFLOW(ptr,init_esp))               \
            success=false;                         \
          else                                     \
            PUT_4BYTES(ptr,value);                 \
          success;                                 \
        })
static thread_func start_process NO_RETURN;
static bool load (const char *cmdline, void (**eip) (void), void **esp);

/** Starts a new thread running a user program loaded from
   CMD_LINE.  The new thread may be scheduled (and may even exit)
   before process_execute() returns. Parent thread will be blocked
   until it know if loading succeed.
   Returns the new process's thread id, or TID_ERROR if the thread 
   cannot be created or loading failed. */
tid_t
process_execute (const char *cmd_line) 
{
  struct thread *cur=thread_current();
  char *cmd_copy;
  tid_t tid;
  char file_name[16]={'\0'};  /* A buffer to copy file name. */
  int c_idx=0,f_idx=0,len;    /* File name can't exceed 14 characters. */
  /* Make a copy of CMD_LINE.
     Otherwise there's a race between the caller and load(). */
  cmd_copy = palloc_get_page (0);
  if (cmd_copy == NULL)
    return TID_ERROR;
  strlcpy (cmd_copy, cmd_line, PGSIZE);
  /* Copy file name. */
  len=strlen(cmd_line);
  while(c_idx<len&&cmd_line[c_idx]==' ')
    c_idx++;
  for(;c_idx<len&&f_idx<15&&cmd_line[c_idx]!=' ';c_idx++,f_idx++)
    file_name[f_idx]=cmd_line[c_idx];
  file_name[f_idx]='\0';
  /* Prepare for a waiting. */
  sema_init(&cur->child_load_sema,0);
  cur->child_load_success=false;
  /* Create a new thread to execute CMD_LINE. */
  tid = thread_create (file_name, PRI_DEFAULT, start_process, cmd_copy);
  if (tid == TID_ERROR)
  {
    palloc_free_page (cmd_copy); 
    return TID_ERROR;
  }
  /* Wait on child_load_sema if thread_create success. */
  sema_down(&cur->child_load_sema);
  return cur->child_load_success?tid:TID_ERROR;
}

/** A thread function that loads a user process and starts it
   running. */
static void
start_process (void *cmd_line_)
{
  struct thread *cur=thread_current();
  char *cmd_line = cmd_line_;
  struct intr_frame if_;
  bool success;
  int argc=0, all_args_length, arg_length, i;
  char *file_name, *arg, *save_ptr;
  void *save_esp, *init_esp;
  char **argv;
  /* Get file name. */
  all_args_length=strlen(cmd_line)+1;
  arg=strtok_r(cmd_line," ",&save_ptr);
  file_name=arg;
  ASSERT(file_name!=NULL);
  /* Initialize interrupt frame and load executable. */
  memset (&if_, 0, sizeof if_);
  if_.gs = if_.fs = if_.es = if_.ds = if_.ss = SEL_UDSEG;
  if_.cs = SEL_UCSEG;
  if_.eflags = FLAG_IF | FLAG_MBS;
  lock_acquire(&filesys_lock);
  success = load (file_name, &if_.eip, &if_.esp);
  lock_release(&filesys_lock);
  /* If load failed, quit. */
  if (!success) 
    goto loading_failed;
  /** Pass arguments. */
  /* Use save_esp to push argument strings, 
    if_.esp to push the address of each string. */
  init_esp=save_esp=if_.esp;
  DECREASE_PTR(if_.esp,all_args_length);
  /* Word-aligned. */
  ALIGN(if_.esp);
  /* Make argc[argv] a null pointer */
  if(!TEST_AND_PUSH(if_.esp,0,init_esp))
    goto loading_failed;
  /* Copy each string on stack and push the address 
    in left-to-right order. */
  do
  {
    argc++;
    arg_length=strlen(arg)+1;
    DECREASE_PTR(save_esp,arg_length);
    strlcpy(save_esp,arg,arg_length);
    if(!TEST_AND_PUSH(if_.esp,save_esp,init_esp))
      goto loading_failed;
  } while ((arg=strtok_r(NULL," ",&save_ptr))!=NULL);
  argv=(char**)if_.esp;
  /* Reverse the argument vector to right-to-left order. */
  for(i=0;i<argc/2;i++)
  {
    save_ptr=argv[i];
    argv[i]=argv[argc-1-i];
    argv[argc-1-i]=save_ptr;
  }
  if(!TEST_AND_PUSH(if_.esp,argv,init_esp)||  /* Push argv. */
     !TEST_AND_PUSH(if_.esp,argc,init_esp)||  /* Push argc. */
     !TEST_AND_PUSH(if_.esp,0,init_esp))      /* Push return address. */
     goto loading_failed;
  /* Loading success! */
  palloc_free_page (cmd_line);
  cur->parent->child_load_success=true;
  sema_up(&cur->parent->child_load_sema);

  /* Start the user process by simulating a return from an
     interrupt, implemented by intr_exit (in
     threads/intr-stubs.S).  Because intr_exit takes all of its
     arguments on the stack in the form of a `struct intr_frame',
     we just point the stack pointer (%esp) to our stack frame
     and jump to it. */
  asm volatile ("movl %0, %%esp; jmp intr_exit" : : "g" (&if_) : "memory");
  NOT_REACHED ();

loading_failed:
  palloc_free_page (cmd_line);
  cur->child_msg->is_terminated=true;
  cur->child_msg->saved_exit_status=-1;
  sema_up(&cur->child_msg->terminated_sema);
  cur->parent->child_load_success=false;
  sema_up(&cur->parent->child_load_sema);
  thread_exit ();
  NOT_REACHED ();
}

/** Waits for thread TID to die and returns its exit status.  If
   it was terminated by the kernel (i.e. killed due to an
   exception), returns -1.  If TID is invalid or if it was not a
   child of the calling process, or if process_wait() has already
   been successfully called for the given TID, returns -1
   immediately, without waiting.

   This function will be implemented in problem 2-2.  For now, it
   does nothing. */
int
process_wait (tid_t child_tid) 
{
  struct child* c=get_child_thread(child_tid);
  int exit_status;
  if(c==NULL)
    return -1;
  /* Wait for the child to exit. */
  sema_down(&c->terminated_sema);
  /* Child should have exited, get its exit status. */
  ASSERT(c->is_terminated==true);
  exit_status=c->saved_exit_status;
  /* Remove it from child_list. */
  list_remove(&c->elem);
  free(c);
  return exit_status;
}

/** Free the current process's resources. */
void
process_exit (void)
{
  struct thread *cur = thread_current ();
  uint32_t *pd;

  /* Destroy the current process's page directory and switch back
     to the kernel-only page directory. */
  pd = cur->pagedir;
  if (pd != NULL) 
    {
      /* Correct ordering here is crucial.  We must set
         cur->pagedir to NULL before switching page directories,
         so that a timer interrupt can't switch back to the
         process page directory.  We must activate the base page
         directory before destroying the process's page
         directory, or our active page directory will be one
         that's been freed (and cleared). */
      cur->pagedir = NULL;
      pagedir_activate (NULL);
      pagedir_destroy (pd);
    }
}

/** Sets up the CPU for running user code in the current
   thread.
   This function is called on every context switch. */
void
process_activate (void)
{
  struct thread *t = thread_current ();

  /* Activate thread's page tables. */
  pagedir_activate (t->pagedir);

  /* Set thread's kernel stack for use in processing
     interrupts. */
  tss_update ();
}

/** We load ELF binaries.  The following definitions are taken
   from the ELF specification, [ELF1], more-or-less verbatim.  */

/** ELF types.  See [ELF1] 1-2. */
typedef uint32_t Elf32_Word, Elf32_Addr, Elf32_Off;
typedef uint16_t Elf32_Half;

/** For use with ELF types in printf(). */
#define PE32Wx PRIx32   /**< Print Elf32_Word in hexadecimal. */
#define PE32Ax PRIx32   /**< Print Elf32_Addr in hexadecimal. */
#define PE32Ox PRIx32   /**< Print Elf32_Off in hexadecimal. */
#define PE32Hx PRIx16   /**< Print Elf32_Half in hexadecimal. */

/** Executable header.  See [ELF1] 1-4 to 1-8.
   This appears at the very beginning of an ELF binary. */
struct Elf32_Ehdr
  {
    unsigned char e_ident[16];
    Elf32_Half    e_type;
    Elf32_Half    e_machine;
    Elf32_Word    e_version;
    Elf32_Addr    e_entry;
    Elf32_Off     e_phoff;
    Elf32_Off     e_shoff;
    Elf32_Word    e_flags;
    Elf32_Half    e_ehsize;
    Elf32_Half    e_phentsize;
    Elf32_Half    e_phnum;
    Elf32_Half    e_shentsize;
    Elf32_Half    e_shnum;
    Elf32_Half    e_shstrndx;
  };

/** Program header.  See [ELF1] 2-2 to 2-4.
   There are e_phnum of these, starting at file offset e_phoff
   (see [ELF1] 1-6). */
struct Elf32_Phdr
  {
    Elf32_Word p_type;
    Elf32_Off  p_offset;
    Elf32_Addr p_vaddr;
    Elf32_Addr p_paddr;
    Elf32_Word p_filesz;
    Elf32_Word p_memsz;
    Elf32_Word p_flags;
    Elf32_Word p_align;
  };

/** Values for p_type.  See [ELF1] 2-3. */
#define PT_NULL    0            /**< Ignore. */
#define PT_LOAD    1            /**< Loadable segment. */
#define PT_DYNAMIC 2            /**< Dynamic linking info. */
#define PT_INTERP  3            /**< Name of dynamic loader. */
#define PT_NOTE    4            /**< Auxiliary info. */
#define PT_SHLIB   5            /**< Reserved. */
#define PT_PHDR    6            /**< Program header table. */
#define PT_STACK   0x6474e551   /**< Stack segment. */

/** Flags for p_flags.  See [ELF3] 2-3 and 2-4. */
#define PF_X 1          /**< Executable. */
#define PF_W 2          /**< Writable. */
#define PF_R 4          /**< Readable. */

static bool setup_stack (void **esp);
static bool validate_segment (const struct Elf32_Phdr *, struct file *);
static bool load_segment (struct file *file, off_t ofs, uint8_t *upage,
                          uint32_t read_bytes, uint32_t zero_bytes,
                          bool writable);

/** Loads an ELF executable from FILE_NAME into the current thread.
   Stores the executable's entry point into *EIP
   and its initial stack pointer into *ESP.
   Returns true if successful, false otherwise. */
bool
load (const char *file_name, void (**eip) (void), void **esp) 
{
  struct thread *t = thread_current ();
  struct Elf32_Ehdr ehdr;
  struct file *file = NULL;
  off_t file_ofs;
  bool success = false;
  int i;

  /* Allocate and activate page directory. */
  t->pagedir = pagedir_create ();
  if (t->pagedir == NULL) 
    goto done;
  process_activate ();

  /* Open executable file. */
  file = filesys_open (file_name);
  if (file == NULL) 
    {
      printf ("load: %s: open failed\n", file_name);
      goto done; 
    }
  file_deny_write(file);
  t->executable=file;
  /* Read and verify executable header. */
  if (file_read (file, &ehdr, sizeof ehdr) != sizeof ehdr
      || memcmp (ehdr.e_ident, "\177ELF\1\1\1", 7)
      || ehdr.e_type != 2
      || ehdr.e_machine != 3
      || ehdr.e_version != 1
      || ehdr.e_phentsize != sizeof (struct Elf32_Phdr)
      || ehdr.e_phnum > 1024) 
    {
      printf ("load: %s: error loading executable\n", file_name);
      goto done; 
    }

  /* Read program headers. */
  file_ofs = ehdr.e_phoff;
  for (i = 0; i < ehdr.e_phnum; i++) 
    {
      struct Elf32_Phdr phdr;

      if (file_ofs < 0 || file_ofs > file_length (file))
        goto done;
      file_seek (file, file_ofs);

      if (file_read (file, &phdr, sizeof phdr) != sizeof phdr)
        goto done;
      file_ofs += sizeof phdr;
      switch (phdr.p_type) 
        {
        case PT_NULL:
        case PT_NOTE:
        case PT_PHDR:
        case PT_STACK:
        default:
          /* Ignore this segment. */
          break;
        case PT_DYNAMIC:
        case PT_INTERP:
        case PT_SHLIB:
          goto done;
        case PT_LOAD:
          if (validate_segment (&phdr, file)) 
            {
              bool writable = (phdr.p_flags & PF_W) != 0;
              uint32_t file_page = phdr.p_offset & ~PGMASK;
              uint32_t mem_page = phdr.p_vaddr & ~PGMASK;
              uint32_t page_offset = phdr.p_vaddr & PGMASK;
              uint32_t read_bytes, zero_bytes;
              if (phdr.p_filesz > 0)
                {
                  /* Normal segment.
                     Read initial part from disk and zero the rest. */
                  read_bytes = page_offset + phdr.p_filesz;
                  zero_bytes = (ROUND_UP (page_offset + phdr.p_memsz, PGSIZE)
                                - read_bytes);
                }
              else 
                {
                  /* Entirely zero.
                     Don't read anything from disk. */
                  read_bytes = 0;
                  zero_bytes = ROUND_UP (page_offset + phdr.p_memsz, PGSIZE);
                }
              if (!load_segment (file, file_page, (void *) mem_page,
                                 read_bytes, zero_bytes, writable))
                goto done;
            }
          else
            goto done;
          break;
        }
    }

  /* Set up stack. */
  if (!setup_stack (esp))
    goto done;

  /* Start address. */
  *eip = (void (*) (void)) ehdr.e_entry;

  success = true;

 done:
  /* We arrive here whether the load is successful or not. */
  return success;
}

/** load() helpers. */

static bool install_spte (void *upage, void *spte, bool writable);
/** Checks whether PHDR describes a valid, loadable segment in
   FILE and returns true if so, false otherwise. */
static bool
validate_segment (const struct Elf32_Phdr *phdr, struct file *file) 
{
  /* p_offset and p_vaddr must have the same page offset. */
  if ((phdr->p_offset & PGMASK) != (phdr->p_vaddr & PGMASK)) 
    return false; 

  /* p_offset must point within FILE. */
  if (phdr->p_offset > (Elf32_Off) file_length (file)) 
    return false;

  /* p_memsz must be at least as big as p_filesz. */
  if (phdr->p_memsz < phdr->p_filesz) 
    return false; 

  /* The segment must not be empty. */
  if (phdr->p_memsz == 0)
    return false;
  
  /* The virtual memory region must both start and end within the
     user address space range. */
  if (!is_user_vaddr ((void *) phdr->p_vaddr))
    return false;
  if (!is_user_vaddr ((void *) (phdr->p_vaddr + phdr->p_memsz)))
    return false;

  /* The region cannot "wrap around" across the kernel virtual
     address space. */
  if (phdr->p_vaddr + phdr->p_memsz < phdr->p_vaddr)
    return false;

  /* Disallow mapping page 0.
     Not only is it a bad idea to map page 0, but if we allowed
     it then user code that passed a null pointer to system calls
     could quite likely panic the kernel by way of null pointer
     assertions in memcpy(), etc. */
  if (phdr->p_vaddr < PGSIZE)
    return false;

  /* It's okay. */
  return true;
}

/** Loads a segment starting at offset OFS in FILE at address
   UPAGE.  In total, READ_BYTES + ZERO_BYTES bytes of virtual
   memory are initialized, as follows:

        - READ_BYTES bytes at UPAGE must be read from FILE
          starting at offset OFS.

        - ZERO_BYTES bytes at UPAGE + READ_BYTES must be zeroed.

   The pages initialized by this function must be writable by the
   user process if WRITABLE is true, read-only otherwise.

   Return true if successful, false if a memory allocation error
   or disk read error occurs. */
static bool
load_segment (struct file *file, off_t ofs, uint8_t *upage,
              uint32_t read_bytes, uint32_t zero_bytes, bool writable) 
{
  ASSERT ((read_bytes + zero_bytes) % PGSIZE == 0);
  ASSERT (pg_ofs (upage) == 0);
  ASSERT (ofs % PGSIZE == 0);

  while (read_bytes > 0 || zero_bytes > 0) 
    {
      /* Calculate how to fill this page.
         We will read PAGE_READ_BYTES bytes from FILE
         and zero the final PAGE_ZERO_BYTES bytes. */
      size_t page_read_bytes = read_bytes < PGSIZE ? read_bytes : PGSIZE;
      size_t page_zero_bytes = PGSIZE - page_read_bytes;
      /* Create a new SPTE contains file position. */
      void *new_spte=create_spte_file(file,ofs,page_read_bytes);
      if(!install_spte(upage,new_spte,writable))
      {
        destory_spte(new_spte);
        return false;
      }
      /* Make new SPTE ready for loading. */
      ready_spte(new_spte);
      /* Advance. */
      read_bytes -= page_read_bytes;
      zero_bytes -= page_zero_bytes;
      upage += PGSIZE;
      ofs+=PGSIZE;
    }
  return true;
}

/** Create a minimal stack by mapping a zeroed page at the top of
   user virtual memory. */
static bool
setup_stack (void **esp) 
{
  bool success = false;
  void *new_spte=create_spte_file(NULL,0,0);
  success=install_spte(((uint8_t *) PHYS_BASE) - PGSIZE,new_spte,true);
  if(success)
  {
    *esp=PHYS_BASE;
    ready_spte(new_spte);
  }
  else
    destory_spte(new_spte);
  return success;
}

/** Add a mapping form current thread's virtual page upage to a SPTE.
*/
static bool 
install_spte (void *upage, void *spte, bool writable)
{
  struct thread *t=thread_current();

  return (pagedir_get_page(t->pagedir,upage)==NULL
          && pagedir_set_spte(t->pagedir,upage,spte,writable,false));
}
