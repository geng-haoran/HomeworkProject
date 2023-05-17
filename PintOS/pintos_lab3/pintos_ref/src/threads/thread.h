#ifndef THREADS_THREAD_H
#define THREADS_THREAD_H

#include <debug.h>
#include <list.h>
#include <stdint.h>
#include "filesys/file.h"
#include "threads/synch.h"
/** States in a thread's life cycle. */
enum thread_status
  {
    THREAD_RUNNING,     /**< Running thread. */
    THREAD_READY,       /**< Not running but ready to run. */
    THREAD_BLOCKED,     /**< Waiting for an event to trigger. */
    THREAD_DYING        /**< About to be destroyed. */
  };

/** Thread identifier type.
   You can redefine this to whatever type you like. */
typedef int tid_t;
#define TID_ERROR ((tid_t) -1)          /**< Error value for tid_t. */

/** Thread priorities. */
#define PRI_MIN 0                       /**< Lowest priority. */
#define PRI_DEFAULT 31                  /**< Default priority. */
#define PRI_MAX 63                      /**< Highest priority. */

/** Thread's open file definition. */
typedef int fd_t;                       /**< File descriptor. */
#define FD_ERROR ((fd_t) -1)            /**< Error value for fd_t. */
typedef int fd_set_t;                   /**< Bitmap of opening fds. */
#define INIT_FD_SET 0x3                 /**< Initial fd set. */
#define MAX_FILE 8*sizeof(fd_set_t)     /**< Maximum files. */
#define USER_FD_MIN (fd_t)2             /**< Lowest fd user can open. */
#define USER_FD_MAX (fd_t)(MAX_FILE-1)  /**< Highest fd user can open. */
#define STDIN_FD 0                      /**< STDIN file descriptor. */
#define STDOUT_FD 1                     /**< STDOUT file descriptor. */
/** Three basic operation on a fd set. */
/* Is FD allocatable in FD_SET. */
#define IS_FREE_FD(fd_set,fd)    (!(((fd_set)>>(fd))&0x1))  
/* Add FD to FD_SET. */
#define ALLOCATE_FD(fd_set,fd)   (fd_set)|=(0x1<<(fd))
/* Remove FD from FD_SET. */
#define REMOVE_FD(fd_set,fd)     (fd_set)&=(~(0x1<<(fd)))
/** A thread's opening files struct. */ 
struct files_struct
{
   fd_set_t       fd_set;                 /**< Open fd set. */
   struct file    *fd_table[MAX_FILE];    /**< Open fd table. */
};

/** A child of thread T would save some variable in T's child_list, 
   in order to let T know if the child has exited or not. 
   Struct child is a list element in child_list. */
struct child
{
   struct thread *t;                      /**< Child thread struct. */
   tid_t tid;                             /**< Child tid. */
   bool is_terminated;                    /**< Child exit or not. */
   int saved_exit_status;                 /**< Saved exit status. */
   struct semaphore terminated_sema;      /**< Semaphore for wait. */
   struct list_elem elem;                 /**< List elem. */
};

/** Thread's mmap definition. */
typedef int mmapid_t;                     /**< Mmapid. */
#define INIT_MMAPID ((mmapid_t)0)         /**< Lowest mmapid. */
#define MMAPID_ERROR ((mmapid_t)-1)       /**< Error mmapid. */
/** A mapping struct. */
struct mmap_t
{
   struct list_elem elem;                 /**< List elem in mmap_list. */
   mmapid_t mmapid;                       /**< Mmapid. */
   struct file *f;                        /**< Mapping file. */
   void *addr;                            /**< Mapping start address. */
   off_t length;                          /**< Mapping total length. */
};
/** A kernel thread or user process.

   Each thread structure is stored in its own 4 kB page.  The
   thread structure itself sits at the very bottom of the page
   (at offset 0).  The rest of the page is reserved for the
   thread's kernel stack, which grows downward from the top of
   the page (at offset 4 kB).  Here's an illustration:

        4 kB +---------------------------------+
             |          kernel stack           |
             |                |                |
             |                |                |
             |                V                |
             |         grows downward          |
             |                                 |
             |                                 |
             |                                 |
             |                                 |
             |                                 |
             |                                 |
             |                                 |
             |                                 |
             +---------------------------------+
             |              magic              |
             |                :                |
             |                :                |
             |               name              |
             |              status             |
        0 kB +---------------------------------+

   The upshot of this is twofold:

      1. First, `struct thread' must not be allowed to grow too
         big.  If it does, then there will not be enough room for
         the kernel stack.  Our base `struct thread' is only a
         few bytes in size.  It probably should stay well under 1
         kB.

      2. Second, kernel stacks must not be allowed to grow too
         large.  If a stack overflows, it will corrupt the thread
         state.  Thus, kernel functions should not allocate large
         structures or arrays as non-static local variables.  Use
         dynamic allocation with malloc() or palloc_get_page()
         instead.

   The first symptom of either of these problems will probably be
   an assertion failure in thread_current(), which checks that
   the `magic' member of the running thread's `struct thread' is
   set to THREAD_MAGIC.  Stack overflow will normally change this
   value, triggering the assertion. */
/** The `elem' member has a dual purpose.  It can be an element in
   the run queue (thread.c), or it can be an element in a
   semaphore wait list (synch.c).  It can be used these two ways
   only because they are mutually exclusive: only a thread in the
   ready state is on the run queue, whereas only a thread in the
   blocked state is on a semaphore wait list. */
struct thread
  {
    /* Owned by thread.c. */
    tid_t tid;                          /**< Thread identifier. */
    enum thread_status status;          /**< Thread state. */
    char name[16];                      /**< Name (for debugging purposes). */
    uint8_t *stack;                     /**< Saved stack pointer. */
    int priority;                       /**< Priority. */
    struct list_elem allelem;           /**< List element for all threads list. */

    /* Shared between thread.c and synch.c. */
    struct list_elem elem;              /**< List element. */

#ifdef USERPROG
    /* Owned by userprog/process.c. */
    uint32_t *pagedir;                  /**< Page directory. */
    /* Used in filesys syscall. */ 
    struct files_struct files;          /**< Files struct. */
    /* Used in syscall exec, wait and exit. */
    struct thread *parent;              /**< Current thread's parent. */
    /* Used in syscall exec to synchronize during loading. */
    bool child_load_success;            /**< Child loading result. */
    struct semaphore child_load_sema;   /**< Semaphore for load. */
    /* Used in syscall exit and wait. */
    bool is_parent_died;                /**< Its parent died or not. */
    struct list child_list;             /**< A list of its child. */
    struct child *child_msg;            /**< Child struct of this thread 
                                             owned by its parent. */
    /* Used for deny write to executables. */
    struct file *executable;            /**< Executable file. */
#endif

#ifdef VM
    struct list mmap_list;             /**< A list of its all mappings. */ 
    mmapid_t mmapid_allocater;         /**< Generate unique mmapid in thread. */
    void *user_esp;                    /**< Saved user stack pointer. */
#endif

    /* Owned by thread.c. */
    unsigned magic;                     /**< Detects stack overflow. */
  };

/** If false (default), use round-robin scheduler.
   If true, use multi-level feedback queue scheduler.
   Controlled by kernel command-line option "-o mlfqs". */
extern bool thread_mlfqs;

void thread_init (void);
void thread_start (void);

void thread_tick (void);
void thread_print_stats (void);

typedef void thread_func (void *aux);
tid_t thread_create (const char *name, int priority, thread_func *, void *);

void thread_block (void);
void thread_unblock (struct thread *);

struct thread *thread_current (void);
tid_t thread_tid (void);
const char *thread_name (void);

void thread_exit (void) NO_RETURN;
void thread_yield (void);

/** Performs some operation on thread t, given auxiliary data AUX. */
typedef void thread_action_func (struct thread *t, void *aux);
void thread_foreach (thread_action_func *, void *);

int thread_get_priority (void);
void thread_set_priority (int);

int thread_get_nice (void);
void thread_set_nice (int);
int thread_get_recent_cpu (void);
int thread_get_load_avg (void);

fd_t thread_open_file(struct file*);
int thread_close_file(fd_t fd);
void thread_close_all(void);
struct file *thread_get_file(fd_t fd);

struct child* get_child_thread(tid_t child_tid);
void free_child_list(void);

mmapid_t thread_map_file(struct file *f,void *addr,off_t length);
int thread_unmap_file(mmapid_t mmapid);
void thread_unmap_all(void);
#endif /**< threads/thread.h */
