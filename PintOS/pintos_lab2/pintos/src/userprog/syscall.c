#include "userprog/syscall.h"
#include <stdio.h>
#include <syscall-nr.h>
#include "threads/interrupt.h"
#include "threads/thread.h"

static void syscall_handler (struct intr_frame *);

void
syscall_init (void) 
{
  intr_register_int (0x30, 3, INTR_ON, syscall_handler, "syscall");
}

static void
syscall_handler (struct intr_frame *f UNUSED) 
{
  // printf ("system call!\n");

  int syscall_esp = *(int *)f->esp;
  if (syscall_esp == SYS_HALT)
  {
    shutdown_power_off();
  }
  else if (syscall_esp == SYS_EXIT)
  {
    int exit_code = *(int *)(f->esp + sizeof(void *));
    thread_current()->exit_code = exit_code;
    printf("exit with code: %d\n", exit_code);
    thread_exit();
  }
  else if (syscall_esp == SYS_WAIT)
  {
   int pid = *(int *)(f->esp + sizeof(void *));
    f->eax = process_wait(pid);
  }
  else if (syscall_esp == SYS_EXEC)
  {
   char *cmd = *(char **)(f->esp + sizeof(void *));
  f->eax = process_execute(cmd);
  }
  else if(syscall_esp == SYS_WRITE)
  {
    int fd = *(int *)(f->esp + sizeof(void *));
    char *buf = *(char **)(f->esp + 2*sizeof(void *));
    int size = *(int *)(f->esp + 3*sizeof(void *));

    if (fd == 1) {
      putbuf(buf, size);
      f->eax = size;
    }
    // int fd = *(int *)(f->esp + sizeof(void *));
    // const void *buffer = *(const void **)(f->esp + 2 * sizeof(void *));
    // unsigned size = *(unsigned *)(f->esp + 3 * sizeof(void *));
    // f->eax = write(fd, buffer, size);
  }
  else
  {
    printf ("system call not implemented: %s\n", syscall_esp);
  }

  // thread_exit ();
}
