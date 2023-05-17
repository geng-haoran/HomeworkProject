#include "userprog/syscall.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <syscall-nr.h>
#include "threads/interrupt.h"
#include "threads/thread.h"
#include "threads/vaddr.h"
#include "threads/malloc.h"
#include "threads/synch.h"
#include "userprog/process.h"
#include "userprog/pagedir.h"
#include "filesys/filesys.h"
#include "filesys/filesys_lock.h"
#include "devices/input.h"
#include "devices/shutdown.h"

#define SYSNUM 15
#define MAXARGS 4

static bool in_syscall_context;     /**< Are we processing a syscall? */

static struct lock exit_lock;       /**< A lock for exit call. */

static void syscall_handler (struct intr_frame *);
/** Syscalls. */
static int sys_halt(uint32_t *argv);
static int sys_exit(uint32_t *argv);
static int sys_exec(uint32_t *argv);
static int sys_wait(uint32_t *argv);
static int sys_create(uint32_t *argv);
static int sys_remove(uint32_t *argv);
static int sys_open(uint32_t *argv);
static int sys_filesize(uint32_t *argv);
static int sys_read(uint32_t *argv);
static int sys_write(uint32_t *argv);
static int sys_seek(uint32_t *argv);
static int sys_tell(uint32_t *argv);
static int sys_close(uint32_t *argv);
static int sys_mmap(uint32_t *argv);
static int sys_munmap(uint32_t *argv);
/** Helper routines. */
static int get_arg_n(int n, void *stack_pointer, uint32_t *dst);
static int get_args(int argc,void *stack_pointer,uint32_t *argv);
static int check_user_string(const uint8_t *uaddr);
static int copy_user_string(const uint8_t *uaddr,uint8_t **kbuf);
static int copy_user_string_bound(const uint8_t *uaddr,uint8_t *kbuf,
                                                    unsigned length);
static int get_user(const uint8_t *uaddr);
static bool put_user (uint8_t *udst, uint8_t byte);
static void get_keyboard_input(uint8_t *buffer,unsigned length);
/** Handler list. */
static int 
(*syscalls[])(uint32_t *argv)={
  [SYS_HALT]    =   sys_halt,
  [SYS_EXIT]    =   sys_exit,
  [SYS_EXEC]    =   sys_exec,
  [SYS_WAIT]    =   sys_wait,
  [SYS_CREATE]  =   sys_create,
  [SYS_REMOVE]  =   sys_remove,
  [SYS_OPEN]    =   sys_open,
  [SYS_FILESIZE]=   sys_filesize,
  [SYS_READ]    =   sys_read,
  [SYS_WRITE]   =   sys_write,
  [SYS_SEEK]    =   sys_seek,
  [SYS_TELL]    =   sys_tell,
  [SYS_CLOSE]   =   sys_close,
  [SYS_MMAP]    =   sys_mmap,
  [SYS_MUNMAP]  =   sys_munmap
};
/** Argc list. */
static int _argc[]={
  [SYS_HALT]    =   1,
  [SYS_EXIT]    =   2,
  [SYS_EXEC]    =   2,
  [SYS_WAIT]    =   2,
  [SYS_CREATE]  =   3,
  [SYS_REMOVE]  =   2,
  [SYS_OPEN]    =   2,
  [SYS_FILESIZE]=   2,
  [SYS_READ]    =   4,
  [SYS_WRITE]   =   4,
  [SYS_SEEK]    =   3,
  [SYS_TELL]    =   2,
  [SYS_CLOSE]   =   2,
  [SYS_MMAP]    =   3,
  [SYS_MUNMAP]  =   2
};

void
syscall_init (void) 
{
  in_syscall_context=false;
  lock_init(&exit_lock);
  intr_register_int (0x30, 3, INTR_ON, syscall_handler, "syscall");
}

/** Returns true during processing of a syscall
   and false at all other times. */
bool 
syscall_context(void)
{
  return in_syscall_context;
}

/** Make current thread exit with exit value STATUS */
void 
_exit(int status)
{
  struct thread* cur=thread_current();
  /* Print exit message. */
  printf("%s: exit(%d)\n",cur->name,status);

  /* Unmap all mapping files. */
  thread_unmap_all();

  /* Close all open files. */
  thread_close_all();

  /* Close its executable file. */
  lock_acquire(&filesys_lock);
  file_close(cur->executable);
  lock_release(&filesys_lock);

  lock_acquire(&exit_lock);
  /* Notify its parent. */
  if(!cur->is_parent_died)
  {
    /* Record exit status in parent's child_list. */
    cur->child_msg->is_terminated=true;
    cur->child_msg->saved_exit_status=status;
    /* Notify the parent. */
    sema_up(&cur->child_msg->terminated_sema);
  }
  /* Notify its child. */
  free_child_list();
  lock_release(&exit_lock);
  
  /* Since _exit() never returns, we should show that 
  syscall has finished. */
  in_syscall_context=false;
  /* Really exit and release resources. */
  thread_exit();
  NOT_REACHED();
}

/** Get the Nth argument on user stack, store at DST
    SYSCALL_NUMBER is arg0. 
    Return 0 if success, otherwise -1. */
static int 
get_arg_n(int n, void *stack_pointer, uint32_t *dst)
{
  uint8_t buffer[sizeof(uint32_t)+1];
  int read_value;
  uint8_t *start_pos=(uint8_t*)((uint32_t*)stack_pointer+n);
  for(int byte=0;byte<4;byte++)
  {
    if(!is_user_vaddr(start_pos+byte)||
       (read_value=get_user(start_pos+byte))<0)
       return -1;
    buffer[byte]=(uint8_t)read_value;
  }
  *dst=*(uint32_t*)buffer;
  return 0;
}

/** Get ARGC arguments on user stack, store at ARGV.
   return 0 if success, otherwise -1. */
static int 
get_args(int argc,void *stack_pointer,uint32_t *argv)
{
  ASSERT(0<argc&&argc<=MAXARGS);
  for(int i=0;i<argc;i++)
  {
    if(get_arg_n(i,stack_pointer,argv+i)<0)
      return -1;
  }
  return 0;
}

/** Check whether user string UADDR is a vaild string.
   Assume user string will end with '\0'. 
   Return string length if success, otherwise -1. */
static int 
check_user_string(const uint8_t *uaddr)
{
  unsigned cnt=0;
  int read_value=-1;
  while(is_user_vaddr(uaddr+cnt)&&
        (read_value=get_user(uaddr+cnt))>0)
          cnt++;
  if(is_user_vaddr(uaddr+cnt)&&read_value==0)
    return cnt;
  else 
    return -1;
}


/** Copy a user string from UADDR to KBUF.
   Assume user string will end with '\0'. 
   Return 0 if success. Use malloc to create a buffer store at *KBUF. 
   Caller must call free on this buffer later.
   Otherwise return -1. No memory allocations. */
static int 
copy_user_string(const uint8_t *uaddr,uint8_t **kbuf)
{
  int len;
  if((len=check_user_string(uaddr))<0)
    return -1;

  uint8_t *ker_buffer=malloc(len+1);
  copy_user_string_bound(uaddr,ker_buffer,len);
  ker_buffer[len]='\0';
  *kbuf=ker_buffer;
  return 0;
}

/** Copy LENGTH bytes from UADDR to KBUF.
   Do not stop when we encounter '\0'.
   Return 0 if success, otherwise -1. */
static int 
copy_user_string_bound(const uint8_t *uaddr,uint8_t *kbuf,unsigned length)
{
  unsigned cnt=0;
  int read_value=-1;
  while(cnt<length&&
      is_user_vaddr(uaddr+cnt)&&
      (read_value=get_user(uaddr+cnt))>=0)
      {
        kbuf[cnt]=read_value;
        cnt++;
      }
  if(cnt==length)
    return 0;
  else
    return -1;
}

/** Reads a byte at user virtual address UADDR.
   UADDR must be below PHYS_BASE.
   Returns the byte value if successful, -1 if a segfault
   occurred. */
static int
get_user (const uint8_t *uaddr)
{
  int result;
  asm ("movl $1f, %0; movzbl %1, %0; 1:"
       : "=&a" (result) : "m" (*uaddr));
  return result;
}

/** Writes BYTE to user address UDST.
   UDST must be below PHYS_BASE.
   Returns true if successful, false if a segfault occurred. */
static bool
put_user (uint8_t *udst, uint8_t byte)
{
  int error_code;
  asm ("movl $1f, %0; movb %b2, %1; 1:"
       : "=&a" (error_code), "=m" (*udst) : "q" (byte));
  return error_code != -1;
}

/** Get LENGTH bytes from keyboard and store to BUFFER. 
   Caller must ensure that BUFFER has enough space. */
static void 
get_keyboard_input(uint8_t *buffer,unsigned length)
{
  unsigned cnt=0;
  while(cnt<length)
    buffer[cnt++]=input_getc();
}

/** Global entry for syscalls. */
static void
syscall_handler (struct intr_frame *f) 
{
  in_syscall_context=true;

  uint32_t syscall_number;
  uint32_t argv[MAXARGS];
  if(get_arg_n(0,f->esp,&syscall_number)<0)
    _exit(-1);
  if(syscall_number>=SYSNUM||
  get_args(_argc[syscall_number],f->esp,argv)<0)
    _exit(-1);
  if(syscall_number<SYSNUM&&syscalls[syscall_number])
    f->eax=syscalls[syscall_number](argv);
  else
    /* Invaild syscall number, return -1. */
    f->eax=-1;

  in_syscall_context=false;
}

/** Syscall handlers. */

static int 
sys_halt(uint32_t *argv UNUSED)
{
  shutdown_power_off();
  NOT_REACHED();
  return -1;
}

static int 
sys_exit(uint32_t *argv)
{
  int status=(int)argv[1];

  _exit(status);
  return -1;
}

static int 
sys_exec(uint32_t *argv)
{
  const char *cmd_line=(const char *)argv[1];

  tid_t tid;
  if(check_user_string((uint8_t*)cmd_line)<0)
    _exit(-1);
  tid=process_execute(cmd_line);
  /* When it reaches here, child should finish loading. 
    Returns a PID equal to TID. */
  return tid;
}

static int 
sys_wait(uint32_t *argv)
{
  tid_t tid=(tid_t)argv[1];

  return process_wait(tid);
}

static int 
sys_create(uint32_t *argv)
{
  const char *file=(const char *)argv[1];
  off_t initial_size=(off_t)argv[2];

  const char *kfile;
  int retval;
  if(copy_user_string((const uint8_t*)file,(uint8_t**)&kfile)<0)
    _exit(-1);
  lock_acquire(&filesys_lock);
  retval=(int)filesys_create(kfile,initial_size);
  lock_release(&filesys_lock);
  free((void*)kfile);
  return retval;
}

static int 
sys_remove(uint32_t *argv)
{
  const char *file=(const char *)argv[1];

  const char *kfile;
  int retval;
  if(copy_user_string((const uint8_t*)file,(uint8_t**)&kfile)<0)
    _exit(-1);
  lock_acquire(&filesys_lock);
  retval=(int)filesys_remove(kfile);
  lock_release(&filesys_lock);
  free((void*)kfile);
  return retval;
}

static int 
sys_open(uint32_t *argv)
{
  const char *file=(const char *)argv[1];

  const char *kfile;
  int retval;
  if(copy_user_string((const uint8_t*)file,(uint8_t**)&kfile)<0)
    _exit(-1);
  lock_acquire(&filesys_lock);
  retval=thread_open_file(filesys_open(kfile));
  lock_release(&filesys_lock);
  free((void*)kfile);
  return retval;
}

static int 
sys_filesize(uint32_t *argv)
{
  fd_t fd=(fd_t)argv[1];

  struct file *f;
  int retval;
  if((f=thread_get_file(fd))==NULL)
    return -1;
  lock_acquire(&filesys_lock);
  retval=file_length(f);
  lock_release(&filesys_lock);
  return retval;
}

static int 
sys_read(uint32_t *argv)
{
  fd_t fd=(fd_t)argv[1];
  uint8_t *user_buffer=(uint8_t*)argv[2];
  unsigned length=(unsigned)argv[3];
  /* Read zero byte does nothing but a fd check. */
  if(length==0)
    return (fd==STDIN_FD||thread_get_file(fd))?0:-1;
  /* Create a kernel buffer to contain reading content. */
  uint8_t *ker_buffer=malloc(length*sizeof(uint8_t));
  unsigned cnt=0;
  if(ker_buffer==NULL)
    return -1;
  if(fd==STDIN_FD)
    /* Read from keyboard. */
    get_keyboard_input(ker_buffer,length);
  else
  {
    /* Read from files. */
    struct file* f=thread_get_file(fd);
    if(fd==STDOUT_FD||f==NULL)
    {
      free(ker_buffer);
      return -1;
    }
    lock_acquire(&filesys_lock);
    length=file_read(f,ker_buffer,length);
    lock_release(&filesys_lock);
  }
  /* Write kernel buffer to user buffer. User address also checks here. */
  while(cnt<length&&
        is_user_vaddr(user_buffer+cnt)&&
        put_user(user_buffer+cnt,ker_buffer[cnt]))
        cnt++;
  free(ker_buffer);
  if(cnt!=length)
    _exit(-1);
  return length;
}

static int 
sys_write(uint32_t *argv)
{
  fd_t fd=(fd_t)argv[1];
  const void *user_buffer=(const void*)argv[2];
  unsigned length=(unsigned)argv[3];
  /* Write zero byte does nothing but a fd check. */
  if(length==0)
    return (fd==STDOUT_FD||thread_get_file(fd))?0:-1;
  void *ker_buffer=malloc(length);
  if(ker_buffer==NULL)
    return -1;
  /* Copy user string. User address also checks here. */
  if(copy_user_string_bound((const uint8_t*)user_buffer,
                          (uint8_t*)ker_buffer,length)<0)
    {
      free(ker_buffer);
      _exit(-1);
    }
  if(fd==STDOUT_FD)
    /* Write to the console. */
    putbuf(ker_buffer,length);
  else
  {
    /* Get the file and write to it. */
    struct file* f=thread_get_file(fd);
    if(fd==STDIN_FD||f==NULL)
    {
      free(ker_buffer);
      return -1;
    }
    lock_acquire(&filesys_lock);
    length=file_write(f,ker_buffer,(off_t)length);
    lock_release(&filesys_lock);
  }
  free(ker_buffer);
  return length;
}

static int 
sys_seek(uint32_t *argv)
{
  fd_t fd=(fd_t)argv[1];
  int position=(int)argv[2];

  struct file *f;
  if((f=thread_get_file(fd))==NULL)
    return -1;
  lock_acquire(&filesys_lock);
  file_seek(f,position);
  lock_release(&filesys_lock);
  return 0;
}

static int 
sys_tell(uint32_t *argv)
{
  fd_t fd=(fd_t)argv[1];

  struct file *f;
  int retval;
  if((f=thread_get_file(fd))==NULL)
    return -1;
  lock_acquire(&filesys_lock);
  retval=file_tell(f);
  lock_release(&filesys_lock);
  return retval;
}

static int 
sys_close(uint32_t *argv)
{
  fd_t fd=(fd_t)argv[1];
  
  int retval=thread_close_file(fd);
  return retval;
}

static int 
sys_mmap(uint32_t *argv)
{
    fd_t fd=(fd_t)argv[1];
    void *addr=(void *)argv[2];

    struct file *f;
    off_t length;
    mmapid_t mmapid;
    /* Basic check. */
    if(addr==NULL||pg_ofs(addr)!=0||(f=thread_get_file(fd))==NULL)
      return -1;
    /* Reopen file in case the file is later closed. */
    lock_acquire(&filesys_lock);
    length=file_length(f);
    if((f=file_reopen(f))==NULL)
    {
      lock_release(&filesys_lock);
      return -1;
    }
    lock_release(&filesys_lock);
    /* Map the file in page table. */
    if(!pagedir_map_page(thread_current()->pagedir,addr,f,length))
    {
      lock_acquire(&filesys_lock);
      file_close(f);
      lock_release(&filesys_lock);
      return -1;
    }
    /* Save the mapping infos in thread's struct. */
    if((mmapid=thread_map_file(f,addr,length))==MMAPID_ERROR)
    {
      lock_acquire(&filesys_lock);
      file_close(f);
      lock_release(&filesys_lock);
      pagedir_unmap_page(thread_current()->pagedir,addr,length);
      return -1;
    }
    return mmapid;
}

static int 
sys_munmap(uint32_t *argv)
{
    mmapid_t mmapid=(mmapid_t)argv[1];

    int retval=thread_unmap_file(mmapid);
    return retval;
}