#ifndef USERPROG_SYSCALL_H
#define USERPROG_SYSCALL_H
#include <stdbool.h>

void syscall_init (void);

bool syscall_context(void);
void _exit(int status);
#endif /**< userprog/syscall.h */
