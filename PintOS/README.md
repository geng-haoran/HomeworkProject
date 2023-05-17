## Key Commands

- docker run with mount
    
    `docker run -it --rm --name pintos --mount type=bind,source=/Users/genghaoran/Code/HomeworkProject/PintOS/pintos_lab3/pintos,target=/home/PKUOS/pintos pkuflyingpig/pintos bash`
    
    `docker run -it --rm --name pintos --mount type=bind,source=/Users/genghaoran/Code/HomeworkProject/PintOS/pintos_lab3/pintos_ref,target=/home/PKUOS/pintos pkuflyingpig/pintos bash`
    
- build and start
    
    `cd pintos/src/threads/
    make
    cd build
    pintos --`
    
- test code
    
    `pintos -- run alarm-multiple`
    
    `ctrl + a + c`: exit
    
- ****Debug and Test****
    
    `pintos --gdb -- run xxx`
    
    another terminal:
    
    `docker exec -it pintos bash`
    
    `cd pintos/src/threads/build
     pintos-gdb kernel.o`
    
    `debugpintos`
    
    ### gdb usage
    
    You can read the GDB manual by typing `info gdb` at a terminal command prompt. Here's a few commonly useful GDB commands:
    
    **GDB Command: c**
    
    Continues execution until Ctrl+C or the next breakpoint.
    
    **GDB Command: si**
    
    Execute one machine instruction.
    
    **GDB Command: s**
    
    Execute until next line reached, step into function calls.
    
    **GDB Command: n**
    
    Execute until next line reached, step over function calls.
    
    **GDB Command: p**** *****expression***
    
    Evaluates the given expression and prints its value. If the expression contains a function call, that function will actually be executed.
    
    **GDB Command: finish**
    
    Run until the selected function (stack frame) returns
    
    **GDB Command: b**** *****function***
    
    **GDB Command: b**** *****file:line***
    
    **GDB Command: b** ****address***
    
    Sets a breakpoint at *function*, at *line* within *file*, or *address*. `b` is short for `break` or `breakpoint`. (Use a 0x prefix to specify an address in hex.)Use `b pintos_init` to make GDB stop when Pintos starts running.
    
    **GDB Command: info** ***registers***Print the general purpose registers, eip, eflags, and the segment selectors. For a much more thorough dump of the machine register state, see QEMU's own info registers command.
    
    **GDB Command:** **x/Nx** ***addr***Display a hex dump of N words starting at virtual address *addr*. If N is omitted, it defaults to 1. *addr* can be any expression.
    
    **GDB Command: x/Ni** ***addr***Display the N assembly instructions starting at *addr*. Using $eip as *addr* will display the instructions at the current instruction pointer.
    
    **GDB Command:** **l** ****address***Lists a few lines of code around *address*. (Use a 0x prefix to specify an address in hex.)
    
    **GDB Command: bt**Prints a stack backtrace similar to that output by the `backtrace` program described above.
    
    **GDB Command:** **frame** ***n***Select frame number n or frame at address n
    
    **GDB Command:** **p/a** ***address***Prints the name of the function or variable that occupies *address*. (Use a 0x prefix to specify an address in hex.)
    
    **GDB Command:** **diassemble** ***function***Disassembles function.
    
    **GDB Command: thread** ***n***GDB focuses on one thread (i.e., CPU) at a time. This command switches that focus to thread n, numbered from zero.
    
    **GDB Command: info** ***threads***List all threads (i.e., CPUs), including their state (active or halted) and what function they're in.
    

## some errors

- Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?