/* minimal start for the gdb Blackfin simulator: stack, main, exit (libgloss syscall: P0 = SYS_exit, R0 = status) */
    .text
    .global __start
__start:
    SP.H = 0x00F0; SP.L = 0x0000;   /* 15 MB into the sim's flat memory */
    FP = SP;
    R0 = SYSCFG;
    BITSET(R0, 1);                  /* CCEN: start the CYCLES counter */
    SYSCFG = R0;
    CALL _main;
    R0 = 0;
    [--SP] = R0;                    /* exit status block */
    R0 = SP;
    P0 = 1;
    EXCPT 0;
1:  JUMP 1b;
