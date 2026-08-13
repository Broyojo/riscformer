/* doomgeneric platform layer for the transformer CPU.
 * All host interaction goes through the ecall bridge. */
#include <stdint.h>
#include "doomgeneric.h"

static inline long ecall3(long n, long a, long b, long c) {
    register long a0 asm("a0") = a;
    register long a1 asm("a1") = b;
    register long a2 asm("a2") = c;
    register long a7 asm("a7") = n;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}

void DG_Init(void) {}

void DG_DrawFrame(void) {
    ecall3(2003, (long)DG_ScreenBuffer, DOOMGENERIC_RESX * DOOMGENERIC_RESY, 0);
}

void DG_SleepMs(uint32_t ms) { (void)ms; }

uint32_t DG_GetTicksMs(void) { return (uint32_t)ecall3(2001, 0, 0, 0); }

int DG_GetKey(int *pressed, unsigned char *doomKey) {
    long v = ecall3(2002, 0, 0, 0);
    if (v < 0) return 0;
    *pressed = (v >> 8) & 1;
    *doomKey = (unsigned char)(v & 0xFF);
    return 1;
}

void DG_SetWindowTitle(const char *title) { (void)title; }

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    static char *av[] = {"doom", "-iwad", "doom1.wad", "-singletics"};
    doomgeneric_Create(4, av);
    for (;;)
        doomgeneric_Tick();
    return 0;
}
