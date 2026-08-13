ENTRY(_start)

MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 64M }

SECTIONS {
    .text : { *(.init) *(.text*) } > RAM
    .rodata : { *(.rodata*) *(.srodata*) } > RAM
    .data : {
        . = ALIGN(8);
        __global_pointer$ = . + 0x800;
        *(.data*) *(.sdata*)
    } > RAM
    .bss : {
        . = ALIGN(4);
        __bss_start = .;
        *(.bss*) *(.sbss*) *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > RAM
    . = ALIGN(16);
    _end = .;
    __heap_limit = ORIGIN(RAM) + LENGTH(RAM) - 2M;
    __stack_top = ORIGIN(RAM) + LENGTH(RAM) - 16;
}
