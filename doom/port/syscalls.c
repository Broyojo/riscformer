/* newlib syscall stubs -> ecall host bridge */
#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <stdint.h>

static inline long ecall3(long n, long a, long b, long c) {
    register long a0 asm("a0") = a;
    register long a1 asm("a1") = b;
    register long a2 asm("a2") = c;
    register long a7 asm("a7") = n;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}

#define SYS_read 63
#define SYS_write 64
#define SYS_close 57
#define SYS_lseek 62
#define SYS_exit 93
#define SYS_open 1024

int _open(const char *path, int flags, int mode) {
    return (int)ecall3(SYS_open, (long)path, flags, mode);
}
int _close(int fd) { return (int)ecall3(SYS_close, fd, 0, 0); }
ssize_t _read(int fd, void *buf, size_t n) {
    return (ssize_t)ecall3(SYS_read, fd, (long)buf, (long)n);
}
ssize_t _write(int fd, const void *buf, size_t n) {
    return (ssize_t)ecall3(SYS_write, fd, (long)buf, (long)n);
}
off_t _lseek(int fd, off_t off, int whence) {
    return (off_t)ecall3(SYS_lseek, fd, (long)off, whence);
}
void _exit(int code) {
    for (;;) ecall3(SYS_exit, code, 0, 0);
}

int _fstat(int fd, struct stat *st) {
    (void)fd;
    st->st_mode = S_IFCHR;
    return 0;
}
int _stat(const char *p, struct stat *st) { (void)p; (void)st; errno = ENOENT; return -1; }
int _isatty(int fd) { return fd < 3; }
int _kill(int pid, int sig) { (void)pid; (void)sig; errno = EINVAL; return -1; }
int _getpid(void) { return 1; }
int _unlink(const char *p) { (void)p; errno = ENOENT; return -1; }
int _link(const char *a, const char *b) { (void)a; (void)b; errno = EMLINK; return -1; }

extern char _end[];              /* from linker script */
extern char __heap_limit[];
static char *brk = _end;

void *_sbrk(ptrdiff_t incr) {
    char *old = brk;
    if (brk + incr > __heap_limit) {
        errno = ENOMEM;
        return (void *)-1;
    }
    brk += incr;
    return old;
}
