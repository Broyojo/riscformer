#ifndef _MINI_STDIO_H
#define _MINI_STDIO_H
#include <stddef.h>
#include <stdarg.h>

typedef struct __FILE { int fd; } FILE;
extern FILE *stdin, *stdout, *stderr;

#define EOF (-1)
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#define BUFSIZ 1024

FILE *fopen(const char *path, const char *mode);
int fclose(FILE *f);
size_t fread(void *p, size_t sz, size_t n, FILE *f);
size_t fwrite(const void *p, size_t sz, size_t n, FILE *f);
int fseek(FILE *f, long off, int whence);
long ftell(FILE *f);
int fflush(FILE *f);
int fgetc(FILE *f);
int feof(FILE *f);

int printf(const char *fmt, ...);
int fprintf(FILE *f, const char *fmt, ...);
int sprintf(char *buf, const char *fmt, ...);
int snprintf(char *buf, size_t n, const char *fmt, ...);
int vsnprintf(char *buf, size_t n, const char *fmt, va_list ap);
int vfprintf(FILE *f, const char *fmt, va_list ap);
int sscanf(const char *s, const char *fmt, ...);
int puts(const char *s);
int fputs(const char *s, FILE *f);
int putchar(int c);
int fputc(int c, FILE *f);
#define putc(c, f) fputc(c, f)
int remove(const char *path);
int rename(const char *a, const char *b);

#endif
