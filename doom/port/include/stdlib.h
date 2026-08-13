#ifndef _MINI_STDLIB_H
#define _MINI_STDLIB_H
#include <stddef.h>
void *malloc(size_t n);
void *calloc(size_t n, size_t sz);
void *realloc(void *p, size_t n);
void free(void *p);
void exit(int code) __attribute__((noreturn));
void abort(void) __attribute__((noreturn));
int abs(int x);
long labs(long x);
int atoi(const char *s);
double atof(const char *s);
long strtol(const char *s, char **end, int base);
char *getenv(const char *name);
int system(const char *cmd);
int rand(void);
void srand(unsigned seed);
#define RAND_MAX 0x7FFFFFFF
#endif
