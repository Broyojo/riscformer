#ifndef _MINI_ASSERT_H
#define _MINI_ASSERT_H
void __assert_fail(const char *expr, const char *file, int line);
#ifdef NDEBUG
#define assert(x) ((void)0)
#else
#define assert(x) ((x) ? (void)0 : __assert_fail(#x, __FILE__, __LINE__))
#endif
#endif
