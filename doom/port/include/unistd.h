#ifndef _MINI_UNISTD_H
#define _MINI_UNISTD_H
#include <stddef.h>
typedef long ssize_t;
typedef long off_t;
int unlink(const char *path);
#endif
