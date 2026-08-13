#ifndef _MINI_SYS_STAT_H
#define _MINI_SYS_STAT_H
struct stat { unsigned st_mode; long st_size; };
#define S_IFCHR 0020000
#define S_IFREG 0100000
int mkdir(const char *path, unsigned mode);
#endif
