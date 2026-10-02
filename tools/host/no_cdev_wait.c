/* rdma-core open_cdev_robust() waits up to 5 s per verbs device for udev (inotify on /dev/char/)
 * when open(/dev/infiniband/uverbsN) fails; on radix hosts that open is EPERM, so every probe costs
 * 5 s. Failing that one inotify watch makes it return at once, as in the container (no /dev/char
 * there). */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
int inotify_add_watch(int fd, const char* path, uint32_t mask) {
  static int (*real)(int, const char*, uint32_t);
  if (path && strncmp(path, "/dev/char", 9) == 0) {
    errno = ENOENT;
    return -1;
  }
  if (!real)
    real = (int (*)(int, const char*, uint32_t))dlsym(RTLD_NEXT, "inotify_add_watch");
  return real(fd, path, mask);
}
