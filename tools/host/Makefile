# Builds the rdma-core udev-wait shim (see docs/bare-host-verbs-udev-wait.md).
#   make -C tools/host            -> tools/host/libno_cdev_wait.so
#   make -C tools/host PREFIX=/data/<you>/tools/shimlib install
CC ?= gcc
PREFIX ?= /usr/local/lib
all: libno_cdev_wait.so
libno_cdev_wait.so: no_cdev_wait.c
	$(CC) -O2 -shared -fPIC -o $@ $< -ldl
install: libno_cdev_wait.so
	mkdir -p $(PREFIX) && cp libno_cdev_wait.so $(PREFIX)/
clean:
	rm -f libno_cdev_wait.so
.PHONY: all install clean
