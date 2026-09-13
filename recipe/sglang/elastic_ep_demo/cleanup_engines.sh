#!/bin/bash
# Stop every demo engine (and probe client) from a file: an inline `pkill -f` inside `bash -c` matches its own command line.
pkill -f "elastic_ep_demo/probe_gen.py" 2>/dev/null
pkill -f "sglang serve" 2>/dev/null; sleep 5; pkill -9 -f "sglang::" 2>/dev/null; pkill -9 -f "sglang serve" 2>/dev/null
for f in $(pgrep -f "multiprocessing.forkserver import main"); do pp=$(awk '/^PPid/{print $2}' /proc/$f/status 2>/dev/null); grep -q early_forkserver /proc/$pp/cmdline 2>/dev/null || kill -9 $f 2>/dev/null; done
# engines forked from a daemon keep the daemon's cmdline until setproctitle renames them: kill every child of the daemons
for j in /tmp/sglang_forkserver.json /tmp/sglang_forkserver_hook.json; do
  fs=$(python3 -c "import json;print(json.load(open('$j'))['pid'])" 2>/dev/null); [ -n "$fs" ] && for c in $(pgrep -P "$fs" 2>/dev/null); do kill -9 $c 2>/dev/null; done
done
sleep 2
