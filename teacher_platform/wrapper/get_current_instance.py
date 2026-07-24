import glob
import json
import os
import re
import sys

def get_ancestor_cmdline():
    pid = os.getpid()
    while pid > 1:
        try:
            with open(f"/proc/{pid}/status") as f:
                ppid = None
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
            if ppid is None or ppid <= 1:
                break
            pid = ppid
            cmdline_path = f"/proc/{pid}/cmdline"
            if os.path.exists(cmdline_path):
                with open(cmdline_path, "rb") as f:
                    cmdline = f.read().replace(b'\x00', b' ').decode('utf-8', errors='ignore')
                if "teacher_platform.py" in cmdline:
                    return cmdline
        except Exception:
            break
    return None

cmd = get_ancestor_cmdline() or ""
tasks_match = re.search(r'--tasks\s+([^\s]+)', cmd)
out_dir_match = re.search(r'--out-dir\s+([^\s]+)', cmd)

if not tasks_match or not out_dir_match:
    sys.exit(1)

tasks_file = tasks_match.group(1)
out_dir = out_dir_match.group(1)

if not os.path.exists(tasks_file):
    sys.exit(1)
with open(tasks_file) as f:
    tasks = [json.loads(line) for line in f if line.strip()]

completed = set()
# ``collect --loop`` records attempts in ``<out-dir>_runN``.  The base
# directory can therefore be empty for the entire run; scan it and its sibling
# batches so the docker wrapper binds each new container to the same next task
# the collector selected.
for results_file in glob.glob(out_dir + "*/results.jsonl"):
    try:
        with open(results_file) as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["instance_id"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        continue

for t in tasks:
    if t["instance_id"] not in completed:
        print(t["instance_id"])
        sys.exit(0)

sys.exit(1)
