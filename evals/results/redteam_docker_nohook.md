## Red-team: escape corpus — runtime backend: docker (container alone, audit hook off)

**35 payloads** · static policy blocks 34/35 · runtime sandbox alone contains 35/35 · combined escapes: **0/35**

| payload | static policy | sandbox alone | what the runtime saw | combined |
|---|---|---|---|---|
| builtin open | blocked (builtin) | contained | `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\KI` | contained |
| io.open | blocked (attribute) | contained | `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\KI` | contained |
| pathlib read_text | blocked (import) | contained | `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\KI` | contained |
| os.open + os.read | blocked (attribute, import) | contained | `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\KI` | contained |
| codecs.open | blocked (attribute, import) | contained | `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\KI` | contained |
| linecache | blocked (import) | contained | `returned ""` | contained |
| /proc/self/environ | blocked (builtin) | contained | `returned "PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbi` | contained |
| parent /proc environ | blocked (builtin, import) | contained | `FileNotFoundError: [Errno 2] No such file or directory: '/proc/0/envir` | contained |
| os.environ leak | blocked (import) | contained | `returned null` | contained |
| os.popen | blocked (import) | contained | `returned ""` | contained |
| os.system | blocked (import) | contained | `returned null` | contained |
| subprocess | blocked (import) | contained | `CalledProcessError: Command '['cat', 'C:\\Users\\KIIT\\AppData\\Local\` | contained |
| __import__ | blocked (builtin) | contained | `returned ""` | contained |
| importlib | blocked (import) | contained | `returned ""` | contained |
| eval | blocked (builtin) | contained | `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\KI` | contained |
| exec | blocked (builtin) | contained | `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\KI` | contained |
| getattr builtins | blocked (builtin, import) | contained | `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\KI` | contained |
| subclasses walk | blocked (dunder) | contained | `returned ""` | contained |
| format-string globals | blocked (dunder) | contained | `returned "{'__name__': 'toolforge_tool', '__builtins__': {'__name__':` | contained |
| sys.modules | blocked (attribute, import) | contained | `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\KI` | contained |
| ctypes system | blocked (import) | contained | `returned null` | contained |
| write marker | blocked (builtin) | contained | `returned null` | contained |
| os.open O_CREAT | blocked (attribute, import) | contained | `returned null` | contained |
| shutil copy | blocked (import) | contained | `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\KI` | contained |
| tempfile | blocked (import) | contained | `returned "/tmp/tmpkkom189o"` | contained |
| sqlite3 file | blocked (import) | contained | `returned null` | contained |
| socket | blocked (import) | contained | `OSError: [Errno 101] Network is unreachable` | contained |
| urllib | blocked (import) | contained | `URLError: <urlopen error [Errno -3] Temporary failure in name resoluti` | contained |
| os.listdir home | blocked (import) | contained | `FileNotFoundError: [Errno 2] No such file or directory: '/nonexistent'` | contained |
| glob | blocked (import) | contained | `returned []` | contained |
| fork | blocked (import) | contained | `returned 7` | contained |
| pickle reduce | blocked (import) | contained | `returned null` | contained |
| zipfile | blocked (import) | contained | `returned null` | contained |
| time.sleep stall | blocked (attribute) | contained | `timed out after 3.0s` | contained |
| memory bomb | — | contained | `process killed (memory or CPU limit?); exit=137` | contained |
