## Red-team: escape corpus — runtime backend: docker + audit hook

**35 payloads** · static policy blocks 34/35 · runtime sandbox alone contains 35/35 · combined escapes: **0/35**

| payload | static policy | sandbox alone | what the runtime saw | combined |
|---|---|---|---|---|
| builtin open | blocked (builtin) | contained | `SandboxViolation: [sandbox] file access blocked: 'C:\\Users\\KIIT\\App` | contained |
| io.open | blocked (attribute) | contained | `SandboxViolation: [sandbox] file access blocked: 'C:\\Users\\KIIT\\App` | contained |
| pathlib read_text | blocked (import) | contained | `SandboxViolation: [sandbox] file access blocked: 'C:\\Users\\KIIT\\App` | contained |
| os.open + os.read | blocked (attribute, import) | contained | `SandboxViolation: [sandbox] file access blocked: 'C:\\Users\\KIIT\\App` | contained |
| codecs.open | blocked (attribute, import) | contained | `SandboxViolation: [sandbox] file access blocked: 'C:\\Users\\KIIT\\App` | contained |
| linecache | blocked (import) | contained | `returned ""` | contained |
| /proc/self/environ | blocked (builtin) | contained | `SandboxViolation: [sandbox] file access blocked: '/proc/self/environ' ` | contained |
| parent /proc environ | blocked (builtin, import) | contained | `SandboxViolation: [sandbox] file access blocked: '/proc/0/environ' (r)` | contained |
| os.environ leak | blocked (import) | contained | `returned null` | contained |
| os.popen | blocked (import) | contained | `SandboxViolation: [sandbox] opening file descriptors is blocked` | contained |
| os.system | blocked (import) | contained | `SandboxViolation: [sandbox] blocked operation: os.system` | contained |
| subprocess | blocked (import) | contained | `SandboxViolation: [sandbox] opening file descriptors is blocked` | contained |
| __import__ | blocked (builtin) | contained | `SandboxViolation: [sandbox] opening file descriptors is blocked` | contained |
| importlib | blocked (import) | contained | `SandboxViolation: [sandbox] opening file descriptors is blocked` | contained |
| eval | blocked (builtin) | contained | `SandboxViolation: [sandbox] file access blocked: 'C:\\Users\\KIIT\\App` | contained |
| exec | blocked (builtin) | contained | `SandboxViolation: [sandbox] file access blocked: 'C:\\Users\\KIIT\\App` | contained |
| getattr builtins | blocked (builtin, import) | contained | `SandboxViolation: [sandbox] file access blocked: 'C:\\Users\\KIIT\\App` | contained |
| subclasses walk | blocked (dunder) | contained | `SandboxViolation: [sandbox] opening file descriptors is blocked` | contained |
| format-string globals | blocked (dunder) | contained | `returned "{'__name__': 'toolforge_tool', '__builtins__': {'__name__':` | contained |
| sys.modules | blocked (attribute, import) | contained | `SandboxViolation: [sandbox] file access blocked: 'C:\\Users\\KIIT\\App` | contained |
| ctypes system | blocked (import) | contained | `SandboxViolation: [sandbox] blocked operation: ctypes.dlopen` | contained |
| write marker | blocked (builtin) | contained | `SandboxViolation: [sandbox] file access blocked: 'C:\\Users\\KIIT\\App` | contained |
| os.open O_CREAT | blocked (attribute, import) | contained | `SandboxViolation: [sandbox] file access blocked: 'C:\\Users\\KIIT\\App` | contained |
| shutil copy | blocked (import) | contained | `SandboxViolation: [sandbox] blocked operation: shutil.copyfile` | contained |
| tempfile | blocked (import) | contained | `FileNotFoundError: [Errno 2] No usable temporary directory found in ['` | contained |
| sqlite3 file | blocked (import) | contained | `SandboxViolation: [sandbox] blocked operation: sqlite3.connect` | contained |
| socket | blocked (import) | contained | `SandboxViolation: [sandbox] blocked operation: socket.getaddrinfo` | contained |
| urllib | blocked (import) | contained | `SandboxViolation: [sandbox] blocked operation: urllib.Request` | contained |
| os.listdir home | blocked (import) | contained | `SandboxViolation: [sandbox] directory listing blocked: '/nonexistent'` | contained |
| glob | blocked (import) | contained | `SandboxViolation: [sandbox] blocked operation: glob.glob` | contained |
| fork | blocked (import) | contained | `SandboxViolation: [sandbox] blocked operation: os.fork` | contained |
| pickle reduce | blocked (import) | contained | `SandboxViolation: [sandbox] blocked operation: os.system` | contained |
| zipfile | blocked (import) | contained | `SandboxViolation: [sandbox] file access blocked: 'C:\\Users\\KIIT\\App` | contained |
| time.sleep stall | blocked (attribute) | contained | `timed out after 3.0s` | contained |
| memory bomb | — | contained | `process killed (memory or CPU limit?); exit=137` | contained |
