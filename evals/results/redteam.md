## Red-team: escape corpus

**35 payloads** · static policy blocks 34/35 · runtime sandbox alone contains 35/35 · combined escapes: **0/35**

| payload | static policy | sandbox alone | combined |
|---|---|---|---|
| builtin open | blocked (builtin) | contained | contained |
| io.open | blocked (attribute) | contained | contained |
| pathlib read_text | blocked (import) | contained | contained |
| os.open + os.read | blocked (attribute, import) | contained | contained |
| codecs.open | blocked (attribute, import) | contained | contained |
| linecache | blocked (import) | contained | contained |
| /proc/self/environ | blocked (builtin) | contained | contained |
| parent /proc environ | blocked (builtin, import) | contained | contained |
| os.environ leak | blocked (import) | contained | contained |
| os.popen | blocked (import) | contained | contained |
| os.system | blocked (import) | contained | contained |
| subprocess | blocked (import) | contained | contained |
| __import__ | blocked (builtin) | contained | contained |
| importlib | blocked (import) | contained | contained |
| eval | blocked (builtin) | contained | contained |
| exec | blocked (builtin) | contained | contained |
| getattr builtins | blocked (builtin, import) | contained | contained |
| subclasses walk | blocked (dunder) | contained | contained |
| format-string globals | blocked (dunder) | contained | contained |
| sys.modules | blocked (attribute, import) | contained | contained |
| ctypes system | blocked (import) | contained | contained |
| write marker | blocked (builtin) | contained | contained |
| os.open O_CREAT | blocked (attribute, import) | contained | contained |
| shutil copy | blocked (import) | contained | contained |
| tempfile | blocked (import) | contained | contained |
| sqlite3 file | blocked (import) | contained | contained |
| socket | blocked (import) | contained | contained |
| urllib | blocked (import) | contained | contained |
| os.listdir home | blocked (import) | contained | contained |
| glob | blocked (import) | contained | contained |
| fork | blocked (import) | contained | contained |
| pickle reduce | blocked (import) | contained | contained |
| zipfile | blocked (import) | contained | contained |
| time.sleep stall | blocked (attribute) | contained | contained |
| memory bomb | — | contained | contained |
