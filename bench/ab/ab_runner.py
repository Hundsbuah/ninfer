# is appended by build_args(). The published run used the fork's original prefix cache,
# which the fork now selects with --use-original-prefix-caching.
    ("--use-original-prefix-caching", None),
    ("--kv-capacity", "auto"), ("--vram-headroom-mib", "0"),
             "that has no flag to lower (the fork's `--vram-headroom-mib 0`), so it cannot start "
             "(the fork's `--vram-headroom-mib 0`), so its minimum KV reservation plus that headroom "
