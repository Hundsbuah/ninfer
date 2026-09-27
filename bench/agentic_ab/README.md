API. The fork runs its default hybrid prefix cache; an optional third arm runs the fork build with
the original prefix cache (`--use-original-prefix-caching`). It produces a
  fork's larger device KV at that context (`--kv-headroom-mib 0` and its measured CUDA Graph
  allowance) is part of what is being compared and is shown in the report header.
   `--arms treatment,alt,control` adds the original-prefix-cache arm (about 50 minutes; the arms
   always run in that order). The control's host cache is translated from the split the fork's
   original cache resolves for the same `--host-cache-mib`, read from one extra startup of the
   fork before the first seed. `--seeds 42,43,44` runs every arm once per workload seed, seed by seed, into
| `AB_ALT_EXTRA_FLAGS` | `--use-original-prefix-caching` (added to the treatment's flags) |
