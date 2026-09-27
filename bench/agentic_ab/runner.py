  2. treatment arm: the launch bat's flags plus AB_TREATMENT_EXTRA_FLAGS, with the fork's
     default hybrid prefix cache;
     original prefix cache), so one run compares both fork configurations to the control;
     --host-state-slots / --host-kv-mib / catalog limits using the split the fork's original
     prefix cache resolves at startup (one model load per run), so the arms get the same
     pinned host RAM and catalog sizes;
# Selects the fork's original checkpoint-catalog prefix cache instead of the default hybrid one.
ORIGINAL_CACHE_FLAG = "--use-original-prefix-caching"
ALT_EXTRA_FLAGS = os.environ.get("AB_ALT_EXTRA_FLAGS", ORIGINAL_CACHE_FLAG).split()
    """The control's explicit host-cache flags equal to an original-cache fork's resolved split."""
def original_cache_split(model, treat_flags, ctx, run_dir):
    """server_start of the treatment build with the original prefix cache, whose --host-cache-mib
    budget resolves into the explicit Host state/KV split and catalog sizes the control takes."""
    flags = set_flag(treat_flags, "--max-context", str(ctx))
    if ORIGINAL_CACHE_FLAG not in dict(flags):
        flags.append((ORIGINAL_CACHE_FLAG, None))
    log("host-cache split: treatment build with %s" % ORIGINAL_CACHE_FLAG)
    probe = Serve(TREATMENT_EXE, model, flags, os.path.join(run_dir, "original_cache_split.jsonl"),
                  os.path.join(run_dir, "original_cache_split_serve.log")).start()
    try:
        start = read_server_start(probe.request_log)
    finally:
        probe.stop()
    if start is None:
        raise SystemExit("the original-cache startup wrote no server_start record")
    return start


def run_seed(seed, plan, run_dir, arms, model, flags, control_host, config, ctx):
        for n, v in control_host:
            ctrl = set_flag(ctrl, n, v)
    control_host = []
    if "control" in arms and "--host-cache-mib" in dict(treat_flags) and \
            "--host-cache-mib" not in ctrl_supported:
        control_host = host_translation(original_cache_split(model, treat_flags, ctx, out_dir))
        log("control host cache = original-cache --host-cache-mib split: %s"
            % flag_str(control_host))
        failures += run_seed(seed, plans[seed], run_dir, arms, model, flags, control_host,
