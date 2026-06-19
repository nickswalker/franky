#!/usr/bin/env zsh
# Sweep tracking_error.py over speed x friction x translational x rotational stiffness.
# Each run appends a TSV row to $TSV (default: data/tracking_metrics.tsv).
# Re-running the script skips combos already in the file (set RESUME=0 to rerun all).
#
# Usage:
#   zsh scripts/sweep_tracking_error.zsh
#   HOST=10.0.0.2 TSV=data/my_run.tsv zsh scripts/sweep_tracking_error.zsh
#
# Plot results with: uv run plot_tracking_metrics.py

set -u

TSV=${TSV:-data/tracking_metrics.tsv}

speeds=(0.25 0.125)
frictions=(true)
inertial_ffs=(true false)
trans_stiffs=(250 500 750 1000 1500 2000)
rot_stiffs=(7.5 15 30 45 60)

# Skip combos known to be unstable.
skip_run() {
  local s=$1 t=$2 r=$3
  [[ $s == 0.5 ]] && (( r > 30 )) && return 0
  return 1
}

# Resume support: collect keys already present in the TSV.
typeset -A done
if [[ ${RESUME:-1} == 1 && -f $TSV ]]; then
  while read -r key; do
    [[ -n $key ]] && done[$key]=1
  done < <(TSV=$TSV uv run python - <<'PY' 2>/dev/null
import csv, os
from pathlib import Path
rows = list(csv.DictReader(open(Path(os.environ["TSV"])), delimiter="\t"))
def fr(v): return "true" if str(v).strip().lower() in ("true", "1", "yes") else "false"
for r in rows:
    try:
        s = float(r["speed"])
        t = float(r["translational_stiffness"])
        rr = float(r["rotational_stiffness"])
    except (KeyError, ValueError, TypeError):
        continue
    print(f"{s:g}|{t:g}|{rr:g}|{fr(r.get('friction_comp', ''))}|{fr(r.get('inertial_ff', ''))}")
PY
)
  print "Resume: ${#done} combos already logged in $TSV will be skipped"
fi

total=$(( ${#speeds} * ${#frictions} * ${#inertial_ffs} * ${#trans_stiffs} * ${#rot_stiffs} ))
i=0
ran=0
print "Sweep: $total combos"

for s in $speeds; do
  for f in $frictions; do
    for ff in $inertial_ffs; do
      for t in $trans_stiffs; do
        for r in $rot_stiffs; do
          i=$(( i + 1 ))
          if skip_run $s $t $r; then
            print "[$i/$total] skip (unstable): speed=$s trans=$t rot=$r"
            continue
          fi
          key="${s}|${t}|${r}|${f}|${ff}"
          if [[ -n ${done[$key]:-} ]]; then
            print "[$i/$total] skip (done): speed=$s friction=$f ff=$ff trans=$t rot=$r"
            continue
          fi
          print "\n=== [$i/$total] speed=$s friction=$f ff=$ff trans=$t rot=$r ==="
          friction_flag=()
          [[ $f == true ]] && friction_flag=(--friction)
          inertial_flag=()
          [[ $ff == true ]] && inertial_flag=(--inertial-ff)
          uv run examples/tracking_error.py \
            --speed "$s" \
            --trans-stiff "$t" \
            --rot-stiff "$r" \
            $friction_flag \
            $inertial_flag \
            --append-tsv "$TSV"
          ran=$(( ran + 1 ))
        done
      done
    done
  done
done

print "\nSweep complete: ran $ran new runs ($i/$total combos) -> $TSV"