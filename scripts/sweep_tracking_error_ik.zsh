#!/usr/bin/env zsh
# Sweep tracking_error_ik.py over speed x stiffness x friction x inertial-ff.
# Appends rows to $TSV (default: data/tracking_metrics.tsv) — same file as the
# cartesian sweep, distinguished by tracker="joint_ik" or "joint_ik_ff".
# Re-runs skip combos already present (set RESUME=0 to rerun all).
#
# Usage:
#   zsh scripts/sweep_tracking_error_ik.zsh
#   TSV=data/my_run.tsv zsh scripts/sweep_tracking_error_ik.zsh

set -u

TSV=${TSV:-data/tracking_metrics.tsv}

speeds=(0.25 0.125)
stiffnesses=(80 160 240 320 400)
frictions=(true false)
inertial_ffs=(true false)

# Resume support: collect keys already present in the TSV for joint_ik trackers.
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
    if r.get("tracker") != "joint_ik":
        continue
    try:
        s = float(r["speed"])
        st = float(r["translational_stiffness"])
    except (KeyError, ValueError, TypeError):
        continue
    print(f"{s:g}|{st:g}|{fr(r.get('friction_comp', ''))}|{fr(r.get('inertial_ff', ''))}")
PY
)
  print "Resume: ${#done} combos already logged in $TSV will be skipped"
fi

total=$(( ${#speeds} * ${#stiffnesses} * ${#frictions} * ${#inertial_ffs} ))
i=0
ran=0
print "Sweep: $total combos"

for s in $speeds; do
  for st in $stiffnesses; do
    for f in $frictions; do
      for ff in $inertial_ffs; do
        i=$(( i + 1 ))
        key="${s}|${st}|${f}|${ff}"
        if [[ -n ${done[$key]:-} ]]; then
          print "[$i/$total] skip (done): speed=$s stiffness=$st friction=$f ff=$ff"
          continue
        fi
        print "\n=== [$i/$total] speed=$s stiffness=$st friction=$f ff=$ff ==="
        friction_flag=()
        [[ $f == true ]] && friction_flag=(--friction)
        ff_flag=()
        [[ $ff == true ]] && ff_flag=(--inertial-ff)
        uv run examples/tracking_error_ik.py \
          --speed "$s" \
          --stiffness "$st" \
          $friction_flag \
          $ff_flag \
          --append-tsv "$TSV"
        ran=$(( ran + 1 ))
      done
    done
  done
done

print "\nSweep complete: ran $ran new runs ($i/$total combos) -> $TSV"