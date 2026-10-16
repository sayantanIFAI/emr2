#!/usr/bin/env bash
# Run on YOUR machine (Git Bash) before renting the pods: is everything the two-node test needs here?
#
#   bash infra/runpod/preflight_laptop.sh            check only
#   bash infra/runpod/preflight_laptop.sh stage      also build what can be built: the national lists and the medicine-name file
#
# The staging folder (default ~/pod_stage, override with POD_STAGE=...) holds data that must NOT be in git:
#   codes/clci_labs.json  codes/cdci_drugs.json   built from the two C-DAC zips (their licence: redistribution within India)
#   codes/medicine_names.txt                      built from the A-Z medicines dataset (archive.zip)
#   pages/                                        real prescription pages for the load test: the OWNER decides which; they are patients' data
# Sources: the C-DAC zips in ~/Downloads (or C_DAC_DIR), the medicines dataset archive in MEDICINE_ZIP (default ~/Downloads/archive.zip).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="${POD_STAGE:-$HOME/pod_stage}"
ZIPS="${C_DAC_DIR:-$HOME/Downloads}"
MEDZIP="${MEDICINE_ZIP:-$HOME/Downloads/archive.zip}"
PY="$(command -v python3 || command -v python || true)"
bad=0
ok()   { printf '  [ok]      %s\n' "$1"; }
miss() { printf '  [MISSING] %s\n' "$1"; bad=1; }
warn() { printf '  [check]   %s\n' "$1"; }

if [ "${1:-}" = "stage" ]; then
  mkdir -p "$STAGE/codes" "$STAGE/pages"
  LABS="$(ls "$ZIPS"/common-lab-codes-for-india-*.zip 2>/dev/null | tail -1)"
  DRUGS="$(ls "$ZIPS"/CommonDrugCodesForIndia_FlatFilePackage*.zip 2>/dev/null | tail -1)"
  if [ -n "$PY" ] && [ -n "$LABS" ] && [ -n "$DRUGS" ] && [ ! -s "$STAGE/codes/clci_labs.json" ]; then
    "$PY" "$HERE/scripts/build_indian_codes.py" --labs "$LABS" --drugs "$DRUGS" --out "$STAGE/codes"
  fi
  if [ -n "$PY" ] && [ -s "$MEDZIP" ] && [ ! -s "$STAGE/codes/medicine_names.txt" ]; then
    "$PY" "$HERE/scripts/build_medicine_lexicon.py" "$MEDZIP" "$STAGE/codes/medicine_names.txt"
  fi
fi

echo "Preflight for the two-node RunPod test  (staging folder: $STAGE)"
echo "tools"
for t in ssh scp git curl; do command -v "$t" >/dev/null && ok "$t" || miss "$t is not installed"; done
[ -n "$PY" ] && ok "python: $("$PY" --version 2>&1)" || miss "python (3.11 or newer)"
echo "ssh key"
[ -s "$HOME/.ssh/id_ed25519" ] && ok "$HOME/.ssh/id_ed25519" || miss "$HOME/.ssh/id_ed25519 (the key RunPod has on its Credentials page)"
echo "code"
if git -C "$HERE" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$HERE" fetch -q origin 2>/dev/null
  if [ -z "$(git -C "$HERE" status --porcelain)" ]; then ok "working tree is clean"; else miss "uncommitted changes: commit and push first (the pods deploy from git)"; fi
  [ "$(git -C "$HERE" rev-parse HEAD)" = "$(git -C "$HERE" rev-parse origin/main 2>/dev/null)" ] && ok "HEAD equals origin/main ($(git -C "$HERE" rev-parse --short HEAD))" || miss "HEAD is not on origin/main: push, because the pods clone origin/main"
fi
echo "data to push to the app node (not in git)"
[ -s "$STAGE/codes/clci_labs.json" ] && [ -s "$STAGE/codes/cdci_drugs.json" ] && ok "national lists built" || miss "national lists: run  bash $0 stage  (needs the two C-DAC zips in $ZIPS)"
[ -s "$STAGE/codes/medicine_names.txt" ] && ok "medicine names ($(wc -l < "$STAGE/codes/medicine_names.txt") words)" || miss "medicine names: run  bash $0 stage  (needs $MEDZIP)"
ls "$ZIPS"/common-lab-codes-for-india-*.zip >/dev/null 2>&1 && ok "C-DAC lab zip (lets the pod rebuild the lists)" || warn "C-DAC lab zip not in $ZIPS (optional)"
n="$(ls "$STAGE/pages" 2>/dev/null | grep -c -i -E '\.(jpe?g|png|pdf|tiff?)$')"
[ "$n" -ge 10 ] && ok "$n load-test pages in $STAGE/pages" || warn "$n pages in $STAGE/pages (use at least 10 real handwritten ones; the owner decides which; they are patients' data)"
echo
if [ "$bad" = 0 ]; then echo "READY: rent the two pods (docs/two-node-runpod.md, section 2)."; else echo "NOT READY: fix the [MISSING] lines above."; fi
exit "$bad"
