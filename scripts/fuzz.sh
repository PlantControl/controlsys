#!/usr/bin/env bash
# Fuzzes every Fuzz* target for FUZZTIME on the host arch, then replays each
# new testdata/fuzz crasher under GOARCH=amd64 (Rosetta on arm64 Macs; CI
# runs amd64 asm gonum). Not wired into CI. Go stops a target at its first
# crasher, so one run finds at most one per target.
# Usage: scripts/fuzz.sh [FUZZTIME] [TARGET_REGEX]   (defaults: 5m, .)
set -uo pipefail
export GOWORK=off

fuzztime=${1:-5m}
filter=${2:-.}
root=$(git rev-parse --show-toplevel)
cd "$root"

snapshot() { find . -path '*/testdata/fuzz/*' -type f 2>/dev/null | sort; }
before=$(mktemp)
snapshot >"$before"

status=0
for pkg in $(go list ./...); do
	for target in $(go test -list '^Fuzz' "$pkg" 2>/dev/null | grep '^Fuzz' | grep -E -- "$filter"); do
		echo "== $pkg $target ($fuzztime)"
		if ! go test -run '^$' -fuzz "^${target}\$" -fuzztime "$fuzztime" "$pkg"; then
			status=1
		fi
	done
done

new=$(snapshot | comm -13 "$before" -)
rm -f "$before"
if [ -z "$new" ]; then
	echo "no new crashers"
	exit "$status"
fi
echo "new crashers:"
echo "$new"
for f in $new; do
	dir=${f%%/testdata/fuzz/*}
	target=$(basename "$(dirname "$f")")
	echo "== replay GOARCH=amd64 $target/$(basename "$f")"
	GOARCH=amd64 go test -count=1 -run "^${target}\$/^$(basename "$f")\$" "$dir" || true
done
exit 1
