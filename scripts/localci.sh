#!/usr/bin/env bash
# Local CI gate (the repo is private; GitHub Actions has no credits). Becomes
# the per-merge gate once the nightly full run below is scheduled. Runs scripts/selecttests against BASE (default
# origin/main) on the working tree and executes the commands of its tier:
# native, GOARCH=amd64 and -race tests plus the bench job, reduced to the
# affected tests and benchmarks when the selector can prove it (see
# scripts/selecttests/doc.go). Any selector error runs the full suite.
#
# Usage: scripts/localci.sh [--full] [--dry-run]
#   --full     skip the selector and run the full suite
#   --dry-run  print the commands without running them
# Env: BASE (default origin/main). Fetch it first; it is not fetched here.
#
# Full-run policy (the selector replaces per-merge full runs only):
#   - Nightly: scripts/localci.sh --full on main.
#   - Before any tag or version bump (controlsys or a gonum bump in go.mod):
#     --full on the commit being tagged or bumped.
#   - On demand: --full for Riccati changes (riccati*.go, Care/Dare/Kalman/Lqr
#     paths) and for gonum changes, even when the selector reduces them.
#   - If a nightly ever fails a test that a merged PR's reduced run skipped,
#     switch to --full on every merge until the selector is fixed.
set -uo pipefail
export GOWORK=off

root=$(git rev-parse --show-toplevel) || exit 1
cd "$root" || exit 1
base=${BASE:-origin/main}
full=0
dry=0
for a in "$@"; do
	case $a in
	--full) full=1 ;;
	--dry-run) dry=1 ;;
	*)
		echo "usage: scripts/localci.sh [--full] [--dry-run]" >&2
		exit 2
		;;
	esac
done

# Mirrors Plan.Commands for tier full; TestLocalCIFullCommands keeps them equal.
full_commands=(
	'test -z "$(gofmt -l .)"'
	'go vet ./...'
	'go fix ./... && git diff --exit-code'
	'go test -count=1 -timeout 0 ./...'
	'GOARCH=amd64 go test -count=1 -timeout 0 ./...'
	'go test -race -count=1 -timeout 0 ./...'
	"go test -run '^\$' -bench . -benchtime=1x -benchmem ./..."
	"GOARCH=amd64 go test -run '^\$' -bench . -benchtime=1x -benchmem ./..."
	'go -C scripts/selecttests vet ./...'
	'go -C scripts/selecttests test ./...'
)

commands=()
if [ "$full" = 1 ]; then
	echo "localci: --full"
	commands=("${full_commands[@]}")
else
	tier=
	cache=$(cd "$(git rev-parse --git-common-dir)" && pwd)/selecttests-cache
	if plan=$(go -C scripts/selecttests run . -repo "$root" -base "$base" -head . -cache "$cache" -format commands); then
		while IFS= read -r line; do
			case $line in
			'# tier: '*) tier=${line#'# tier: '} ;;
			'#'*) echo "localci: ${line#'# '}" ;;
			?*) commands+=("$line") ;;
			esac
		done <<<"$plan"
	fi
	if [ "${#commands[@]}" = 0 ] || [ -z "$tier" ]; then
		echo "localci: selector failed; running full suite" >&2
		commands=("${full_commands[@]}")
		tier=full
	fi
	echo "localci: tier $tier against $base"
fi

status=0
for c in "${commands[@]}"; do
	echo "== $c"
	[ "$dry" = 1 ] && continue
	start=$SECONDS
	if bash -c "$c"; then
		echo "-- ok $((SECONDS - start))s"
	else
		echo "-- FAIL $((SECONDS - start))s: $c" >&2
		status=1
	fi
done
exit $status
