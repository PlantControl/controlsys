#!/usr/bin/env bash
# Measures dense vs Hessenberg frequency sweeps (BenchmarkFrequencyDispatch)
# on this host and writes raw samples, benchstat by path, and environment.
# Usage: scripts/crossover.sh OUT_DIR [ROUNDS] [BENCHTIME]
set -euo pipefail
export GOWORK=off

out=${1:?usage: scripts/crossover.sh OUT_DIR [ROUNDS] [BENCHTIME]}
rounds=${2:-10}
benchtime=${3:-100ms}
root=$(git rev-parse --show-toplevel)
mkdir "$out"
out=$(cd "$out" && pwd)

cd "$root"
go test -c -o "$out/controlsys.test" .
{
	echo "commit: $(git rev-parse HEAD)"
	echo "dirty: $(test -z "$(git status --porcelain)" && echo false || echo true)"
	go version
	go env -json GOOS GOARCH GOAMD64 GOARM64 CGO_ENABLED GOFLAGS GOEXPERIMENT
	uname -a
	case $(uname -s) in
	Darwin) sysctl -n machdep.cpu.brand_string hw.ncpu ;;
	Linux) lscpu | sed -n 's/^Model name: *//p'; nproc ;;
	esac
	echo "GOMAXPROCS env: ${GOMAXPROCS:-unset}"
	echo "rounds: $rounds benchtime: $benchtime"
	"$out/controlsys.test" -test.run '^TestNativeBackend$' | grep '^controlsys-backend'
	go version -m "$(command -v benchstat)" | sed -n '1,3p'
} >"$out/environment.txt"

for ((i = 1; i <= rounds; i++)); do
	"$out/controlsys.test" -test.run '^$' -test.bench '^BenchmarkFrequencyDispatch$' \
		-test.benchtime "$benchtime" -test.benchmem -test.count 1 >"$out/round-$i.txt"
	if ! grep -q '^PASS$' "$out/round-$i.txt" || grep -q -e '--- FAIL' -e '^panic:' "$out/round-$i.txt"; then
		echo "crossover: round $i failed; see $out/round-$i.txt" >&2
		exit 1
	fi
done
cat "$out"/round-*.txt >"$out/raw.txt"
rm "$out/controlsys.test"
benchstat -col /path -row /n,/m,/w,/a -filter '.unit:sec/op' "$out/raw.txt" >"$out/benchstat.txt"
benchstat -col /path -row /n,/m,/w,/a -format csv "$out/raw.txt" >"$out/benchstat.csv" 2>/dev/null
echo "crossover: $out"
