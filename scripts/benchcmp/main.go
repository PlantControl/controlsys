// Command benchcmp compares the benchmarks of a baseline and a candidate
// revision on the current host and retains the evidence.
//
// Both revisions are exported from git and compiled with go test -c before any
// timing. The prebuilt binaries then run in interleaved rounds (baseline first
// on odd rounds, candidate first on even rounds). Raw output, benchstat
// results, a Markdown summary and metadata describing the revisions,
// toolchain, host and flags are written to a new output directory.
//
// A failed build, failed or panicking benchmark, inconsistent case set or
// benchstat failure makes the comparison fail with status "failed" and no
// benchstat output. Regressions never fail the comparison; the summary lists
// them for review.
//
// Usage:
//
//	go run ./scripts/benchcmp -base origin/main -candidate HEAD -bench . -out /tmp/cmp
//
// -candidate . compares the working tree, including uncommitted changes.
package main

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"
)

type config struct {
	base, candidate string
	bench           string
	benchtime       string
	rounds          int
	cpu             string
	timeout         time.Duration
	out             string
	benchstat       string
	threshold       float64
}

type revision struct {
	Ref        string   `json:"ref"`
	Commit     string   `json:"commit"`
	Dirty      bool     `json:"dirty"`
	DiffSHA256 string   `json:"diff_sha256,omitempty"`
	Untracked  []string `json:"untracked,omitempty"`
}

type binary struct {
	Path      string `json:"path"`
	SHA256    string `json:"sha256"`
	BuildInfo string `json:"build_info"`
	Backend   string `json:"backend"`
}

type run struct {
	Round   int     `json:"round"`
	Label   string  `json:"label"`
	Status  string  `json:"status"`
	Stdout  string  `json:"stdout"`
	Stderr  string  `json:"stderr"`
	Seconds float64 `json:"seconds"`
}

type metadata struct {
	Status      string              `json:"status"`
	Error       string              `json:"error,omitempty"`
	Started     time.Time           `json:"started"`
	Finished    time.Time           `json:"finished,omitzero"`
	Revisions   map[string]revision `json:"revisions"`
	GoVersion   string              `json:"go_version"`
	GoEnv       map[string]string   `json:"go_env"`
	Host        map[string]string   `json:"host"`
	RuntimeEnv  map[string]string   `json:"runtime_env"`
	BuildCmd    []string            `json:"build_command"`
	Flags       []string            `json:"flags"`
	Rounds      int                 `json:"rounds"`
	Timeout     string              `json:"timeout_per_run"`
	Binaries    map[string]*binary  `json:"binaries"`
	Benchstat   string              `json:"benchstat,omitempty"`
	Runs        []*run              `json:"runs"`
	Cases       *caseReport         `json:"cases,omitempty"`
	Regressions []change            `json:"regressions,omitempty"`
}

func (c config) flags() []string {
	f := []string{"-test.run=^$", "-test.bench=" + c.bench, "-test.benchtime=" + c.benchtime, "-test.benchmem", "-test.count=1"}
	if c.cpu != "" {
		f = append(f, "-test.cpu="+c.cpu)
	}
	return f
}

const (
	baseLabel  = "baseline"
	candLabel  = "candidate"
	unreported = "unreported (revision lacks TestNativeBackend)"
)

func main() {
	var c config
	flag.StringVar(&c.base, "base", "origin/main", "baseline git revision")
	flag.StringVar(&c.candidate, "candidate", "HEAD", "candidate git revision, or . for the working tree")
	flag.StringVar(&c.bench, "bench", ".", "go test -bench regular expression")
	flag.StringVar(&c.benchtime, "benchtime", "100ms", "go test -benchtime per sample")
	flag.IntVar(&c.rounds, "rounds", 10, "interleaved samples per revision")
	flag.StringVar(&c.cpu, "cpu", "", "optional go test -cpu list, identical for both binaries")
	flag.DurationVar(&c.timeout, "timeout", 30*time.Minute, "limit for one binary invocation")
	flag.StringVar(&c.out, "out", "", "new output directory (must not exist)")
	flag.StringVar(&c.benchstat, "benchstat", "benchstat", "benchstat executable")
	flag.Float64Var(&c.threshold, "review-threshold", 5, "percent time or B/op increase flagged for review")
	flag.Parse()
	if c.out == "" || c.rounds < 1 {
		fmt.Fprintln(os.Stderr, "benchcmp: -out is required and -rounds must be positive")
		os.Exit(2)
	}
	if err := compare(c); err != nil {
		fmt.Fprintln(os.Stderr, "benchcmp: FAILED:", err)
		os.Exit(1)
	}
	fmt.Println("benchcmp: comparison complete:", c.out)
}

func compare(c config) (err error) {
	root, err := output("", "git", "rev-parse", "--show-toplevel")
	if err != nil {
		return err
	}
	if err := os.Mkdir(c.out, 0o755); err != nil {
		return fmt.Errorf("output directory: %w", err)
	}
	out, err := filepath.Abs(c.out)
	if err != nil {
		return err
	}
	for _, d := range []string{"raw", "bin", "src"} {
		if err := os.Mkdir(filepath.Join(out, d), 0o755); err != nil {
			return err
		}
	}
	md := &metadata{
		Status:     "running",
		Started:    time.Now().UTC(),
		Revisions:  map[string]revision{},
		Binaries:   map[string]*binary{},
		RuntimeEnv: envSubset("GOMAXPROCS", "GODEBUG", "GOGC", "GOMEMLIMIT", "GOFLAGS", "GOEXPERIMENT"),
		Rounds:     c.rounds,
		Timeout:    c.timeout.String(),
		Host:       hostInfo(),
	}
	save := func() {
		md.Finished = time.Now().UTC()
		data, _ := json.MarshalIndent(md, "", "  ")
		_ = os.WriteFile(filepath.Join(out, "metadata.json"), append(data, '\n'), 0o644)
	}
	defer func() {
		if err != nil {
			md.Status = "failed"
			md.Error = err.Error()
			_ = os.WriteFile(filepath.Join(out, "summary.md"), []byte(failedSummary(md)), 0o644)
		}
		save()
		_ = os.RemoveAll(filepath.Join(out, "src"))
		_ = os.RemoveAll(filepath.Join(out, "bin"))
	}()

	md.GoVersion, _ = output(root, "go", "version")
	md.GoEnv = goEnv(root)
	md.BuildCmd = []string{"GOWORK=off", "go", "test", "-c", "-o", "<bin>", "."}
	md.Flags = c.flags()
	if path, err := exec.LookPath(c.benchstat); err == nil {
		info, _ := output("", "go", "version", "-m", path)
		md.Benchstat = path + "\n" + info
	} else {
		return fmt.Errorf("benchstat not found (go install golang.org/x/perf/cmd/benchstat@latest): %w", err)
	}

	for _, rv := range []struct{ label, ref string }{{baseLabel, c.base}, {candLabel, c.candidate}} {
		r, dir, err := checkout(root, out, rv.label, rv.ref)
		if err != nil {
			return fmt.Errorf("%s %s: %w", rv.label, rv.ref, err)
		}
		md.Revisions[rv.label] = r
		save()
		bin, err := build(dir, filepath.Join(out, "bin", rv.label+".test"))
		if err != nil {
			return fmt.Errorf("build %s: %w", rv.label, err)
		}
		md.Binaries[rv.label] = bin
	}
	save()

	samples := map[string][]string{}
	sets := map[string]caseSet{}
	for i := range c.rounds {
		order := []string{baseLabel, candLabel}
		if i%2 == 1 {
			order = []string{candLabel, baseLabel}
		}
		for _, label := range order {
			r := &run{Round: i + 1, Label: label, Status: "running",
				Stdout: fmt.Sprintf("raw/%03d-%s.stdout", i+1, label),
				Stderr: fmt.Sprintf("raw/%03d-%s.stderr", i+1, label)}
			md.Runs = append(md.Runs, r)
			save()
			stdout, err := runBinary(c, md.Binaries[label].Path, filepath.Join(out, r.Stdout), filepath.Join(out, r.Stderr), &r.Seconds)
			if err != nil {
				r.Status = "failed"
				return fmt.Errorf("round %d %s: %w", i+1, label, err)
			}
			set, err := parseResults(stdout)
			if err != nil {
				r.Status = "failed"
				return fmt.Errorf("round %d %s: %w", i+1, label, err)
			}
			if prev, ok := sets[label]; ok && !prev.equal(set) {
				r.Status = "failed"
				return fmt.Errorf("round %d %s: benchmark cases or units differ from round 1", i+1, label)
			}
			sets[label] = set
			samples[label] = append(samples[label], stdout)
			r.Status = "complete"
		}
	}
	report := compareSets(sets[baseLabel], sets[candLabel])
	md.Cases = &report
	if report.Common == 0 {
		return errors.New("no benchmark case completed on both revisions")
	}
	for label, s := range samples {
		if err := os.WriteFile(filepath.Join(out, label+".txt"), []byte(strings.Join(s, "\n")), 0o644); err != nil {
			return err
		}
	}
	args := []string{baseLabel + "=" + filepath.Join(out, baseLabel+".txt"), candLabel + "=" + filepath.Join(out, candLabel+".txt")}
	text, err := output(out, c.benchstat, args...)
	if err != nil {
		return fmt.Errorf("benchstat: %w", err)
	}
	csv, err := output(out, c.benchstat, append([]string{"-format", "csv"}, args...)...)
	if err != nil {
		return fmt.Errorf("benchstat csv: %w", err)
	}
	changes, err := parseBenchstatCSV(csv)
	if err != nil {
		return fmt.Errorf("benchstat csv: %w", err)
	}
	md.Regressions = reviewItems(changes, c.threshold)
	md.Status = "complete"
	if err := os.WriteFile(filepath.Join(out, "benchstat.txt"), []byte(text+"\n"), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "benchstat.csv"), []byte(csv+"\n"), 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, "summary.md"), []byte(summary(md, changes, c.threshold)), 0o644)
}

// checkout resolves ref and returns the directory to build. Commits are
// exported with git archive; "." builds the working tree in place.
func checkout(root, out, label, ref string) (revision, string, error) {
	if ref == "." {
		commit, err := output(root, "git", "rev-parse", "HEAD")
		if err != nil {
			return revision{}, "", err
		}
		r := revision{Ref: ".", Commit: commit}
		diff, err := output(root, "git", "diff", "--binary", "HEAD")
		if err != nil {
			return revision{}, "", err
		}
		untracked, err := output(root, "git", "ls-files", "--others", "--exclude-standard")
		if err != nil {
			return revision{}, "", err
		}
		if untracked != "" {
			r.Untracked = strings.Split(untracked, "\n")
		}
		if diff != "" || untracked != "" {
			r.Dirty = true
			sum := sha256.Sum256([]byte(diff))
			r.DiffSHA256 = hex.EncodeToString(sum[:])
			if err := os.WriteFile(filepath.Join(out, label+".diff"), []byte(diff+"\n"), 0o644); err != nil {
				return revision{}, "", err
			}
		}
		return r, root, nil
	}
	commit, err := output(root, "git", "rev-parse", "--verify", ref+"^{commit}")
	if err != nil {
		return revision{}, "", err
	}
	dir := filepath.Join(out, "src", label)
	if err := export(root, commit, dir); err != nil {
		return revision{}, "", err
	}
	return revision{Ref: ref, Commit: commit}, dir, nil
}

func export(root, commit, dir string) error {
	cmd := exec.Command("git", "archive", "--format=tar", commit)
	cmd.Dir = root
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	pipe, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	tr := tar.NewReader(pipe)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		name := filepath.Join(dir, filepath.FromSlash(h.Name))
		if !strings.HasPrefix(name, dir+string(filepath.Separator)) {
			continue
		}
		switch h.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(name, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
				return err
			}
			f, err := os.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, h.FileInfo().Mode().Perm())
			if err != nil {
				return err
			}
			_, err = io.Copy(f, tr)
			if cerr := f.Close(); err == nil {
				err = cerr
			}
			if err != nil {
				return err
			}
		}
	}
	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("git archive: %w: %s", err, stderr.String())
	}
	return nil
}

func build(dir, bin string) (*binary, error) {
	if _, err := output(dir, "go", "test", "-c", "-o", bin, "."); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(bin)
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(data)
	b := &binary{Path: bin, SHA256: hex.EncodeToString(sum[:])}
	b.BuildInfo, _ = output("", "go", "version", "-m", bin)
	backend, _ := output(dir, bin, "-test.run=^TestNativeBackend$", "-test.v")
	b.Backend = unreported
	if m := regexp.MustCompile(`(?m)^controlsys-backend (.*)$`).FindStringSubmatch(backend); m != nil {
		b.Backend = m[1]
	}
	return b, nil
}

func runBinary(c config, bin, stdoutPath, stderrPath string, seconds *float64) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, c.flags()...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	start := time.Now()
	runErr := cmd.Run()
	*seconds = time.Since(start).Seconds()
	if err := os.WriteFile(stdoutPath, stdout.Bytes(), 0o644); err != nil {
		return "", err
	}
	if err := os.WriteFile(stderrPath, stderr.Bytes(), 0o644); err != nil {
		return "", err
	}
	if runErr != nil {
		if _, err := parseResults(stdout.String()); err != nil {
			runErr = fmt.Errorf("%w: %w", runErr, err)
		}
		return "", fmt.Errorf("%w (see %s)", runErr, filepath.Base(stdoutPath))
	}
	return stdout.String(), nil
}

// output runs a helper command. GOWORK=off keeps builds to each revision's
// own go.mod even when the output directory lies under a go.work.
func output(dir, name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("%s %s: %w: %s", name, strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return strings.TrimSpace(stdout.String()), nil
}

func envSubset(keys ...string) map[string]string {
	m := map[string]string{}
	for _, k := range keys {
		m[k] = os.Getenv(k)
	}
	return m
}

func goEnv(dir string) map[string]string {
	keys := []string{"GOOS", "GOARCH", "GOAMD64", "GOARM64", "CGO_ENABLED", "GOFLAGS", "GOEXPERIMENT", "GOTOOLCHAIN", "GOVERSION"}
	m := map[string]string{}
	text, err := output(dir, "go", append([]string{"env", "-json"}, keys...)...)
	if err == nil {
		_ = json.Unmarshal([]byte(text), &m)
	}
	return m
}

func hostInfo() map[string]string {
	m := map[string]string{
		"num_cpu": fmt.Sprint(runtime.NumCPU()),
		"os":      runtime.GOOS,
		"arch":    runtime.GOARCH,
	}
	m["uname"], _ = output("", "uname", "-a")
	switch runtime.GOOS {
	case "darwin":
		m["cpu"], _ = output("", "sysctl", "-n", "machdep.cpu.brand_string")
		m["rosetta"], _ = output("", "sysctl", "-in", "sysctl.proc_translated")
	case "linux":
		m["cpu"], _ = output("", "sh", "-c", "lscpu | sed -n 's/^Model name: *//p' | head -1")
	}
	for _, k := range []string{"RUNNER_NAME", "RUNNER_OS", "RUNNER_ARCH", "ImageOS", "ImageVersion", "GITHUB_RUN_ID", "GITHUB_WORKFLOW_REF"} {
		if v := os.Getenv(k); v != "" {
			m[k] = v
		}
	}
	return m
}
