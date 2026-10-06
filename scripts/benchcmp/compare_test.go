package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func gitRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Chdir(dir)
	t.Setenv("GIT_CONFIG_GLOBAL", os.DevNull)
	t.Setenv("GIT_CONFIG_NOSYSTEM", "1")
	git(t, "init", "-q")
	git(t, "config", "user.email", "benchcmp@example.com")
	git(t, "config", "user.name", "benchcmp")
	write(t, "go.mod", "module example.com/b\n\ngo 1.24\n")
	return dir
}

func git(t *testing.T, args ...string) string {
	t.Helper()
	out, err := exec.Command("git", args...).CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func write(t *testing.T, name, content string) {
	t.Helper()
	if err := os.WriteFile(name, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func commit(t *testing.T, bench string) string {
	t.Helper()
	write(t, "b_test.go", "package b\n\nimport \"testing\"\n\n"+bench)
	git(t, "add", "-A")
	git(t, "commit", "-q", "-m", "c")
	return git(t, "rev-parse", "HEAD")
}

func readMetadata(t *testing.T, out string) metadata {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(out, "metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	var md metadata
	if err := json.Unmarshal(data, &md); err != nil {
		t.Fatal(err)
	}
	return md
}

const benchA = "func BenchmarkA(b *testing.B) {\n\tfor b.Loop() {\n\t}\n}\n\n"

func TestCompareRevisions(t *testing.T) {
	if testing.Short() {
		t.Skip("builds test binaries")
	}
	benchstat, err := exec.LookPath("benchstat")
	if err != nil {
		t.Skip("benchstat not installed")
	}
	gitRepo(t)
	base := commit(t, benchA+"func BenchmarkOld(b *testing.B) {\n\tfor b.Loop() {\n\t}\n}\n")
	cand := commit(t, benchA+"func BenchmarkNew(b *testing.B) {\n\tfor b.Loop() {\n\t}\n}\n")
	out := filepath.Join(t.TempDir(), "cmp")
	c := config{base: base, candidate: "HEAD", bench: ".", benchtime: "100x", rounds: 2, timeout: time.Minute, out: out, benchstat: benchstat, threshold: 5}
	if err := compare(c); err != nil {
		t.Fatal(err)
	}
	md := readMetadata(t, out)
	if md.Status != "complete" || md.Revisions[baseLabel].Commit != base || md.Revisions[candLabel].Commit != cand {
		t.Fatalf("metadata = %+v", md)
	}
	var order []string
	for _, r := range md.Runs {
		order = append(order, r.Label)
	}
	if want := []string{baseLabel, candLabel, candLabel, baseLabel}; !slices.Equal(order, want) {
		t.Fatalf("run order = %v, want %v", order, want)
	}
	if md.Cases.Common != 1 || len(md.Cases.BaselineOnly) != 1 || len(md.Cases.CandidateOnly) != 1 {
		t.Fatalf("cases = %+v", md.Cases)
	}
	for _, f := range []string{"benchstat.txt", "summary.md", "baseline.txt", "candidate.txt", "raw/001-baseline.stdout"} {
		if _, err := os.Stat(filepath.Join(out, f)); err != nil {
			t.Error(err)
		}
	}
	if _, err := os.Stat(filepath.Join(out, "src")); !os.IsNotExist(err) {
		t.Errorf("exported sources retained: %v", err)
	}
	if err := compare(c); err == nil {
		t.Fatal("compare overwrote an existing output directory")
	}
}

func TestCompareFailingBenchmarkProducesNoComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("builds test binaries")
	}
	gitRepo(t)
	base := commit(t, benchA)
	commit(t, benchA+"func BenchmarkBroken(b *testing.B) {\n\tb.Fatal(\"broken\")\n}\n")
	fake := filepath.Join(t.TempDir(), "benchstat")
	write(t, fake, "#!/bin/sh\nexit 0\n")
	if err := os.Chmod(fake, 0o755); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "cmp")
	c := config{base: base, candidate: "HEAD", bench: ".", benchtime: "10x", rounds: 2, timeout: time.Minute, out: out, benchstat: fake, threshold: 5}
	if err := compare(c); err == nil {
		t.Fatal("compare succeeded with a failing benchmark")
	}
	md := readMetadata(t, out)
	if md.Status != "failed" || !strings.Contains(md.Error, "benchmark failed") {
		t.Fatalf("status %q error %q", md.Status, md.Error)
	}
	if _, err := os.Stat(filepath.Join(out, "benchstat.txt")); !os.IsNotExist(err) {
		t.Fatalf("benchstat.txt written for failed comparison: %v", err)
	}
	summary, err := os.ReadFile(filepath.Join(out, "summary.md"))
	if err != nil || !strings.Contains(string(summary), "FAILED") {
		t.Fatalf("summary = %q, %v", summary, err)
	}
}

func TestCompareWorkingTreeRecordsDiff(t *testing.T) {
	if testing.Short() {
		t.Skip("builds test binaries")
	}
	benchstat, err := exec.LookPath("benchstat")
	if err != nil {
		t.Skip("benchstat not installed")
	}
	gitRepo(t)
	head := commit(t, benchA)
	write(t, "b_test.go", "package b\n\nimport \"testing\"\n\n"+benchA+"func BenchmarkLocal(b *testing.B) {\n\tfor b.Loop() {\n\t}\n}\n")
	out := filepath.Join(t.TempDir(), "cmp")
	c := config{base: "HEAD", candidate: ".", bench: ".", benchtime: "10x", rounds: 1, timeout: time.Minute, out: out, benchstat: benchstat, threshold: 5}
	if err := compare(c); err != nil {
		t.Fatal(err)
	}
	md := readMetadata(t, out)
	r := md.Revisions[candLabel]
	if r.Commit != head || !r.Dirty || r.DiffSHA256 == "" || len(md.Cases.CandidateOnly) != 1 {
		t.Fatalf("candidate = %+v cases = %+v", r, md.Cases)
	}
	if _, err := os.Stat(filepath.Join(out, "candidate.diff")); err != nil {
		t.Fatal(err)
	}
}
