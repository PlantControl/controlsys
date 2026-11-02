package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

const graphSchema = "v1"

func main() {
	repo := flag.String("repo", ".", "repository root")
	base := flag.String("base", "origin/main", "base revision; its merge base with -head is used")
	head := flag.String("head", "HEAD", "head revision, or . for the working tree")
	format := flag.String("format", "json", "json, commands or tier")
	cache := flag.String("cache", "", "directory for per-commit graphs")
	baseDir := flag.String("basedir", "", "compare this directory instead of -base")
	headDir := flag.String("headdir", "", "compare this directory instead of -head")
	flag.Parse()
	if *baseDir != "" && *headDir != "" {
		emit(Select(*baseDir, *headDir, Options{BaseCache: *cache}), *format)
		return
	}
	emit(run(*repo, *base, *head, *cache), *format)
}

func run(repo, base, head, cache string) Plan {
	tmp, err := os.MkdirTemp("", "selecttests")
	if err != nil {
		return fullPlan(err.Error())
	}
	defer os.RemoveAll(tmp)
	mergeArg := head
	if head == "." {
		mergeArg = "HEAD"
	}
	mb, err := gitOut(repo, "merge-base", base, mergeArg)
	if err != nil {
		return fullPlan("merge-base: " + err.Error())
	}
	baseDir := filepath.Join(tmp, "base")
	if err := exportRev(repo, mb, baseDir); err != nil {
		return fullPlan("export base: " + err.Error())
	}
	headDir := filepath.Join(tmp, "head")
	opt := Options{BaseCache: cachePrefix(cache, mb)}
	if head == "." {
		err = exportWorkTree(repo, headDir)
	} else {
		var sha string
		if sha, err = gitOut(repo, "rev-parse", head+"^{commit}"); err == nil {
			opt.HeadCache = cachePrefix(cache, sha)
			err = exportRev(repo, sha, headDir)
		}
	}
	if err != nil {
		return fullPlan("export head: " + err.Error())
	}
	return Select(baseDir, headDir, opt)
}

func cachePrefix(dir, sha string) string {
	if dir == "" {
		return ""
	}
	return filepath.Join(dir, graphSchema+"-"+runtime.Version()+"-"+sha)
}

func gitOut(repo string, args ...string) (string, error) {
	out, err := exec.Command("git", append([]string{"-C", repo}, args...)...).Output()
	if err != nil {
		return "", fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	return strings.TrimSpace(string(out)), nil
}

func exportRev(repo, rev, dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	cmd := exec.Command("sh", "-c", `git -C "$0" archive "$1" | tar -x -C "$2"`, repo, rev, dir)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("%w: %s", err, out)
	}
	return nil
}

func exportWorkTree(repo, dir string) error {
	list, err := gitOut(repo, "ls-files", "-co", "--exclude-standard")
	if err != nil {
		return err
	}
	for rel := range strings.SplitSeq(list, "\n") {
		b, err := os.ReadFile(filepath.Join(repo, rel))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return err
		}
		dst := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(dst, b, 0o644); err != nil {
			return err
		}
	}
	return nil
}

func emit(p Plan, format string) {
	switch format {
	case "commands":
		for _, c := range p.Commands() {
			fmt.Println(c)
		}
	case "tier":
		fmt.Println(p.Tier)
	default:
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.SetEscapeHTML(false)
		_ = enc.Encode(p)
	}
}
