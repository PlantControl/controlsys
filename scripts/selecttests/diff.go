package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type declKind int

const (
	declFunc declKind = iota
	declVar
	declType
)

type decl struct {
	kind   declKind
	key    string
	recv   string
	hash   string
	file   string
	isTest bool
	root   RootKind
	sig    string
	// initNames are identifiers a variable's initializer names outside
	// imported packages; initOpaque marks one it cannot vouch for.
	initNames  []string
	initOpaque bool
}

// tree is the parsed view of one checkout's root package.
type tree struct {
	files   map[string][]byte
	decls   map[string]*decl
	roots   map[string]Root
	readers map[string]bool
	parsed  map[string]*goFile
	// ifaceMethods holds method names declared by interface types.
	ifaceMethods map[string]bool
}

type goFile struct {
	directives  []string
	sideImports []string
	pkg         string
	constrained bool
	generated   bool
}

var (
	generatedRE = regexp.MustCompile(`(?m)^// Code generated .* DO NOT EDIT\.$`)
	sourceLitRE = regexp.MustCompile(`\.(go|md|mod|sum|s)\b|assets/|README|AGENTS`)
	archSuffix  = regexp.MustCompile(`_(aix|android|darwin|dragonfly|freebsd|hurd|illumos|ios|js|linux|netbsd|openbsd|plan9|solaris|wasip1|windows|386|amd64|arm|arm64|loong64|mips|mips64|mips64le|mipsle|ppc64|ppc64le|riscv64|s390x|sparc64|wasm)(_test)?\.go$`)
)

var sourceImports = map[string]bool{
	"go/ast": true, "go/parser": true, "go/build": true, "go/token": true, "go/types": true,
	"go/printer": true, "go/format": true, "go/importer": true, "go/doc": true, "os/exec": true,
	"golang.org/x/tools/go/packages": true, "golang.org/x/tools/go/ssa": true,
}

var sourceCalls = map[string]bool{
	"runtime.Caller": true, "runtime.Callers": true, "runtime.FuncForPC": true, "debug.Stack": true,
	"os.ReadDir": true, "os.DirFS": true, "filepath.Walk": true, "filepath.WalkDir": true, "filepath.Glob": true,
}

func readTree(dir string) (*tree, error) {
	t := &tree{files: map[string][]byte{}, decls: map[string]*decl{}, roots: map[string]Root{}, readers: map[string]bool{}, parsed: map[string]*goFile{}, ifaceMethods: map[string]bool{}}
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		if d.IsDir() {
			if rel == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		t.files[filepath.ToSlash(rel)] = b
		return nil
	})
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(t.files))
	for n := range t.files {
		if isRootGo(n) {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	for _, n := range names {
		if err := t.parse(n); err != nil {
			return nil, err
		}
	}
	return t, nil
}

func isRootGo(name string) bool {
	return !strings.Contains(name, "/") && strings.HasSuffix(name, ".go")
}

func (t *tree) parse(name string) error {
	src := t.files[name]
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, parser.ParseComments)
	if err != nil {
		return err
	}
	gf := &goFile{pkg: f.Name.Name, generated: generatedRE.Match(src), constrained: archSuffix.MatchString(name)}
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			if strings.HasPrefix(c.Text, "//go:") {
				gf.directives = append(gf.directives, c.Text)
				if strings.HasPrefix(c.Text, "//go:build") {
					gf.constrained = true
				}
			}
			if strings.HasPrefix(c.Text, "// +build") {
				gf.constrained = true
			}
		}
	}
	sort.Strings(gf.directives)
	for _, imp := range f.Imports {
		if imp.Name != nil && (imp.Name.Name == "_" || imp.Name.Name == ".") {
			gf.sideImports = append(gf.sideImports, imp.Name.Name+imp.Path.Value)
		}
	}
	sort.Strings(gf.sideImports)
	t.parsed[name] = gf

	isTest := strings.HasSuffix(name, "_test.go")
	span := func(n ast.Node) []byte { return src[fset.Position(n.Pos()).Offset:fset.Position(n.End()).Offset] }
	add := func(key string, d decl, payload string) {
		d.key, d.file, d.isTest = key, name, isTest
		sum := sha256.Sum256([]byte(payload))
		d.hash = fmt.Sprintf("%x", sum[:8])
		if prev, ok := t.decls[key]; ok {
			sum := sha256.Sum256([]byte(prev.hash + d.hash))
			prev.hash = fmt.Sprintf("%x", sum[:8])
			return
		}
		t.decls[key] = &d
	}
	var fileRoots []Root
	for _, dc := range f.Decls {
		switch v := dc.(type) {
		case *ast.FuncDecl:
			recv := ""
			if v.Recv != nil && len(v.Recv.List) > 0 {
				recv = recvName(v.Recv.List[0].Type)
			}
			d := decl{kind: declFunc, recv: recv}
			if recv != "" {
				d.sig = tokenHash(span(v.Type))
			}
			payload := tokenHash(span(v))
			if isTest {
				if k, ok := classifyRoot(v); ok {
					d.root = k
					r := Root{Name: v.Name.Name, Kind: k, File: name}
					t.roots[r.Name] = r
					fileRoots = append(fileRoots, r)
					if strings.HasPrefix(r.Name, "Example") {
						// The Output comment is the expectation.
						payload = string(span(v))
					}
				}
			}
			if v.Doc != nil {
				for _, c := range v.Doc.List {
					if strings.HasPrefix(c.Text, "//go:") {
						payload += c.Text
					}
				}
			}
			add(funcKey(gf.pkg, recv, v.Name.Name), d, payload)
		case *ast.GenDecl:
			for _, sp := range v.Specs {
				switch s := sp.(type) {
				case *ast.ValueSpec:
					if v.Tok != token.VAR {
						continue
					}
					// Initialization order depends on file order.
					names, opaque := initializerNames(s, f)
					for _, id := range s.Names {
						add("var:"+funcKey(gf.pkg, "", id.Name), decl{kind: declVar, initNames: names, initOpaque: opaque}, name+tokenHash(span(s)))
					}
				case *ast.TypeSpec:
					add("type:"+funcKey(gf.pkg, "", s.Name.Name), decl{kind: declType}, tokenHash(span(s)))
				}
			}
		}
	}
	ast.Inspect(f, func(n ast.Node) bool {
		if it, ok := n.(*ast.InterfaceType); ok {
			for _, m := range it.Methods.List {
				for _, id := range m.Names {
					t.ifaceMethods[id.Name] = true
				}
			}
		}
		return true
	})
	if isTest && readsSource(name, f) {
		for _, r := range fileRoots {
			t.readers[r.Name] = true
		}
	}
	return nil
}

// initializerNames lists the identifiers s names, skipping struct keys and
// selections from imported packages. A function literal, or a call of
// anything but a named function, conversion or imported package function,
// makes the initializer opaque.
func initializerNames(s *ast.ValueSpec, f *ast.File) (names []string, opaque bool) {
	imports := map[string]bool{}
	for _, imp := range f.Imports {
		p, _ := strconv.Unquote(imp.Path.Value)
		n := p[strings.LastIndex(p, "/")+1:]
		if imp.Name != nil {
			n = imp.Name.Name
		}
		imports[n] = true
	}
	var walk func(n ast.Node) bool
	walk = func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.FuncLit:
			opaque = true
			return false
		case *ast.KeyValueExpr:
			if _, ok := v.Key.(*ast.Ident); !ok {
				ast.Inspect(v.Key, walk)
			}
			ast.Inspect(v.Value, walk)
			return false
		case *ast.SelectorExpr:
			if x, ok := v.X.(*ast.Ident); ok && imports[x.Name] {
				return false
			}
		case *ast.CallExpr:
			switch fn := ast.Unparen(v.Fun).(type) {
			case *ast.Ident, *ast.IndexExpr, *ast.IndexListExpr, *ast.ArrayType, *ast.MapType, *ast.ChanType, *ast.FuncType, *ast.InterfaceType, *ast.StructType, *ast.StarExpr:
			case *ast.SelectorExpr:
				if x, ok := fn.X.(*ast.Ident); !ok || !imports[x.Name] {
					opaque = true
					return false
				}
			default:
				opaque = true
				return false
			}
		case *ast.Ident:
			names = append(names, v.Name)
		}
		return true
	}
	if s.Type != nil {
		ast.Inspect(s.Type, walk)
	}
	for _, v := range s.Values {
		ast.Inspect(v, walk)
	}
	return names, opaque
}

// selfContained reports whether d's initializer names no function or
// variable of the package, so evaluating it cannot reach package code or
// touch package state.
func (t *tree) selfContained(d *decl) bool {
	if d.initOpaque {
		return false
	}
	pre := ""
	if strings.HasPrefix(d.key, "var:_test:") {
		pre = "_test:"
	}
	for _, n := range d.initNames {
		for _, k := range []string{pre + n, "var:" + pre + n, n, "var:" + n} {
			if o := t.decls[k]; o != nil && o.kind != declType && o != d {
				return false
			}
		}
	}
	return true
}

func readsSource(name string, f *ast.File) bool {
	if strings.HasPrefix(name, "architecture_") {
		return true
	}
	for _, imp := range f.Imports {
		p, _ := strconv.Unquote(imp.Path.Value)
		if sourceImports[p] {
			return true
		}
	}
	found := false
	ast.Inspect(f, func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.BasicLit:
			if v.Kind == token.STRING && sourceLitRE.MatchString(v.Value) {
				found = true
			}
		case *ast.SelectorExpr:
			if x, ok := v.X.(*ast.Ident); ok && sourceCalls[x.Name+"."+v.Sel.Name] {
				found = true
			}
		}
		return !found
	})
	return found
}

// tokenHash hashes src's tokens, so formatting and comments do not count.
func tokenHash(src []byte) string {
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(src))
	var s scanner.Scanner
	s.Init(file, src, func(token.Position, string) {}, 0)
	h := sha256.New()
	var pending token.Token
	for {
		_, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}
		if tok == token.SEMICOLON || tok == token.COMMA {
			pending = tok
			continue
		}
		// Separators before a closer are optional: gofmt layout, not meaning.
		if pending != 0 && tok != token.RBRACE && tok != token.RPAREN {
			fmt.Fprintf(h, "%d\n", pending)
		}
		pending = 0
		fmt.Fprintf(h, "%d %s\n", tok, lit)
	}
	return fmt.Sprintf("%x", h.Sum(nil)[:8])
}

// fileChange is the file-level outcome of comparing two trees.
type fileChange struct {
	reasons []string
	goDiff  bool
	any     bool
}

func docFile(name string) bool {
	return strings.HasSuffix(name, ".md") || name == "LICENSE" ||
		strings.HasPrefix(name, "assets/") || strings.HasPrefix(name, ".ergo/")
}

func compareFiles(base, head *tree) fileChange {
	var fc fileChange
	seen := map[string]bool{}
	for _, m := range []map[string][]byte{base.files, head.files} {
		for n := range m {
			seen[n] = true
		}
	}
	names := make([]string, 0, len(seen))
	for n := range seen {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		b, inB := base.files[n]
		h, inH := head.files[n]
		if inB && inH && bytes.Equal(b, h) {
			continue
		}
		fc.any = true
		switch {
		case docFile(n):
		case isRootGo(n):
			fc.goDiff = true
			fc.reasons = append(fc.reasons, goFileReasons(n, base.parsed[n], head.parsed[n])...)
		default:
			fc.reasons = append(fc.reasons, "non-source file changed: "+n)
		}
	}
	return fc
}

func goFileReasons(n string, b, h *goFile) []string {
	var r []string
	for _, f := range []*goFile{b, h} {
		if f == nil {
			continue
		}
		if f.constrained {
			r = append(r, "build-constrained or arch-specific file: "+n)
			break
		}
		if f.generated {
			r = append(r, "generated file: "+n)
			break
		}
	}
	if b != nil && h != nil {
		if b.pkg != h.pkg {
			r = append(r, "package clause changed: "+n)
		}
		if strings.Join(b.directives, "\n") != strings.Join(h.directives, "\n") {
			r = append(r, "go: directive changed: "+n)
		}
		if strings.Join(b.sideImports, "\n") != strings.Join(h.sideImports, "\n") {
			r = append(r, "blank or dot import changed: "+n)
		}
	} else {
		f := b
		if f == nil {
			f = h
		}
		if len(f.directives) > 0 {
			r = append(r, "go: directive changed: "+n)
		}
		if len(f.sideImports) > 0 {
			r = append(r, "blank or dot import changed: "+n)
		}
	}
	return r
}

// declChanges lists decls differing between trees, split by consequence.
type declChanges struct {
	reasons []string
	lib     []string
	test    []string
	roots   []string
	// vars are changed package variables declared only in test files.
	vars []string
}

func compareDecls(base, head *tree) declChanges {
	var c declChanges
	keys := map[string]bool{}
	for k := range base.decls {
		keys[k] = true
	}
	for k := range head.decls {
		keys[k] = true
	}
	sorted := make([]string, 0, len(keys))
	for k := range keys {
		sorted = append(sorted, k)
	}
	sort.Strings(sorted)
	for _, k := range sorted {
		b, h := base.decls[k], head.decls[k]
		if b != nil && h != nil && b.hash == h.hash {
			continue
		}
		d := h
		if d == nil {
			d = b
		}
		switch d.kind {
		case declVar:
			if !strings.HasSuffix(k, ":_") && (b == nil || b.isTest && base.selfContained(b)) && (h == nil || h.isTest && head.selfContained(h)) {
				c.vars = append(c.vars, k)
			} else {
				c.reasons = append(c.reasons, "package variable changed: "+k)
			}
			continue
		case declType:
			if b != nil {
				c.reasons = append(c.reasons, "type declaration changed: "+k)
			}
			continue
		}
		name := k[strings.LastIndex(k, ":")+1:]
		if name == "init" {
			c.reasons = append(c.reasons, "init changed")
			continue
		}
		if name == "TestMain" {
			c.reasons = append(c.reasons, "TestMain changed")
			continue
		}
		// An unexported method can only satisfy an interface declared here.
		method := name[strings.LastIndex(name, ".")+1:]
		visible := d.recv != "" && (ast.IsExported(method) || base.ifaceMethods[method] || head.ifaceMethods[method])
		if visible && (b == nil || h == nil) && base.decls["type:"+typeKeyOf(k, d.recv)] != nil {
			c.reasons = append(c.reasons, "method set changed: "+k)
			continue
		}
		if visible && b != nil && h != nil && b.sig != h.sig {
			c.reasons = append(c.reasons, "method signature changed: "+k)
			continue
		}
		switch {
		case d.root != "":
			c.roots = append(c.roots, k)
		case d.isTest:
			c.test = append(c.test, k)
		default:
			c.lib = append(c.lib, k)
		}
	}
	return c
}

func typeKeyOf(methodKey, recv string) string {
	if strings.HasPrefix(methodKey, "_test:") {
		return "_test:" + recv
	}
	return recv
}
