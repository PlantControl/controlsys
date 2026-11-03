package main

import (
	"compress/gzip"
	"encoding/gob"
	"errors"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// Config is one build configuration whose file set contributes call edges.
type Config struct {
	Name   string
	GOARCH string
	Tags   string
}

// DefaultConfigs is every combination of the constraints the root package
// uses: GOARCH asm or noasm, race or norace. Changes to constrained files themselves fail open.
var DefaultConfigs = []Config{
	{Name: "arm64", GOARCH: "arm64"},
	{Name: "arm64-race", GOARCH: "arm64", Tags: "race"},
	{Name: "amd64", GOARCH: "amd64"},
	{Name: "amd64-race", GOARCH: "amd64", Tags: "race"},
}

// RootKind is the go test flag family that runs a root.
type RootKind string

const (
	KindTest  RootKind = "test"
	KindBench RootKind = "bench"
)

// Root is a function go test can start.
type Root struct {
	Name string
	Kind RootKind
	File string
}

// initRootName is an extra bit in each reach set: package initialization.
const initRootName = "(init)"

// Graph is the condensed reach relation of one tree in one configuration.
type Graph struct {
	Roots []Root
	// Reach maps a top-level function key to the bitset of Roots reaching it.
	Reach map[string][]uint64
	// Consts maps a package-level constant to its type and value.
	Consts map[string]string
	// ConstUsers maps a constant to the functions naming it.
	ConstUsers map[string][]string
	// ConstOutside lists constants named outside any function body.
	ConstOutside map[string]bool
	// TestWrites maps a test function to the package variables it may write;
	// changing it can leak into any later test reading them.
	TestWrites map[string][]string
	// GlobalReaders maps a package variable to the functions referencing it.
	GlobalReaders map[string][]string
	// Uncertain lists reasons this tree cannot be analyzed soundly.
	Uncertain []string
	// Stats is informational.
	Nodes, Edges int
}

// RootsReaching returns the names of roots reaching key; initReached reports
// package initialization reaching it; found is false for unknown keys.
func (g *Graph) RootsReaching(key string) (roots []Root, initReached, found bool) {
	bits, ok := g.Reach[key]
	if !ok {
		return nil, false, false
	}
	for i, r := range g.Roots {
		if bits[i/64]&(1<<(i%64)) == 0 {
			continue
		}
		if r.Name == initRootName {
			initReached = true
			continue
		}
		roots = append(roots, r)
	}
	return roots, initReached, true
}

// LoadGraph analyzes the module at dir, reusing cachePath when it exists.
func LoadGraph(dir string, cfg Config, cachePath string) (*Graph, error) {
	if cachePath != "" {
		if g, err := readGraph(cachePath); err == nil {
			return g, nil
		}
	}
	g, err := buildGraph(dir, cfg)
	if err != nil {
		return nil, err
	}
	if cachePath != "" {
		if err := writeGraph(cachePath, g); err != nil {
			return nil, err
		}
	}
	return g, nil
}

func readGraph(path string) (*Graph, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	g := new(Graph)
	if err := gob.NewDecoder(zr).Decode(g); err != nil {
		return nil, err
	}
	return g, nil
}

func writeGraph(path string, g *Graph) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), "graph-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	zw := gzip.NewWriter(tmp)
	if err := gob.NewEncoder(zw).Encode(g); err != nil {
		return err
	}
	if err := zw.Close(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

func buildGraph(dir string, cfg Config) (*Graph, error) {
	env := append(os.Environ(), "GOWORK=off", "CGO_ENABLED=0", "GOFLAGS=-mod=readonly")
	if cfg.GOARCH != "" {
		env = append(env, "GOARCH="+cfg.GOARCH)
	}
	lc := &packages.Config{
		Mode: packages.LoadAllSyntax,
		Dir:  dir,
		Env:  env,
		// Patterns load only the root package; nested tools are out of scope.
		Tests: true,
	}
	if cfg.Tags != "" {
		lc.BuildFlags = []string{"-tags=" + cfg.Tags}
	}
	pkgs, err := packages.Load(lc, ".")
	if err != nil {
		return nil, fmt.Errorf("load: %w", err)
	}
	if n := packages.PrintErrors(pkgs); n > 0 {
		return nil, fmt.Errorf("load: %d package errors", n)
	}
	var targets []*packages.Package
	for _, p := range pkgs {
		if strings.HasSuffix(p.ID, ".test") || !strings.HasSuffix(p.ID, ".test]") {
			continue
		}
		targets = append(targets, p)
	}
	if len(targets) == 0 {
		// A package with no test files has no test variant.
		for _, p := range pkgs {
			if !strings.HasSuffix(p.ID, ".test") {
				targets = append(targets, p)
			}
		}
	}
	if len(targets) == 0 {
		return nil, errors.New("load: no packages")
	}
	prog, ssaPkgs := ssautil.AllPackages(targets, ssa.InstantiateGenerics)
	prog.Build()

	a := &analysis{
		g: &Graph{
			Reach:         map[string][]uint64{},
			Consts:        map[string]string{},
			ConstUsers:    map[string][]string{},
			ConstOutside:  map[string]bool{},
			TestWrites:    map[string][]string{},
			GlobalReaders: map[string][]string{},
		},
		ssaPkg: map[*ssa.Package]bool{},
		prog:   prog,
	}
	for i, sp := range ssaPkgs {
		if sp == nil {
			return nil, fmt.Errorf("no ssa for %s", targets[i].ID)
		}
		a.ssaPkg[sp] = true
	}
	for _, p := range targets {
		a.collectSyntax(p)
	}
	a.collectEdges()
	a.condense()
	return a.g, nil
}

type analysis struct {
	g      *Graph
	prog   *ssa.Program
	ssaPkg map[*ssa.Package]bool
	nodes  []*ssa.Function
	id     map[*ssa.Function]int
	succ   [][]int32
	rootOf map[int][]int
}

// funcKey identifies a top-level declaration: "Name" or "Recv.Name". The
// external test package is prefixed so it cannot collide.
func funcKey(pkgPath string, recv, name string) string {
	k := name
	if recv != "" {
		k = recv + "." + name
	}
	if strings.HasSuffix(pkgPath, "_test") {
		return "_test:" + k
	}
	return k
}

func recvName(e ast.Expr) string {
	for {
		switch t := e.(type) {
		case *ast.StarExpr:
			e = t.X
		case *ast.ParenExpr:
			e = t.X
		case *ast.IndexExpr:
			e = t.X
		case *ast.IndexListExpr:
			e = t.X
		case *ast.Ident:
			return t.Name
		default:
			return ""
		}
	}
}

// isRootName mirrors go test's rule: the prefix, then end or a non-lowercase.
func isRootName(name, prefix string) bool {
	if !strings.HasPrefix(name, prefix) {
		return false
	}
	rest := name[len(prefix):]
	if rest == "" {
		return true
	}
	r, _ := utf8.DecodeRuneInString(rest)
	return !unicode.IsLower(r)
}

func classifyRoot(d *ast.FuncDecl) (RootKind, bool) {
	if d.Recv != nil || d.Name == nil {
		return "", false
	}
	n := d.Name.Name
	switch {
	case isRootName(n, "Test"), isRootName(n, "Fuzz"), isRootName(n, "Example"):
		if n == "TestMain" {
			return "", false
		}
		return KindTest, true
	case isRootName(n, "Benchmark"):
		return KindBench, true
	}
	return "", false
}

func (a *analysis) collectSyntax(p *packages.Package) {
	scope := p.Types.Scope()
	for _, name := range scope.Names() {
		if c, ok := scope.Lookup(name).(*types.Const); ok {
			a.g.Consts[funcKey(p.PkgPath, "", name)] = c.Type().String() + "=" + constKey(c.Val())
		}
	}
	users := map[string]map[string]bool{}
	for _, file := range p.Syntax {
		fname := filepath.Base(p.Fset.Position(file.Pos()).Filename)
		if strings.Contains(string(fileHeadBytes(p.Fset, file)), "import \"C\"") {
			a.g.Uncertain = append(a.g.Uncertain, "cgo in "+fname)
		}
		for _, cg := range file.Comments {
			for _, c := range cg.List {
				if strings.HasPrefix(c.Text, "//go:linkname") {
					a.g.Uncertain = append(a.g.Uncertain, "linkname in "+fname)
				}
			}
		}
		for _, decl := range file.Decls {
			owner := ""
			if fd, ok := decl.(*ast.FuncDecl); ok {
				r := ""
				if fd.Recv != nil && len(fd.Recv.List) > 0 {
					r = recvName(fd.Recv.List[0].Type)
				}
				owner = funcKey(p.PkgPath, r, fd.Name.Name)
				if kind, ok := classifyRoot(fd); ok && strings.HasSuffix(fname, "_test.go") {
					a.g.Roots = append(a.g.Roots, Root{Name: fd.Name.Name, Kind: kind, File: fname})
				}
			}
			ast.Inspect(decl, func(n ast.Node) bool {
				id, ok := n.(*ast.Ident)
				if !ok {
					return true
				}
				c, ok := p.TypesInfo.Uses[id].(*types.Const)
				if !ok || c.Pkg() == nil || c.Parent() != c.Pkg().Scope() {
					return true
				}
				if pp := c.Pkg().Path(); pp != strings.TrimSuffix(p.PkgPath, "_test") && pp != p.PkgPath {
					return true
				}
				ck := funcKey(c.Pkg().Path(), "", c.Name())
				if owner == "" {
					a.g.ConstOutside[ck] = true
				} else {
					if users[ck] == nil {
						users[ck] = map[string]bool{}
					}
					users[ck][owner] = true
				}
				return true
			})
		}
	}
	for ck, set := range users {
		for k := range set {
			a.g.ConstUsers[ck] = append(a.g.ConstUsers[ck], k)
		}
		sort.Strings(a.g.ConstUsers[ck])
	}
}

func fileHeadBytes(fset *token.FileSet, f *ast.File) []byte {
	var b strings.Builder
	for _, imp := range f.Imports {
		if imp.Path.Value == `"C"` {
			b.WriteString(`import "C"`)
		}
	}
	return []byte(b.String())
}

func constKey(v constant.Value) string {
	if v == nil {
		return "<nil>"
	}
	return v.ExactString()
}

// topLevel returns the analysis key of fn: its top-level declaration, with
// closures and generic instances mapped to it.
func (a *analysis) topLevel(fn *ssa.Function) (string, bool) {
	for fn.Parent() != nil {
		fn = fn.Parent()
	}
	if o := fn.Origin(); o != nil {
		fn = o
	}
	if fn.Pkg == nil || !a.ssaPkg[fn.Pkg] {
		return "", false
	}
	if fn.Synthetic != "" {
		return "", false
	}
	recv := ""
	if sig := fn.Signature; sig.Recv() != nil {
		t := sig.Recv().Type()
		if p, ok := t.(*types.Pointer); ok {
			t = p.Elem()
		}
		if n, ok := types.Unalias(t).(*types.Named); ok {
			recv = n.Obj().Name()
		} else {
			return "", false
		}
	}
	name := fn.Name()
	if i := strings.IndexByte(name, '['); i >= 0 {
		name = name[:i]
	}
	return funcKey(fn.Pkg.Pkg.Path(), recv, name), true
}

func (a *analysis) node(fn *ssa.Function) int32 {
	if i, ok := a.id[fn]; ok {
		return int32(i)
	}
	i := len(a.nodes)
	a.nodes = append(a.nodes, fn)
	a.id[fn] = i
	a.succ = append(a.succ, nil)
	return int32(i)
}

// collectEdges builds the sound call relation documented in doc.go.
func (a *analysis) collectEdges() {
	all := ssautil.AllFunctions(a.prog)
	a.id = make(map[*ssa.Function]int, len(all))
	funcs := make([]*ssa.Function, 0, len(all))
	for fn := range all {
		funcs = append(funcs, fn)
	}
	sort.Slice(funcs, func(i, j int) bool { return funcs[i].String() < funcs[j].String() })
	for _, fn := range funcs {
		a.node(fn)
	}

	methods := map[string][]*ssa.Function{}
	type globalUse struct{ readers, writers []*ssa.Function }
	globals := map[*ssa.Global]*globalUse{}
	var reflective []string
	testWriters := map[string]map[string]bool{}
	funcState := map[string]bool{}

	for _, fn := range funcs {
		if sig := fn.Signature; sig.Recv() != nil && !types.IsInterface(sig.Recv().Type()) {
			methods[fn.Name()] = append(methods[fn.Name()], fn)
		}
		for _, b := range fn.Blocks {
			for _, in := range b.Instrs {
				var opbuf [16]*ssa.Value
				ops := in.Operands(opbuf[:0])
				if call, ok := in.(ssa.CallInstruction); ok {
					c := call.Common()
					if !c.IsInvoke() {
						if _, isFn := c.Value.(*ssa.Function); isFn {
							ops = ops[1:]
						}
					}
					if s := c.StaticCallee(); s != nil && a.ssaPkg[fn.Pkg] && s.Pkg != nil && s.Pkg.Pkg.Path() == "reflect" {
						switch s.Name() {
						case "Call", "CallSlice", "Method", "MethodByName", "MakeFunc":
							reflective = append(reflective, fn.String()+" uses reflect."+s.Name())
						}
					}
				}
				for _, op := range ops {
					if op == nil || *op == nil {
						continue
					}
					switch v := (*op).(type) {
					case *ssa.Function:
						a.addEdge(fn, v)
					case *ssa.Global:
						if !a.ssaPkg[v.Pkg] {
							continue
						}
						u := globals[v]
						if u == nil {
							u = &globalUse{}
							globals[v] = u
						}
						u.readers = append(u.readers, fn)
						if load, isLoad := in.(*ssa.UnOp); !isLoad || mutatesThrough(load, map[ssa.Value]bool{}) {
							u.writers = append(u.writers, fn)
						}
					}
				}
			}
		}
	}

	for _, fn := range funcs {
		for _, b := range fn.Blocks {
			for _, in := range b.Instrs {
				call, ok := in.(ssa.CallInstruction)
				if !ok {
					continue
				}
				c := call.Common()
				if c.IsInvoke() {
					want := sigKey(c.Method.Type().(*types.Signature))
					for _, m := range methods[c.Method.Name()] {
						if m.TypeParams() != nil || m.Signature.Recv() != nil && hasTypeParam(m.Signature) || sigKey(m.Signature) == want {
							a.addEdge(fn, m)
						}
					}
					continue
				}
				if s := c.StaticCallee(); s != nil {
					a.addEdge(fn, s)
					continue
				}
			}
		}
	}

	for g, u := range globals {
		var writers []*ssa.Function
		for _, w := range u.writers {
			if !isInitializer(w) && storesFunc(g, w) {
				funcState[g.String()] = true
			}
			switch {
			case isInitializer(w):
			case a.inTestFile(w):
				if k, ok := a.topLevel(w); ok {
					if testWriters[k] == nil {
						testWriters[k] = map[string]bool{}
					}
					testWriters[k][g.String()] = true
				}
			default:
				writers = append(writers, w)
			}
		}
		readerKeys := map[string]bool{}
		for _, r := range u.readers {
			if isInitializer(r) {
				continue
			}
			if k, ok := a.topLevel(r); ok {
				readerKeys[k] = true
			}
			for _, w := range writers {
				if r != w {
					a.addEdge(r, w)
				}
			}
		}
		for k := range readerKeys {
			a.g.GlobalReaders[g.String()] = append(a.g.GlobalReaders[g.String()], k)
		}
		sort.Strings(a.g.GlobalReaders[g.String()])
	}
	for k := range funcState {
		a.g.Uncertain = append(a.g.Uncertain, "function value stored in package variable after init: "+k)
	}
	for k, set := range testWriters {
		for g := range set {
			a.g.TestWrites[k] = append(a.g.TestWrites[k], g)
		}
		sort.Strings(a.g.TestWrites[k])
	}
	sort.Strings(reflective)
	a.g.Uncertain = append(a.g.Uncertain, reflective...)
}

// storesFunc reports whether writer w may put a function value into g, which
// would let one root call code that another root referenced: storing or
// map-updating a func-carrying value (any value, when g's type has a func
// component) through g, passing one to a method of g, or handing a reference
// into g to a function that does so. Callees outside the analyzed packages
// are assumed not to store function values into their arguments; dynamic
// callees are assumed to.
func storesFunc(g *ssa.Global, w *ssa.Function) bool {
	direct, iface := holdsFunc(g.Type(), map[types.Type]bool{})
	if !direct && !iface {
		return false
	}
	return storesFuncVia(w, g, direct, g.Pkg, map[*ssa.Function]bool{})
}

func storesFuncVia(w *ssa.Function, base ssa.Value, direct bool, pkg *ssa.Package, seen map[*ssa.Function]bool) bool {
	if seen[w] {
		return false
	}
	seen[w] = true
	carries := func(v ssa.Value) bool {
		if d, _ := holdsFunc(v.Type(), map[types.Type]bool{}); d {
			return true
		}
		if mi, ok := v.(*ssa.MakeInterface); ok {
			d, i := holdsFunc(mi.X.Type(), map[types.Type]bool{})
			return d || i
		}
		return false
	}
	from := func(v ssa.Value) bool { return derivesFrom(v, base, map[ssa.Value]bool{}) }
	for _, b := range w.Blocks {
		for _, in := range b.Instrs {
			switch x := in.(type) {
			case *ssa.Store:
				if from(x.Addr) && (direct || carries(x.Val)) {
					return true
				}
			case *ssa.MapUpdate:
				if from(x.Map) && (direct || carries(x.Value) || carries(x.Key)) {
					return true
				}
			case ssa.CallInstruction:
				c := x.Common()
				if c.IsInvoke() && from(c.Value) {
					for _, a := range c.Args {
						if carries(a) {
							return true
						}
					}
				}
				if b, ok := c.Value.(*ssa.Builtin); ok {
					if (b.Name() == "copy" || b.Name() == "append") && from(c.Args[0]) && (direct || len(c.Args) > 1 && carries(c.Args[1])) {
						return true
					}
					continue
				}
				callee := c.StaticCallee()
				for i, a := range c.Args {
					if !aliases(a.Type(), map[types.Type]bool{}) || !from(a) {
						continue
					}
					switch {
					case callee == nil:
						if direct {
							return true
						}
					case callee.Pkg == nil || callee.Pkg.Pkg.Path() != pkg.Pkg.Path() && callee.Pkg.Pkg.Path() != pkg.Pkg.Path()+"_test":
						// Unanalyzed callee handed a reference and a function may keep both,
						// except calls that only run their function argument.
						if isRunOnly(callee) {
							continue
						}
						for j, o := range c.Args {
							if j != i && carries(o) {
								return true
							}
						}
					case len(callee.Params) > i && storesFuncVia(callee, callee.Params[i], direct, pkg, seen):
						return true
					}
				}
			}
		}
	}
	return false
}

// holdsFunc reports whether values of t contain a func-typed component
// (direct) or an interface component (iface).
func holdsFunc(t types.Type, seen map[types.Type]bool) (direct, iface bool) {
	if seen[t] {
		return false, false
	}
	seen[t] = true
	or := func(d, i bool) {
		direct, iface = direct || d, iface || i
	}
	// sync and sync/atomic keep private function fields of their own; user
	// values enter only as any (Map, Value) or type arguments (Pointer[T]).
	if n, ok := types.Unalias(t).(*types.Named); ok && n.Obj().Pkg() != nil {
		switch n.Obj().Pkg().Path() {
		case "sync", "sync/atomic":
			switch n.Obj().Name() {
			case "Pool":
				return true, false
			case "Map", "Value":
				return false, true
			}
			for i := range n.TypeArgs().Len() {
				or(holdsFunc(n.TypeArgs().At(i), seen))
			}
			return direct, iface
		}
	}
	switch u := t.Underlying().(type) {
	case *types.Signature:
		return true, false
	case *types.Interface:
		return false, true
	case *types.Pointer:
		or(holdsFunc(u.Elem(), seen))
	case *types.Slice:
		or(holdsFunc(u.Elem(), seen))
	case *types.Array:
		or(holdsFunc(u.Elem(), seen))
	case *types.Map:
		or(holdsFunc(u.Key(), seen))
		or(holdsFunc(u.Elem(), seen))
	case *types.Chan:
		or(holdsFunc(u.Elem(), seen))
	case *types.Struct:
		for i := range u.NumFields() {
			or(holdsFunc(u.Field(i).Type(), seen))
		}
	}
	return direct, iface
}

// mutatesThrough reports whether the value v, loaded from a package variable,
// may be used to modify shared state: a map or element update, a store of
// it or through it, a send, capture, or any non-builtin call receiving it.
func mutatesThrough(v ssa.Value, seen map[ssa.Value]bool) bool {
	if seen[v] {
		return false
	}
	seen[v] = true
	refs := v.Referrers()
	if refs == nil {
		return false
	}
	for _, r := range *refs {
		switch x := r.(type) {
		case *ssa.MapUpdate:
			return true
		case *ssa.Store:
			return true
		case *ssa.Send, *ssa.MakeClosure, *ssa.Go, *ssa.Defer:
			return true
		case *ssa.Call:
			if b, ok := x.Call.Value.(*ssa.Builtin); ok {
				switch b.Name() {
				case "len", "cap", "print", "println":
					continue
				}
			}
			if x.Call.Value == v {
				if x.Call.IsInvoke() {
					return true
				}
				continue
			}
			// Arguments are copies; only references inside them can be written.
			if aliases(v.Type(), map[types.Type]bool{}) {
				return true
			}
		case *ssa.IndexAddr, *ssa.FieldAddr, *ssa.Slice, *ssa.ChangeType, *ssa.MakeInterface, *ssa.Convert,
			*ssa.Field, *ssa.Index, *ssa.Lookup, *ssa.Phi, *ssa.TypeAssert, *ssa.Extract, *ssa.SliceToArrayPointer:
			if mutatesThrough(x.(ssa.Value), seen) {
				return true
			}
		case *ssa.UnOp:
			if x.Op == token.MUL {
				if _, isPtr := x.X.Type().Underlying().(*types.Pointer); isPtr && !isValueOnly(x.Type()) {
					if mutatesThrough(x, seen) {
						return true
					}
				}
			}
		}
	}
	return false
}

// aliases reports whether a copy of a t value shares memory with the original.
// Interface values count as not aliasing: their methods are the invoke case.
func aliases(t types.Type, seen map[types.Type]bool) bool {
	if seen[t] {
		return false
	}
	seen[t] = true
	switch u := t.Underlying().(type) {
	case *types.Pointer, *types.Slice, *types.Map, *types.Chan:
		return true
	case *types.Array:
		return aliases(u.Elem(), seen)
	case *types.Struct:
		for i := range u.NumFields() {
			if aliases(u.Field(i).Type(), seen) {
				return true
			}
		}
	}
	return false
}

// isValueOnly reports whether t holds no references, so copies cannot alias.
func isValueOnly(t types.Type) bool {
	switch u := t.Underlying().(type) {
	case *types.Basic:
		return true
	case *types.Array:
		return isValueOnly(u.Elem())
	case *types.Struct:
		for i := range u.NumFields() {
			if !isValueOnly(u.Field(i).Type()) {
				return false
			}
		}
		return true
	}
	return false
}

func isInitializer(fn *ssa.Function) bool {
	for fn.Parent() != nil {
		fn = fn.Parent()
	}
	return fn.Signature.Recv() == nil && (fn.Name() == "init" || strings.HasPrefix(fn.Name(), "init#"))
}

func (a *analysis) inTestFile(fn *ssa.Function) bool {
	return strings.HasSuffix(a.prog.Fset.Position(fn.Pos()).Filename, "_test.go")
}

func hasTypeParam(sig *types.Signature) bool {
	return sig.RecvTypeParams() != nil && sig.RecvTypeParams().Len() > 0
}

func sigKey(sig *types.Signature) string {
	return types.TypeString(types.NewSignatureType(nil, nil, nil, sig.Params(), sig.Results(), sig.Variadic()), nil)
}

func (a *analysis) addEdge(from, to *ssa.Function) {
	i, j := a.node(from), a.node(to)
	a.succ[i] = append(a.succ[i], j)
}

// condense computes, for each declared function, the set of roots reaching it.
func (a *analysis) condense() {
	n := len(a.nodes)
	comp, order := tarjan(n, a.succ)
	ncomp := len(order)

	rootIdx := map[string]int{}
	var uniq []Root
	for _, r := range a.g.Roots {
		if _, dup := rootIdx[r.Name]; !dup {
			rootIdx[r.Name] = len(uniq)
			uniq = append(uniq, r)
		}
	}
	a.g.Roots = uniq
	initBit := len(a.g.Roots)
	a.g.Roots = append(a.g.Roots, Root{Name: initRootName})
	words := (len(a.g.Roots) + 63) / 64
	reach := make([][]uint64, ncomp)
	get := func(c int32) []uint64 {
		if reach[c] == nil {
			reach[c] = make([]uint64, words)
		}
		return reach[c]
	}
	for i, fn := range a.nodes {
		if fn.Parent() != nil || fn.Pkg == nil || !a.ssaPkg[fn.Pkg] {
			continue
		}
		if fn.Signature.Recv() == nil && fn.Synthetic != "" && strings.HasPrefix(fn.Synthetic, "package initializer") {
			get(comp[i])[initBit/64] |= 1 << (initBit % 64)
			continue
		}
		if fn.Signature.Recv() != nil {
			continue
		}
		name := fn.Name()
		if b := strings.IndexByte(name, '['); b >= 0 {
			name = name[:b]
		}
		if j, ok := rootIdx[name]; ok && fn.TypeParams() == nil {
			get(comp[i])[j/64] |= 1 << (j % 64)
		}
	}
	// A package's init functions are reached through its synthetic initializer.
	edges := 0
	members := make([][]int32, ncomp)
	for i := range a.nodes {
		members[comp[i]] = append(members[comp[i]], int32(i))
	}
	// order is reverse topological; walking it backwards visits predecessors first.
	for oi := ncomp - 1; oi >= 0; oi-- {
		c := order[oi]
		rs := reach[c]
		for _, m := range members[c] {
			edges += len(a.succ[m])
			if rs == nil {
				continue
			}
			for _, s := range a.succ[m] {
				sc := comp[s]
				if sc == c {
					continue
				}
				dst := get(sc)
				for w := range rs {
					dst[w] |= rs[w]
				}
			}
		}
	}
	a.g.Nodes, a.g.Edges = n, edges
	empty := make([]uint64, words)
	for i, fn := range a.nodes {
		key, ok := a.topLevel(fn)
		if !ok {
			continue
		}
		cur, ok := a.g.Reach[key]
		if !ok {
			cur = append([]uint64(nil), empty...)
			a.g.Reach[key] = cur
		}
		if rs := reach[comp[i]]; rs != nil {
			for w := range rs {
				cur[w] |= rs[w]
			}
		}
	}
}

// tarjan returns each node's component and the components in reverse
// topological order (a component precedes every component that reaches it).
func tarjan(n int, succ [][]int32) (comp []int32, order []int32) {
	comp = make([]int32, n)
	index := make([]int32, n)
	low := make([]int32, n)
	on := make([]bool, n)
	for i := range comp {
		comp[i] = -1
	}
	var stack []int32
	type frame struct {
		v int32
		i int
	}
	next := int32(1)
	for s := range n {
		if index[s] != 0 {
			continue
		}
		call := []frame{{int32(s), 0}}
		index[s], low[s] = next, next
		next++
		stack = append(stack, int32(s))
		on[s] = true
		for len(call) > 0 {
			f := &call[len(call)-1]
			if f.i < len(succ[f.v]) {
				w := succ[f.v][f.i]
				f.i++
				if index[w] == 0 {
					index[w], low[w] = next, next
					next++
					stack = append(stack, w)
					on[w] = true
					call = append(call, frame{w, 0})
				} else if on[w] && index[w] < low[f.v] {
					low[f.v] = index[w]
				}
				continue
			}
			v := f.v
			call = call[:len(call)-1]
			if len(call) > 0 {
				p := call[len(call)-1].v
				if low[v] < low[p] {
					low[p] = low[v]
				}
			}
			if low[v] == index[v] {
				c := int32(len(order))
				for {
					w := stack[len(stack)-1]
					stack = stack[:len(stack)-1]
					on[w] = false
					comp[w] = c
					if w == v {
						break
					}
				}
				order = append(order, c)
			}
		}
	}
	return comp, order
}

func isRunOnly(fn *ssa.Function) bool {
	if fn.Pkg == nil {
		return false
	}
	switch fn.Pkg.Pkg.Path() + "." + fn.Name() {
	case "sync.Range", "sync.Do":
		return true
	}
	return false
}

func derivesFrom(v ssa.Value, base ssa.Value, seen map[ssa.Value]bool) bool {
	if v == base {
		return true
	}
	if seen[v] {
		return false
	}
	seen[v] = true
	switch x := v.(type) {
	case *ssa.UnOp:
		return derivesFrom(x.X, base, seen)
	case *ssa.FieldAddr:
		return derivesFrom(x.X, base, seen)
	case *ssa.IndexAddr:
		return derivesFrom(x.X, base, seen)
	case *ssa.Field:
		return derivesFrom(x.X, base, seen)
	case *ssa.Index:
		return derivesFrom(x.X, base, seen)
	case *ssa.Slice:
		return derivesFrom(x.X, base, seen)
	case *ssa.Lookup:
		return derivesFrom(x.X, base, seen)
	case *ssa.ChangeType:
		return derivesFrom(x.X, base, seen)
	case *ssa.TypeAssert:
		return derivesFrom(x.X, base, seen)
	case *ssa.Phi:
		for _, e := range x.Edges {
			if derivesFrom(e, base, seen) {
				return true
			}
		}
	}
	return false
}
