package controlsys

import (
	"errors"
	"go/ast"
	"go/parser"
	gotoken "go/token"
	"io/fs"
	"math"
	"math/rand/v2"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"plantcontrol.org/v1/gonum/mat"
)

// returnsWithinDeadline panics when f does not return in time. A hung Exp spins
// until the process exits, so the run stops here rather than at later tests.
func returnsWithinDeadline(t *testing.T, name string, f func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		f()
	}()
	select {
	case <-done:
	case <-time.After(time.Minute):
		panic(name + ": still running at the deadline")
	}
}

func nonfiniteExpPlant(t *testing.T, a []float64, dt float64) *System {
	t.Helper()
	sys, err := New(mat.NewDense(2, 2, a), mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(1, 1, []float64{0}), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

var (
	nonfiniteExpSeed   = []float64{1e5, 1, 0, .5}
	nonfiniteExpRot    = []float64{1.5289, 1.9267, -1.9267, 1.5289}
	nonfiniteExpStable = []float64{-1, 2, 0, -3}
	nonfiniteExpHuge   = []float64{1e308, 1e308, -1e308, 1e308}
)

func TestC2DNonFiniteExpNormReturnsErrOverflow(t *testing.T) {
	for _, tc := range []struct {
		name string
		a    []float64
		dt   float64
	}{
		{"huge at unit dt", nonfiniteExpHuge, 1},
		{"huge at 1e30", nonfiniteExpHuge, 1e30},
		{"rotation at 1e308", nonfiniteExpRot, 1e308},
		{"stable at 1e308", nonfiniteExpStable, 1e308},
	} {
		for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodImpulse} {
			t.Run(tc.name+" "+string(method), func(t *testing.T) {
				sys := nonfiniteExpPlant(t, tc.a, 0)
				returnsWithinDeadline(t, "C2D", func() {
					if _, err := sys.C2D(tc.dt, C2DOptions{Method: method}); !errors.Is(err, ErrOverflow) {
						t.Errorf("C2D err = %v, want ErrOverflow", err)
					}
				})
				returnsWithinDeadline(t, "C2DMap", func() {
					if _, _, err := sys.C2DMap(tc.dt, C2DOptions{Method: method}); !errors.Is(err, ErrOverflow) {
						t.Errorf("C2DMap err = %v, want ErrOverflow", err)
					}
				})
			})
		}
	}
}

func TestD2CNonFiniteResultReturnsErrOverflow(t *testing.T) {
	for _, tc := range []struct {
		name   string
		a      []float64
		dt     float64
		method C2DMethod
	}{
		{"foh of the Exp hang", nonfiniteExpSeed, 1e-308, C2DMethodFOH},
		{"zoh Inf A", nonfiniteExpSeed, 1e-308, C2DMethodZOH},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys := nonfiniteExpPlant(t, tc.a, tc.dt)
			returnsWithinDeadline(t, "D2C", func() {
				if _, err := sys.D2C(D2COptions{Method: tc.method}); !errors.Is(err, ErrOverflow) {
					t.Errorf("D2C err = %v, want ErrOverflow", err)
				}
			})
			returnsWithinDeadline(t, "D2CMap", func() {
				if _, _, err := sys.D2CMap(D2COptions{Method: tc.method}); !errors.Is(err, ErrOverflow) {
					t.Errorf("D2CMap err = %v, want ErrOverflow", err)
				}
			})
		})
	}
}

// On amd64 these reach the exit with NaN B, D and G; natively their reconstruction
// fails as singular first. Neither may return a non-finite model with nil error.
func TestD2CReconstructionResultFiniteOrError(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 777))
	plant := stiffRotatedDescriptor(t, rng, []float64{-1, 1e10, 1, 0, -2, 1, 0, 0, -1}, []float64{1, 1, 1e-13}, []float64{1, 1.3, 1.6}, []float64{1, .8, .6})
	pencil, err := plant.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for name, sys := range map[string]*System{"stiff pencil": pencil, "huge": nonfiniteExpPlant(t, nonfiniteExpHuge, .1)} {
		for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodTustin, C2DMethodFOH} {
			t.Run(name+" "+string(method), func(t *testing.T) {
				out, g, err := sys.D2CMap(D2COptions{Method: method})
				if err == nil {
					if ferr := finiteSystemMatrices("result", out, ErrOverflow); ferr != nil {
						t.Errorf("D2CMap nil error with %v", ferr)
					}
					if ferr := finiteDense("D2CMap", "G", g, ErrOverflow); ferr != nil {
						t.Errorf("D2CMap nil error with %v", ferr)
					}
				}
				if out, err = sys.D2C(D2COptions{Method: method}); err == nil {
					if ferr := finiteSystemMatrices("result", out, ErrOverflow); ferr != nil {
						t.Errorf("D2C nil error with %v", ferr)
					}
				}
			})
		}
	}
}

func TestD2CFOHTinyDtNeighborValues(t *testing.T) {
	sys := nonfiniteExpPlant(t, nonfiniteExpSeed, 1e-300)
	out, g, err := sys.D2CMap(D2COptions{Method: C2DMethodFOH})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		m    *mat.Dense
		want []float64
	}{
		{"A", out.A, []float64{1.1512925464970227e+301, 1.2206133676198554e+296, 0, -6.931471805599452e+299}},
		{"B", out.B, []float64{-1.92049615048854e+295, 1.92181205567280e+300}},
		{"D", out.D, []float64{-0.77259099536260}},
		{"G", g, []float64{1, 0, 2.27312282338e-06, 0, 1, 0.77258872223978}},
	} {
		raw := c.m.RawMatrix()
		for i, w := range c.want {
			got := raw.Data[i]
			if math.Abs(got-w) > 1e-10*math.Abs(w) {
				t.Errorf("%s[%d] = %g, want %g", c.name, i, got, w)
			}
		}
	}
}

func TestResponsesNonFiniteExpNormReturnNaN(t *testing.T) {
	sys := nonfiniteExpPlant(t, nonfiniteExpHuge, 0)
	x0 := mat.NewVecDense(2, []float64{1, 1})
	for _, tc := range []struct {
		name string
		call func() (*TimeResponse, error)
	}{
		{"Step", func() (*TimeResponse, error) { return Step(sys, 1) }},
		{"Impulse", func() (*TimeResponse, error) { return Impulse(sys, 1) }},
		{"Initial", func() (*TimeResponse, error) { return Initial(sys, x0, 1) }},
		{"Lsim", func() (*TimeResponse, error) {
			return Lsim(sys, mat.NewDense(3, 1, []float64{1, 1, 1}), []float64{0, 1, 2}, nil)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			returnsWithinDeadline(t, tc.name, func() {
				resp, err := tc.call()
				if err != nil {
					t.Errorf("err = %v, want nil", err)
					return
				}
				if requireFiniteDense("resp", "Y", resp.Y) == nil {
					t.Errorf("all samples finite, want non-finite past t=0")
				}
			})
		})
	}
}

// e^{a·dt} underflows to the exact 0 of the 1×1 path of Exp when a·dt is -Inf.
func TestScalarStableInfiniteExponentKeepsFiniteValues(t *testing.T) {
	scalar := func(t *testing.T, a float64, m int) *System {
		t.Helper()
		var b, d *mat.Dense
		if m > 0 {
			b = mat.NewDense(1, m, []float64{1, 1}[:m])
			d = mat.NewDense(1, m, nil)
		}
		sys, err := New(mat.NewDense(1, 1, []float64{a}), b, mat.NewDense(1, 1, []float64{1}), d, 0)
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	for _, tc := range []struct{ a, dt float64 }{{-10, 1e308}, {-1e308, 1e30}, {-1e308, 1e308}} {
		t.Run("C2D no input FOH", func(t *testing.T) {
			out, err := scalar(t, tc.a, 0).C2D(tc.dt, C2DOptions{Method: C2DMethodFOH})
			if err != nil {
				t.Fatal(err)
			}
			if got, c := out.A.At(0, 0), out.C.At(0, 0); got != 0 || c != 1 {
				t.Errorf("a=%g dt=%g: A=%g C=%g, want 0, 1", tc.a, tc.dt, got, c)
			}
		})
	}
	for _, tFinal := range []float64{1e30, 1e308} {
		for _, delay := range []bool{false, true} {
			t.Run("Initial", func(t *testing.T) {
				sys := scalar(t, -1e308, 1)
				if delay {
					if err := sys.SetOutputDelay([]float64{.5}); err != nil {
						t.Fatal(err)
					}
				}
				resp, err := Initial(sys, mat.NewVecDense(1, []float64{1}), tFinal)
				if err != nil {
					t.Fatal(err)
				}
				_, steps := resp.Y.Dims()
				for k := range steps {
					want := 0.0
					if k == 0 {
						want = 1
					}
					if got := resp.Y.At(0, k); got != want {
						t.Fatalf("tFinal=%g delay=%v: Y[%d]=%g, want %g", tFinal, delay, k, got, want)
					}
				}
			})
		}
	}
	for _, tc := range []struct{ a, tau, want float64 }{{-10, 1e308, .1}, {-1e308, 1e30, 0}, {-1e308, 1e308, 0}} {
		t.Run("Covar two input delays", func(t *testing.T) {
			sys := scalar(t, tc.a, 2)
			if err := sys.SetInputDelay([]float64{0, tc.tau}); err != nil {
				t.Fatal(err)
			}
			p, err := Covar(sys, mat.NewDense(2, 2, []float64{1, 0, 0, 1}))
			if err != nil {
				t.Fatal(err)
			}
			if got := p.At(0, 0); got != tc.want {
				t.Errorf("a=%g tau=%g: P=%g, want %g", tc.a, tc.tau, got, tc.want)
			}
		})
	}
}

func TestExpCallSitesUseGuard(t *testing.T) {
	fset := gotoken.NewFileSet()
	var bypass []string
	guarded := 0
	err := filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return err
		}
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return err
		}
		for _, decl := range file.Decls {
			inGuard := false
			if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "expGuarded" {
				inGuard = true
			}
			ast.Inspect(decl, func(n ast.Node) bool {
				sel, ok := n.(*ast.SelectorExpr)
				if !ok || sel.Sel.Name != "Exp" {
					return true
				}
				if pkg, ok := sel.X.(*ast.Ident); ok && (pkg.Name == "math" || pkg.Name == "cmplx") {
					return true
				}
				if inGuard {
					guarded++
				} else {
					bypass = append(bypass, fset.Position(sel.Pos()).String())
				}
				return true
			})
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(bypass) > 0 {
		t.Errorf("Exp calls outside expGuarded: %s", strings.Join(bypass, ", "))
	}
	if guarded != 1 {
		t.Errorf("Exp calls in expGuarded = %d, want 1", guarded)
	}
}
