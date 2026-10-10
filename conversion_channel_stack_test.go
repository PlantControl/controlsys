package controlsys

import (
	"fmt"
	"math"
	"math/rand/v2"
	"reflect"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestConversionChannelStackMatchesParallelFold(t *testing.T) {
	rnd := rand.New(rand.NewPCG(7, 7))
	value := func() float64 {
		switch rnd.IntN(6) {
		case 0:
			return 0
		case 1:
			return math.Copysign(0, -1)
		}
		return rnd.NormFloat64()
	}
	dense := func(r, c int) *mat.Dense {
		d := newDense(r, c)
		for i := range r {
			for j := range c {
				d.Set(i, j, value())
			}
		}
		return d
	}
	channel := func(dt float64) *System {
		n, count := rnd.IntN(4), rnd.IntN(3)
		sys := &System{A: dense(n, n), B: dense(n, 1), C: dense(1, n), D: dense(1, 1), Dt: dt}
		if count > 0 || rnd.IntN(4) == 0 {
			sys.LFT = &LFTDelay{B2: dense(n, count), C2: dense(count, n), D12: dense(1, count), D21: dense(count, 1), D22: dense(count, count)}
			for range count {
				sys.LFT.Tau = append(sys.LFT.Tau, float64(1+rnd.IntN(5)))
			}
		}
		return sys
	}
	for trial := range 300 {
		m, p := 1+rnd.IntN(3), 1+rnd.IntN(3)
		stack := conversionChannelStack{m: m, p: p}
		var parts []conversionChannel
		for i := range p {
			for j := range m {
				ch := channel(0.1)
				stack.add(ch, i, j)
				parts = append(parts, conversionChannel{sys: ch, output: i, input: j})
			}
		}
		want, err := foldConversionChannels(parts, m, p)
		if err != nil {
			t.Fatalf("trial %d: fold: %v", trial, err)
		}
		if got, wantFP := systemFingerprint(stack.system()), systemFingerprint(want); got != wantFP {
			t.Fatalf("trial %d (p=%d m=%d):\ngot  %s\nwant %s", trial, p, m, got, wantFP)
		}
	}
}

func TestConversionChannelStackMatchesParallelFoldDiscretized(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), mat.NewDense(2, 2, []float64{1, .5, -.3, 1}), mat.NewDense(2, 2, []float64{1, -.2, .4, 1}), mat.NewDense(2, 2, []float64{0, .1, -.2, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{.035, .1}
	sys.OutputDelay = []float64{.2, .015}
	sys.Delay = mat.NewDense(2, 2, []float64{0, .12, .05, 0})
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodImpulse} {
		stack := conversionChannelStack{m: 2, p: 2}
		var parts []conversionChannel
		for i := range 2 {
			for j := range 2 {
				disc, err := discretizeConversionChannel(sys, i, j, .1, C2DOptions{Method: method})
				if err != nil {
					t.Fatalf("%s: %v", method, err)
				}
				stack.add(disc, i, j)
				parts = append(parts, conversionChannel{sys: disc, output: i, input: j})
			}
		}
		want, err := foldConversionChannels(parts, 2, 2)
		if err != nil {
			t.Fatalf("%s: fold: %v", method, err)
		}
		if got, wantFP := systemFingerprint(stack.system()), systemFingerprint(want); got != wantFP {
			t.Fatalf("%s:\ngot  %s\nwant %s", method, got, wantFP)
		}
	}
}

// foldConversionChannels is the reference assembly conversionChannelStack
// replaces: Parallel folded over each channel's p×m embedding.
func foldConversionChannels(parts []conversionChannel, m, p int) (*System, error) {
	var combined *System
	for _, part := range parts {
		expanded := embedConversionChannel(part.sys, m, p, part.output, part.input)
		if combined == nil {
			combined = expanded
			continue
		}
		var err error
		if combined, err = Parallel(combined, expanded); err != nil {
			return nil, err
		}
	}
	return combined, nil
}

func embedConversionChannel(sys *System, m, p, output, input int) *System {
	n, _, _ := sys.Dims()
	out := &System{A: denseCopy(sys.A), B: newDense(n, m), C: newDense(p, n), D: newDense(p, m), Dt: sys.Dt}
	for i := range n {
		out.B.Set(i, input, sys.B.At(i, 0))
		out.C.Set(output, i, sys.C.At(0, i))
	}
	out.D.Set(output, input, sys.D.At(0, 0))
	if sys.LFT != nil {
		count := len(sys.LFT.Tau)
		out.LFT = &LFTDelay{Tau: append([]float64(nil), sys.LFT.Tau...), B2: denseCopy(sys.LFT.B2), C2: denseCopy(sys.LFT.C2), D12: newDense(p, count), D21: newDense(count, m), D22: denseCopy(sys.LFT.D22)}
		for k := range count {
			out.LFT.D12.Set(output, k, sys.LFT.D12.At(0, k))
			out.LFT.D21.Set(k, input, sys.LFT.D21.At(k, 0))
		}
	}
	return out
}

// systemFingerprint renders every field of sys exactly, including float bit
// patterns, nil versus empty slices and nil versus empty matrices.
func systemFingerprint(sys *System) string {
	var sb strings.Builder
	var walk func(v reflect.Value)
	walk = func(v reflect.Value) {
		switch x := v.Interface().(type) {
		case *mat.Dense:
			if x == nil {
				sb.WriteString("nil")
				return
			}
			r, c := x.Dims()
			fmt.Fprintf(&sb, "%dx%d[", r, c)
			for i := range r {
				for j := range c {
					fmt.Fprintf(&sb, "%x ", math.Float64bits(x.At(i, j)))
				}
			}
			sb.WriteString("]")
			return
		case []float64:
			if x == nil {
				sb.WriteString("nil")
				return
			}
			sb.WriteString("[")
			for _, f := range x {
				fmt.Fprintf(&sb, "%x ", math.Float64bits(f))
			}
			sb.WriteString("]")
			return
		case float64:
			fmt.Fprintf(&sb, "%x", math.Float64bits(x))
			return
		}
		switch v.Kind() {
		case reflect.Pointer:
			if v.IsNil() {
				sb.WriteString("nil")
				return
			}
			walk(v.Elem())
		case reflect.Struct:
			sb.WriteString("{")
			for i := range v.NumField() {
				if !v.Type().Field(i).IsExported() {
					continue
				}
				fmt.Fprintf(&sb, "%s:", v.Type().Field(i).Name)
				walk(v.Field(i))
				sb.WriteString(" ")
			}
			sb.WriteString("}")
		default:
			fmt.Fprintf(&sb, "%#v", v.Interface())
		}
	}
	walk(reflect.ValueOf(sys))
	return sb.String()
}
