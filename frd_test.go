package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"reflect"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestFRDCopyIsIndependent(t *testing.T) {
	frd := &FRD{
		Response:   [][][]complex128{{{1 + 2i}}},
		Omega:      []float64{3},
		Dt:         0.1,
		InputName:  []string{"u"},
		OutputName: []string{"y"},
	}

	cp := frd.Copy()
	frd.Response[0][0][0] = 99
	frd.Omega[0] = 99
	frd.InputName[0] = "mutated"
	frd.OutputName[0] = "mutated"

	if cp.Response[0][0][0] != 1+2i || cp.Omega[0] != 3 {
		t.Fatalf("Copy aliases response storage: %+v", cp)
	}
	if cp.InputName[0] != "u" || cp.OutputName[0] != "y" {
		t.Fatalf("Copy aliases names: %+v", cp)
	}
}

func TestFRD_SISOFromSystem(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := []float64{0.1, 1, 10}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}

	p, m := frd.Dims()
	if p != 1 || m != 1 {
		t.Fatalf("Dims() = (%d,%d), want (1,1)", p, m)
	}
	if frd.NumFrequencies() != 3 {
		t.Fatalf("NumFrequencies() = %d, want 3", frd.NumFrequencies())
	}
	if !frd.IsContinuous() {
		t.Fatal("expected continuous")
	}

	wantH := []complex128{
		complex(1, 0) / complex(1, 0.1),
		complex(1, 0) / complex(1, 1),
		complex(1, 0) / complex(1, 10),
	}
	wantMag := []float64{
		1.0 / math.Sqrt(1.01),
		1.0 / math.Sqrt(2),
		1.0 / math.Sqrt(101),
	}

	for k := range omega {
		h := frd.At(k, 0, 0)
		if cmplx.Abs(h-wantH[k]) > 1e-10 {
			t.Errorf("w=%v: H=%v, want %v", omega[k], h, wantH[k])
		}
		mag := cmplx.Abs(h)
		if math.Abs(mag-wantMag[k]) > 1e-10 {
			t.Errorf("w=%v: |H|=%v, want %v", omega[k], mag, wantMag[k])
		}
	}
}

func TestFRD_MIMOFromSystem(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := []float64{1.0}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}

	p, m := frd.Dims()
	if p != 2 || m != 2 {
		t.Fatalf("Dims() = (%d,%d), want (2,2)", p, m)
	}

	// H(j) = (jI - A)^{-1} since C=I, B=I, D=0
	// jI - A = [j, -1; 2, j+3]
	// det = j(j+3) + 2 = -1+3j+2 = 1+3j
	det := complex(1, 3)
	want := [2][2]complex128{
		{complex(0, 1) + 3, 1},
		{-2, complex(0, 1)},
	}
	for i := range 2 {
		for j := range 2 {
			want[i][j] /= det
		}
	}

	for i := range 2 {
		for j := range 2 {
			got := frd.At(0, i, j)
			if cmplx.Abs(got-want[i][j]) > 1e-10 {
				t.Errorf("H[%d][%d] = %v, want %v", i, j, got, want[i][j])
			}
		}
	}
}

func TestFRD_DirectConstruction(t *testing.T) {
	resp := [][][]complex128{
		{{0.5 - 0.5i}},
		{{0.01 - 0.1i}},
	}
	omega := []float64{1, 10}
	frd, err := NewFRD(resp, omega, 0)
	if err != nil {
		t.Fatal(err)
	}

	p, m := frd.Dims()
	if p != 1 || m != 1 {
		t.Fatalf("Dims() = (%d,%d), want (1,1)", p, m)
	}
	if frd.NumFrequencies() != 2 {
		t.Fatalf("NumFrequencies() = %d, want 2", frd.NumFrequencies())
	}

	if frd.At(0, 0, 0) != 0.5-0.5i {
		t.Errorf("At(0,0,0) = %v, want 0.5-0.5i", frd.At(0, 0, 0))
	}
	if frd.At(1, 0, 0) != 0.01-0.1i {
		t.Errorf("At(1,0,0) = %v, want 0.01-0.1i", frd.At(1, 0, 0))
	}
}

func TestFRD_OwnsConstructedAndExportedSampledResponseData(t *testing.T) {
	resp := [][][]complex128{
		{{1 + 2i}},
		{{3 + 4i}},
	}
	omega := []float64{1, 2}
	frd, err := NewFRD(resp, omega, 0)
	if err != nil {
		t.Fatal(err)
	}

	resp[0][0][0] = 99
	omega[0] = 99
	if got := frd.At(0, 0, 0); got != 1+2i {
		t.Fatalf("FRD response aliases constructor input: got %v", got)
	}
	if got := frd.Omega[0]; got != 1 {
		t.Fatalf("FRD omega aliases constructor input: got %v", got)
	}

	matrix := frd.FreqResponse()
	matrix.Data[0] = 77
	matrix.Omega[0] = 77
	if got := frd.At(0, 0, 0); got != 1+2i {
		t.Fatalf("FreqResponse data aliases FRD storage: got %v", got)
	}
	if got := frd.Omega[0]; got != 1 {
		t.Fatalf("FreqResponse omega aliases FRD storage: got %v", got)
	}
}

func TestFRDConvenienceOperationsPreserveGridAndMetadata(t *testing.T) {
	frd, err := NewFRD([][][]complex128{
		{{3 + 4i, 1 - 1i}, {0 + 2i, -2}},
		{{1 + 0i, 0 + 3i}, {4 + 0i, 0 - 1i}},
		{{0 + 2i, 2 + 0i}, {1 + 1i, -30 + 40i}},
	}, []float64{0.5, 1, 2}, 0)
	if err != nil {
		t.Fatal(err)
	}
	frd.InputName = []string{"u1", "u2"}
	frd.OutputName = []string{"y1", "y2"}

	abs := frd.Abs()
	if abs.At(0, 0, 0) != 5 {
		t.Fatalf("Abs response[0][0][0] = %v, want 5", abs.At(0, 0, 0))
	}
	if abs.Omega[2] != 2 || abs.InputName[1] != "u2" || abs.OutputName[0] != "y1" {
		t.Fatalf("Abs did not preserve frequency grid or metadata")
	}

	selected, err := frd.SelectFrequencies([]int{0, 2})
	if err != nil {
		t.Fatal(err)
	}
	if selected.NumFrequencies() != 2 || selected.Omega[1] != 2 {
		t.Fatalf("selected grid = %v, want [0.5 2]", selected.Omega)
	}
	if selected.At(1, 1, 1) != -30+40i {
		t.Fatalf("selected response = %v, want -30+40i", selected.At(1, 1, 1))
	}

	ranged, err := frd.SelectFrequencyRange(0.75, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ranged.Omega, []float64{1, 2}) {
		t.Fatalf("range grid = %v, want [1 2]", ranged.Omega)
	}

	mapped, err := frd.MapResponse(func(freq int, omega float64, h [][]complex128) ([][]complex128, error) {
		out := make([][]complex128, len(h))
		for i := range h {
			out[i] = make([]complex128, len(h[i]))
			for j := range h[i] {
				out[i][j] = complex(omega, 0) * h[i][j] * complex(float64(freq+1), 0)
			}
		}
		return out, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if mapped.At(1, 0, 1) != 0+6i {
		t.Fatalf("mapped response = %v, want 0+6i", mapped.At(1, 0, 1))
	}

	peak, err := frd.PeakGain()
	if err != nil {
		t.Fatal(err)
	}
	if peak.Frequency != 2 {
		t.Fatalf("peak frequency = %g, want 2", peak.Frequency)
	}
	if peak.Gain <= 5 {
		t.Fatalf("peak gain = %g, want above 5 for MIMO sample", peak.Gain)
	}
}

func TestFRDConcatValidatesCompatibilityAndFrequencyOrder(t *testing.T) {
	left, err := NewFRD([][][]complex128{{{1}}, {{2}}}, []float64{1, 2}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	right, err := NewFRD([][][]complex128{{{3}}, {{4}}}, []float64{3, 4}, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	joined, err := FRDConcat(left, right)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(joined.Omega, []float64{1, 2, 3, 4}) {
		t.Fatalf("joined omega = %v, want [1 2 3 4]", joined.Omega)
	}
	if joined.At(3, 0, 0) != 4 {
		t.Fatalf("joined final response = %v, want 4", joined.At(3, 0, 0))
	}

	_, err = FRDConcat(left, left)
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("duplicate frequency error = %v, want ErrInvalidArgument", err)
	}

	differentDt, err := NewFRD([][][]complex128{{{5}}}, []float64{3}, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	_, err = FRDConcat(left, differentDt)
	if !errors.Is(err, ErrDomainMismatch) {
		t.Fatalf("domain mismatch error = %v, want ErrDomainMismatch", err)
	}
}

func TestFRD_Discrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := []float64{0.1, 1, 5}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}

	if !frd.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if frd.IsContinuous() {
		t.Fatal("expected not continuous")
	}

	for k, w := range omega {
		z := cmplx.Exp(complex(0, w*0.1))
		want := 1 / (z - 0.5)
		got := frd.At(k, 0, 0)
		if cmplx.Abs(got-want) > 1e-10 {
			t.Errorf("w=%v: H=%v, want %v", w, got, want)
		}
	}
}

func TestFRD_ValidationErrors(t *testing.T) {
	t.Run("unsorted omega", func(t *testing.T) {
		resp := [][][]complex128{{{1 + 0i}}, {{2 + 0i}}}
		_, err := NewFRD(resp, []float64{10, 1}, 0)
		if err == nil {
			t.Fatal("expected error for unsorted omega")
		}
	})

	t.Run("negative omega", func(t *testing.T) {
		resp := [][][]complex128{{{1 + 0i}}, {{2 + 0i}}}
		_, err := NewFRD(resp, []float64{-1, 1}, 0)
		if err == nil {
			t.Fatal("expected error for negative omega")
		}
	})

	t.Run("mismatched response dimensions", func(t *testing.T) {
		resp := [][][]complex128{
			{{1 + 0i}},
			{{2 + 0i, 3 + 0i}},
		}
		_, err := NewFRD(resp, []float64{1, 2}, 0)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Fatalf("expected ErrDimensionMismatch, got %v", err)
		}
	})

	t.Run("response/omega length mismatch", func(t *testing.T) {
		resp := [][][]complex128{{{1 + 0i}}}
		_, err := NewFRD(resp, []float64{1, 2}, 0)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Fatalf("expected ErrDimensionMismatch, got %v", err)
		}
	})

	t.Run("negative sample time", func(t *testing.T) {
		resp := [][][]complex128{{{1 + 0i}}}
		_, err := NewFRD(resp, []float64{1}, -0.1)
		if !errors.Is(err, ErrInvalidSampleTime) {
			t.Fatalf("expected ErrInvalidSampleTime, got %v", err)
		}
	})

	t.Run("exceeds Nyquist", func(t *testing.T) {
		resp := [][][]complex128{{{1 + 0i}}}
		dt := 0.1
		nyq := math.Pi / dt
		_, err := NewFRD(resp, []float64{nyq + 1}, dt)
		if !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("expected ErrInvalidArgument, got %v", err)
		}
	})
}

func TestFRD_CrossValidateWithFreqResponse(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := []float64{0.01, 0.1, 1, 10, 100}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}

	_, m, p := sys.Dims()
	for k := range omega {
		for i := range p {
			for j := range m {
				got := frd.At(k, i, j)
				want := resp.At(k, i, j)
				if cmplx.Abs(got-want) > 1e-10 {
					t.Errorf("w=%v [%d,%d]: FRD=%v, FreqResponse=%v", omega[k], i, j, got, want)
				}
			}
		}
	}
}

func TestFreqResponseMatrixCarriesFrequencyGrid(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{3}),
		mat.NewDense(1, 1, []float64{0.5}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	omega := []float64{0.2, 1.5, 4.0}

	resp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Omega) != len(omega) {
		t.Fatalf("len(FreqResponse.Omega) = %d, want %d", len(resp.Omega), len(omega))
	}
	for i := range omega {
		if resp.Omega[i] != omega[i] {
			t.Fatalf("FreqResponse.Omega[%d] = %v, want %v", i, resp.Omega[i], omega[i])
		}
	}
	frd, err := sys.FRD(resp.Omega)
	if err != nil {
		t.Fatal(err)
	}
	fromFRD := frd.FreqResponse()
	if len(fromFRD.Omega) != frd.NumFrequencies() {
		t.Fatalf("len(FRD.FreqResponse.Omega) = %d, want %d", len(fromFRD.Omega), frd.NumFrequencies())
	}
	for i := range fromFRD.Omega {
		if fromFRD.Omega[i] != frd.Omega[i] {
			t.Fatalf("FRD.FreqResponse.Omega[%d] = %v, want %v", i, fromFRD.Omega[i], frd.Omega[i])
		}
	}
}

func TestFRD_CrossValidateMIMO(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := []float64{0.01, 0.1, 1, 10, 100}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}

	_, m, p := sys.Dims()
	for k := range omega {
		for i := range p {
			for j := range m {
				got := frd.At(k, i, j)
				want := resp.At(k, i, j)
				if cmplx.Abs(got-want) > 1e-10 {
					t.Errorf("w=%v [%d,%d]: FRD=%v, FreqResponse=%v", omega[k], i, j, got, want)
				}
			}
		}
	}
}

func TestFRD_EmptyOmega(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	frd, err := sys.FRD(nil)
	if !errors.Is(err, ErrInvalidArgument) || frd != nil {
		t.Fatalf("FRD(nil) = %v, %v; want nil, ErrInvalidArgument", frd, err)
	}
}

func TestFRD_EvalFr(t *testing.T) {
	resp := [][][]complex128{
		{{1 + 2i, 3 + 4i}, {5 + 6i, 7 + 8i}},
	}
	frd, err := NewFRD(resp, []float64{1.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	grid := frd.EvalFr(0)
	if len(grid) != 2 || len(grid[0]) != 2 {
		t.Fatalf("EvalFr dims = %dx%d, want 2x2", len(grid), len(grid[0]))
	}
	if grid[0][0] != 1+2i || grid[0][1] != 3+4i || grid[1][0] != 5+6i || grid[1][1] != 7+8i {
		t.Errorf("EvalFr mismatch: %v", grid)
	}
}

func TestFRD_FreqResponseMatrix(t *testing.T) {
	resp := [][][]complex128{
		{{1 + 2i}},
		{{3 + 4i}},
	}
	frd, err := NewFRD(resp, []float64{1, 10}, 0)
	if err != nil {
		t.Fatal(err)
	}

	frm := frd.FreqResponse()
	if frm.NFreq != 2 || frm.P != 1 || frm.M != 1 {
		t.Fatalf("FreqResponseMatrix dims: NFreq=%d P=%d M=%d", frm.NFreq, frm.P, frm.M)
	}
	if frm.At(0, 0, 0) != 1+2i {
		t.Errorf("At(0,0,0) = %v, want 1+2i", frm.At(0, 0, 0))
	}
	if frm.At(1, 0, 0) != 3+4i {
		t.Errorf("At(1,0,0) = %v, want 3+4i", frm.At(1, 0, 0))
	}
}

func TestFRDSeries_MIMO(t *testing.T) {
	f1, err := NewFRD([][][]complex128{
		{{1 + 1i, 2 - 1i}, {3 + 0i, -1 + 2i}},
	}, []float64{1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	f2, err := NewFRD([][][]complex128{
		{{2 + 0i, -1 + 1i}, {0.5 - 0.5i, 4 + 0i}},
	}, []float64{1}, 0)
	if err != nil {
		t.Fatal(err)
	}

	got, err := FRDSeries(f1, f2)
	if err != nil {
		t.Fatal(err)
	}

	want := [2][2]complex128{
		{
			f2.At(0, 0, 0)*f1.At(0, 0, 0) + f2.At(0, 0, 1)*f1.At(0, 1, 0),
			f2.At(0, 0, 0)*f1.At(0, 0, 1) + f2.At(0, 0, 1)*f1.At(0, 1, 1),
		},
		{
			f2.At(0, 1, 0)*f1.At(0, 0, 0) + f2.At(0, 1, 1)*f1.At(0, 1, 0),
			f2.At(0, 1, 0)*f1.At(0, 0, 1) + f2.At(0, 1, 1)*f1.At(0, 1, 1),
		},
	}
	for i := range 2 {
		for j := range 2 {
			if cmplx.Abs(got.At(0, i, j)-want[i][j]) > 1e-12 {
				t.Fatalf("series[%d,%d] = %v, want %v", i, j, got.At(0, i, j), want[i][j])
			}
		}
	}
}

func TestFRDInterconnectionsPreserveSignalMetadata(t *testing.T) {
	f1, err := NewFRD([][][]complex128{
		{{1, 2}, {3, 4}},
	}, []float64{1}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	f1.InputName = []string{"u1", "u2"}
	f1.OutputName = []string{"v1", "v2"}
	f2, err := NewFRD([][][]complex128{
		{{5, 6}, {7, 8}},
	}, []float64{1}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	f2.InputName = []string{"v1", "v2"}
	f2.OutputName = []string{"y1", "y2"}

	series, err := FRDSeries(f1, f2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(series.InputName, []string{"u1", "u2"}) {
		t.Fatalf("series InputName = %v, want [u1 u2]", series.InputName)
	}
	if !reflect.DeepEqual(series.OutputName, []string{"y1", "y2"}) {
		t.Fatalf("series OutputName = %v, want [y1 y2]", series.OutputName)
	}
	if series.Dt != 0.1 {
		t.Fatalf("series Dt = %v, want 0.1", series.Dt)
	}

	parallel, err := FRDParallel(f1, f1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(parallel.InputName, []string{"u1", "u2"}) {
		t.Fatalf("parallel InputName = %v, want [u1 u2]", parallel.InputName)
	}
	if !reflect.DeepEqual(parallel.OutputName, []string{"v1", "v2"}) {
		t.Fatalf("parallel OutputName = %v, want [v1 v2]", parallel.OutputName)
	}

	feedback, err := FRDFeedback(f1, f2, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(feedback.InputName, []string{"u1", "u2"}) {
		t.Fatalf("feedback InputName = %v, want [u1 u2]", feedback.InputName)
	}
	if !reflect.DeepEqual(feedback.OutputName, []string{"v1", "v2"}) {
		t.Fatalf("feedback OutputName = %v, want [v1 v2]", feedback.OutputName)
	}
}

func TestFRDParallel_MIMO(t *testing.T) {
	f1, err := NewFRD([][][]complex128{
		{{1 + 1i, 2 - 1i}, {3 + 0i, -1 + 2i}},
	}, []float64{1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	f2, err := NewFRD([][][]complex128{
		{{2 + 0i, -1 + 1i}, {0.5 - 0.5i, 4 + 0i}},
	}, []float64{1}, 0)
	if err != nil {
		t.Fatal(err)
	}

	got, err := FRDParallel(f1, f2)
	if err != nil {
		t.Fatal(err)
	}

	for i := range 2 {
		for j := range 2 {
			want := f1.At(0, i, j) + f2.At(0, i, j)
			if cmplx.Abs(got.At(0, i, j)-want) > 1e-12 {
				t.Fatalf("parallel[%d,%d] = %v, want %v", i, j, got.At(0, i, j), want)
			}
		}
	}
}

func TestFRDFeedback_MIMO(t *testing.T) {
	plant, err := NewFRD([][][]complex128{
		{{2 + 0.5i, 0}, {0, 1 - 0.25i}},
		{{1 - 0.5i, 0}, {0, 0.5 + 0.75i}},
	}, []float64{1, 2}, 0)
	if err != nil {
		t.Fatal(err)
	}
	controller, err := NewFRD([][][]complex128{
		{{3 - 0.5i, 0}, {0, -0.5 + 0.25i}},
		{{2 + 0.5i, 0}, {0, 1 + 0.5i}},
	}, []float64{1, 2}, 0)
	if err != nil {
		t.Fatal(err)
	}

	got, err := FRDFeedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}

	for k := range 2 {
		for i := range 2 {
			g := plant.At(k, i, i)
			c := controller.At(k, i, i)
			want := g / (1 + c*g)
			if cmplx.Abs(got.At(k, i, i)-want) > 1e-12 {
				t.Fatalf("feedback[%d,%d,%d] = %v, want %v", k, i, i, got.At(k, i, i), want)
			}
		}
		if cmplx.Abs(got.At(k, 0, 1)) > 1e-12 || cmplx.Abs(got.At(k, 1, 0)) > 1e-12 {
			t.Fatalf("feedback off-diagonals = [%v %v], want 0", got.At(k, 0, 1), got.At(k, 1, 0))
		}
	}
}

func TestFRDFeedback_CoupledRectangularMIMO(t *testing.T) {
	plant, err := NewFRD([][][]complex128{{
		{1 + 0.2i, -0.5i, 0.25},
		{0.3 - 0.1i, 2, -0.4 + 0.6i},
	}}, []float64{1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	controller, err := NewFRD([][][]complex128{{
		{0.5, -0.1 + 0.2i},
		{0.3i, 0.4},
		{-0.2, 0.1 - 0.1i},
	}}, []float64{1}, 0)
	if err != nil {
		t.Fatal(err)
	}

	got, err := FRDFeedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 2 {
		for j := range 3 {
			var product complex128
			for l := range 3 {
				m := complex(0, 0)
				if l == j {
					m = 1
				}
				for k := range 2 {
					m += controller.At(0, l, k) * plant.At(0, k, j)
				}
				product += got.At(0, i, l) * m
			}
			if diff := cmplx.Abs(product - plant.At(0, i, j)); diff > 1e-12 {
				t.Fatalf("closed-loop residual (%d,%d) = %g", i, j, diff)
			}
		}
	}
}

func TestFRDFeedback_Singular(t *testing.T) {
	plant, err := NewFRD([][][]complex128{{{1}}}, []float64{1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	controller, err := NewFRD([][][]complex128{{{-1}}}, []float64{1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := FRDFeedback(plant, controller, -1); err == nil {
		t.Fatal("expected singular FRD feedback error")
	}
}

func TestFRDInterconnectionRejectsSampleTimeMismatch(t *testing.T) {
	f1, err := NewFRD([][][]complex128{{{1}}}, []float64{1}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	f2, err := NewFRD([][][]complex128{{{1}}}, []float64{1}, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := FRDParallel(f1, f2); !errors.Is(err, ErrDomainMismatch) {
		t.Fatalf("FRDParallel sample-time mismatch error = %v, want ErrDomainMismatch", err)
	}
}

func TestFRD_Bode(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := []float64{0.1, 1, 10}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}

	bode := frd.Bode()
	modelBode, err := sys.Bode(omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	wantMagDB := []float64{
		20 * math.Log10(1.0/math.Sqrt(1.01)),
		20 * math.Log10(1.0/math.Sqrt(2)),
		20 * math.Log10(1.0/math.Sqrt(101)),
	}
	for k := range omega {
		got := bode.MagDBAt(k, 0, 0)
		if math.Abs(got-wantMagDB[k]) > 1e-8 {
			t.Errorf("w=%v: magDB=%v, want %v", omega[k], got, wantMagDB[k])
		}
		if math.Abs(got-modelBode.MagDBAt(k, 0, 0)) > 1e-12 {
			t.Errorf("w=%v: FRD magDB=%v, model magDB=%v", omega[k], got, modelBode.MagDBAt(k, 0, 0))
		}
		if math.Abs(bode.PhaseAt(k, 0, 0)-modelBode.PhaseAt(k, 0, 0)) > 1e-12 {
			t.Errorf("w=%v: FRD phase=%v, model phase=%v", omega[k], bode.PhaseAt(k, 0, 0), modelBode.PhaseAt(k, 0, 0))
		}
	}
}

func TestFRD_DataIsolation(t *testing.T) {
	resp := [][][]complex128{{{1 + 0i}}}
	omega := []float64{1.0}
	frd, err := NewFRD(resp, omega, 0)
	if err != nil {
		t.Fatal(err)
	}

	resp[0][0][0] = 999 + 0i
	omega[0] = 999
	if frd.At(0, 0, 0) != 1+0i {
		t.Error("FRD response mutated by external modification")
	}
	if frd.Omega[0] != 1.0 {
		t.Error("FRD omega mutated by external modification")
	}
}

func TestFRD_NyquistBoundary(t *testing.T) {
	dt := 0.1
	nyq := math.Pi / dt
	resp := [][][]complex128{{{1 + 0i}}}
	_, err := NewFRD(resp, []float64{nyq}, dt)
	if err != nil {
		t.Fatalf("Nyquist frequency should be allowed, got %v", err)
	}
}

func TestFRD_NyquistFromFRD(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	w := logspace(-1, 1, 50)
	f, err := sys.FRD(w)
	if err != nil {
		t.Fatal(err)
	}
	res, err := f.Nyquist()
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Contour) != 50 {
		t.Fatalf("contour len = %d, want 50", len(res.Contour))
	}
	if cmplx.Abs(res.Contour[0]-f.Response[0][0][0]) > 1e-12 {
		t.Errorf("contour[0] mismatch")
	}
}

func frdTestPolyMul(a, b []float64) []float64 {
	out := make([]float64, len(a)+len(b)-1)
	for i, x := range a {
		for j, y := range b {
			out[i+j] += x * y
		}
	}
	return out
}

// frdTestCompanion realizes the strictly proper num/den (descending powers,
// monic den) in controllable canonical form, which has a non-symmetric A.
func frdTestCompanion(t *testing.T, num, den []float64) *System {
	t.Helper()
	n := len(den) - 1
	a := mat.NewDense(n, n, nil)
	for i := range n - 1 {
		a.Set(i, i+1, 1)
	}
	for j := range n {
		a.Set(n-1, j, -den[n-j]/den[0])
	}
	b := mat.NewDense(n, 1, nil)
	b.Set(n-1, 0, 1)
	c := mat.NewDense(1, n, nil)
	for k, v := range num {
		c.Set(0, len(num)-1-k, v/den[0])
	}
	sys, err := New(a, b, c, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func frdTestUnstableCount(t *testing.T, a *mat.Dense, continuous bool) int {
	t.Helper()
	var eig mat.Eigen
	if !eig.Factorize(a, mat.EigenNone) {
		t.Fatal("eigen failed")
	}
	count := 0
	for _, v := range eig.Values(nil) {
		if (continuous && real(v) > 1e-9) || (!continuous && cmplx.Abs(v) > 1+1e-9) {
			count++
		}
	}
	return count
}

// frdTestNyquistOracle returns Z - P: unstable closed-loop poles under unit
// negative feedback minus unstable open-loop poles.
func frdTestNyquistOracle(t *testing.T, sys *System) int {
	t.Helper()
	d := sys.D.At(0, 0)
	var bc mat.Dense
	bc.Mul(sys.B, sys.C)
	bc.Scale(1/(1+d), &bc)
	var acl mat.Dense
	acl.Sub(sys.A, &bc)
	ct := sys.IsContinuous()
	return frdTestUnstableCount(t, &acl, ct) - frdTestUnstableCount(t, sys.A, ct)
}

func TestFRD_NyquistEncirclements(t *testing.T) {
	s1, s2, s3 := []float64{1, 1}, []float64{1, 2}, []float64{1, 3}
	s := []float64{1, 0}
	s10 := []float64{1, 10}
	cases := []struct {
		name     string
		num, den []float64
		want     int
	}{
		{"repro K=180", []float64{180}, frdTestPolyMul(frdTestPolyMul(s1, s2), s3), 2},
		{"K=30", []float64{30}, frdTestPolyMul(frdTestPolyMul(s1, s2), s3), 0},
		{"type1 K=5", []float64{5}, frdTestPolyMul(s, s1), 0},
		{"type1 K=-2", []float64{-2}, frdTestPolyMul(s, s1), 1},
		{"type1 K=10 unstable", []float64{10}, frdTestPolyMul(frdTestPolyMul(s, s1), s2), 2},
		{"type1 K=3", []float64{3}, frdTestPolyMul(frdTestPolyMul(s, s1), s2), 0},
		{"type2 K=1", []float64{1}, frdTestPolyMul(frdTestPolyMul(s, s), s1), 2},
		{"type2 lead", []float64{20, 20}, frdTestPolyMul(frdTestPolyMul(s, s), s10), 0},
		{"type3 K=20", frdTestPolyMul([]float64{20, 20}, s1), frdTestPolyMul(frdTestPolyMul(frdTestPolyMul(s, s), s), s10), 0},
		{"unstable open loop K=2", []float64{2}, []float64{1, -1}, -1},
		{"unstable open loop K=0.5", []float64{0.5}, []float64{1, -1}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ct := frdTestCompanion(t, tc.num, tc.den)
			if got := frdTestNyquistOracle(t, ct); got != tc.want {
				t.Fatalf("oracle Z-P = %d, want %d", got, tc.want)
			}
			dt := 0.05
			dsys, err := ct.C2D(dt, C2DOptions{})
			if err != nil {
				t.Fatal(err)
			}
			for _, sys := range []*System{ct, dsys} {
				wHi := 3.0
				if sys.IsDiscrete() {
					wHi = math.Log10(math.Pi / dt * (1 - 1e-12))
				}
				f, err := sys.FRD(logspace(-4, wHi, 1500))
				if err != nil {
					t.Fatal(err)
				}
				res, err := f.Nyquist()
				if err != nil {
					t.Fatal(err)
				}
				want := frdTestNyquistOracle(t, sys)
				if res.Encirclements != want {
					t.Errorf("dt=%v: Encirclements = %d, want %d", sys.Dt, res.Encirclements, want)
				}
			}
		})
	}
}

func TestFRD_NyquistMatchesSystemNyquist(t *testing.T) {
	sys := frdTestCompanion(t, []float64{180}, frdTestPolyMul(frdTestPolyMul([]float64{1, 1}, []float64{1, 2}), []float64{1, 3}))
	w := logspace(-3, 3, 600)
	f, err := sys.FRD(w)
	if err != nil {
		t.Fatal(err)
	}
	got, err := f.Nyquist()
	if err != nil {
		t.Fatal(err)
	}
	want, err := sys.Nyquist(w, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.Encirclements != want.Encirclements || got.Encirclements != 2 {
		t.Fatalf("FRD Encirclements = %d, System = %d, want 2", got.Encirclements, want.Encirclements)
	}
	nw := len(w)
	for k := range nw {
		if got.ContourN[k] != cmplx.Conj(got.Contour[nw-1-k]) {
			t.Fatalf("ContourN[%d] = %v, want conj(Contour[%d]) = %v", k, got.ContourN[k], nw-1-k, cmplx.Conj(got.Contour[nw-1-k]))
		}
	}
}

func TestFRD_NyquistNonFinite(t *testing.T) {
	f, err := NewFRD([][][]complex128{{{cmplx.Inf()}}, {{1}}}, []float64{0, 1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Nyquist(); err == nil {
		t.Fatal("expected error for non-finite response")
	}
}

func TestFRD_Sigma(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	w := logspace(-1, 1, 20)
	f, err := sys.FRD(w)
	if err != nil {
		t.Fatal(err)
	}
	sig, err := f.Sigma()
	if err != nil {
		t.Fatal(err)
	}
	for k, wk := range w {
		want := 1 / math.Sqrt(1+wk*wk)
		got := sig.sv[k]
		if math.Abs(got-want)/want > 1e-4 {
			t.Errorf("w=%g: sigma=%g, want %g", wk, got, want)
		}
	}
}

func TestFRDMargin(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0, 1, 0, 0, 0, 1, -6, -11, -6})
	B := mat.NewDense(3, 1, []float64{0, 0, 60})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)
	w := logspace(-2, 2, 2000)
	f, err := sys.FRD(w)
	if err != nil {
		t.Fatal(err)
	}
	mr, err := FRDMargin(f)
	if err != nil {
		t.Fatal(err)
	}
	sysMr, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsInf(sysMr.PhaseMargin, 0) {
		t.Skip("system has no gain crossover")
	}
	if math.Abs(mr.PhaseMargin-sysMr.PhaseMargin) > 3 {
		t.Errorf("PM = %g, sys PM = %g", mr.PhaseMargin, sysMr.PhaseMargin)
	}
}

func TestFRD_Sigma_MIMO_Regression(t *testing.T) {
	// Bug: real block form doubles SVs; diag(2,1) was returning [2,2] not [2,1]
	resp := [][][]complex128{
		{{2, 0}, {0, 1}},
	}
	f, _ := NewFRD(resp, []float64{1.0}, 0)
	sig, err := f.Sigma()
	if err != nil {
		t.Fatal(err)
	}
	if sig.nSV != 2 {
		t.Fatalf("nSV = %d, want 2", sig.nSV)
	}
	if math.Abs(sig.sv[0]-2) > 1e-10 {
		t.Errorf("sv[0] = %g, want 2", sig.sv[0])
	}
	if math.Abs(sig.sv[1]-1) > 1e-10 {
		t.Errorf("sv[1] = %g, want 1", sig.sv[1])
	}
}

func TestFRDMargin_NegativeMargins(t *testing.T) {
	// K/(s+1)^3 with K=10: unstable closed-loop, negative margins
	// Margin() returns GM≈-1.94dB, PM≈-7°
	A := mat.NewDense(3, 3, []float64{-1, 1, 0, 0, -1, 1, 0, 0, -1})
	B := mat.NewDense(3, 1, []float64{0, 0, 10})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	sysMr, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if sysMr.GainMargin >= 0 {
		t.Skipf("expected negative GM, got %g", sysMr.GainMargin)
	}

	w := logspace(-2, 2, 5000)
	f, err := sys.FRD(w)
	if err != nil {
		t.Fatal(err)
	}
	mr, err := FRDMargin(f)
	if err != nil {
		t.Fatal(err)
	}

	if math.IsInf(mr.GainMargin, 1) || math.IsNaN(mr.GainMargin) {
		t.Errorf("FRDMargin GM = %g, want finite negative (sys GM = %g)", mr.GainMargin, sysMr.GainMargin)
	}
	if math.Abs(mr.GainMargin-sysMr.GainMargin) > 1 {
		t.Errorf("FRDMargin GM = %g, sys GM = %g", mr.GainMargin, sysMr.GainMargin)
	}
	if math.IsInf(mr.PhaseMargin, 1) || math.IsNaN(mr.PhaseMargin) {
		t.Errorf("FRDMargin PM = %g, want finite negative (sys PM = %g)", mr.PhaseMargin, sysMr.PhaseMargin)
	}
	if math.Abs(mr.PhaseMargin-sysMr.PhaseMargin) > 3 {
		t.Errorf("FRDMargin PM = %g, sys PM = %g", mr.PhaseMargin, sysMr.PhaseMargin)
	}
	if math.IsNaN(mr.WgFreq) {
		t.Errorf("FRDMargin WgFreq = NaN, want finite (sys = %g)", sysMr.WgFreq)
	}
	if math.IsNaN(mr.WpFreq) {
		t.Errorf("FRDMargin WpFreq = NaN, want finite (sys = %g)", sysMr.WpFreq)
	}
}

func TestLsim_NonUniformGrid_Rejected(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	tNonUniform := []float64{0, 0.1, 0.3, 0.6}
	u := mat.NewDense(4, 1, []float64{1, 1, 1, 1})
	_, err := Lsim(sys, u, tNonUniform, nil)
	if err == nil {
		t.Error("Lsim should reject non-uniform time grid")
	}
}

func TestLsim_DiscreteDtMismatch_Rejected(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)
	tGrid := []float64{0, 0.05, 0.10, 0.15}
	u := mat.NewDense(4, 1, []float64{1, 1, 1, 1})
	_, err := Lsim(sys, u, tGrid, nil)
	if err == nil {
		t.Error("Lsim should reject discrete system when t spacing != sys.Dt")
	}
}

func TestInv_DelayRejected(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), 0)
	sys.SetInputDelay([]float64{0.5})
	_, err := Inv(sys)
	if err == nil {
		t.Error("Inv should reject delayed system")
	}
}

type frdMarginCase struct {
	name  string
	dt    float64
	omega []float64
	resp  func(w float64) complex128
	phase func(w float64) float64 // continuous phase in degrees
}

func bisectFRDOracle(lo, hi float64, f func(float64) float64) float64 {
	flo := f(lo)
	for range 200 {
		mid := 0.5 * (lo + hi)
		fm := f(mid)
		if flo*fm <= 0 {
			hi = mid
		} else {
			lo, flo = mid, fm
		}
	}
	return 0.5 * (lo + hi)
}

func frdMarginOracle(c frdMarginCase) (gms, wps, pms, wgs []float64) {
	w0, w1 := c.omega[0], c.omega[len(c.omega)-1]
	n := 100000
	ws := make([]float64, n)
	for i := range ws {
		ws[i] = w0 * math.Pow(w1/w0, float64(i)/float64(n-1))
	}
	ph0, ph1 := c.phase(w0), c.phase(w1)
	kLo := math.Ceil((math.Min(ph0, ph1) + 180) / 360)
	kHi := math.Floor((math.Max(ph0, ph1) + 180) / 360)
	for k := kLo; k <= kHi; k++ {
		target := -180 + 360*k
		g := func(w float64) float64 { return c.phase(w) - target }
		for i := 1; i < n; i++ {
			if g(ws[i-1])*g(ws[i]) < 0 {
				w := bisectFRDOracle(ws[i-1], ws[i], g)
				wps = append(wps, w)
				gms = append(gms, -20*math.Log10(cmplx.Abs(c.resp(w))))
			}
		}
	}
	m := func(w float64) float64 { return math.Log(cmplx.Abs(c.resp(w))) }
	for i := 1; i < n; i++ {
		if m(ws[i-1])*m(ws[i]) < 0 {
			w := bisectFRDOracle(ws[i-1], ws[i], m)
			pm := math.Mod(180+c.phase(w), 360)
			if pm <= -180 {
				pm += 360
			} else if pm > 180 {
				pm -= 360
			}
			wgs = append(wgs, w)
			pms = append(pms, pm)
		}
	}
	return
}

// pickMargin is MATLAB margin's rule: the margin closest to 0.
func pickMargin(vals, freqs []float64) (float64, float64) {
	best, bw := math.Inf(1), math.NaN()
	for i, v := range vals {
		if math.Abs(v) < math.Abs(best) {
			best, bw = v, freqs[i]
		}
	}
	return best, bw
}

func TestFRDMargin_WrappedCrossings(t *testing.T) {
	cases := []frdMarginCase{
		{
			name:  "delay30",
			omega: logspace(-2, math.Log10(5), 20000),
			resp: func(w float64) complex128 {
				return 2 * cmplx.Exp(complex(0, -30*w)) / complex(1, w)
			},
			phase: func(w float64) float64 { return (-math.Atan(w) - 30*w) * 180 / math.Pi },
		},
		{
			name:  "lag7_unstable_minus540",
			omega: logspace(-2, 2, 20000),
			resp: func(w float64) complex128 {
				return 10 / cmplx.Pow(complex(1, w), 7)
			},
			phase: func(w float64) float64 { return -7 * math.Atan(w) * 180 / math.Pi },
		},
		{
			name:  "discrete_delay12",
			dt:    0.1,
			omega: logspace(-2, math.Log10(math.Pi/0.1*0.999), 20000),
			resp: func(w float64) complex128 {
				z := cmplx.Exp(complex(0, w*0.1))
				return 1.5 * cmplx.Pow(z, -12) * 0.4 / (z - 0.6)
			},
			phase: func(w float64) float64 {
				th := w * 0.1
				return (-13*th - math.Atan2(0.6*math.Sin(th), 1-0.6*math.Cos(th))) * 180 / math.Pi
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := make([][][]complex128, len(c.omega))
			for k, w := range c.omega {
				resp[k] = [][]complex128{{c.resp(w)}}
			}
			f, err := NewFRD(resp, c.omega, c.dt)
			if err != nil {
				t.Fatal(err)
			}
			mr, err := FRDMargin(f)
			if err != nil {
				t.Fatal(err)
			}
			gms, wps, pms, wgs := frdMarginOracle(c)
			wantGM, wantWp := pickMargin(gms, wps)
			wantPM, wantWg := pickMargin(pms, wgs)

			if math.Abs(mr.GainMargin-wantGM) > 1e-3 || math.Abs(mr.WpFreq-wantWp) > 1e-4*wantWp {
				t.Errorf("GM = %g @ %g, want %g @ %g", mr.GainMargin, mr.WpFreq, wantGM, wantWp)
			}
			if math.Abs(mr.PhaseMargin-wantPM) > 1e-3 || math.Abs(mr.WgFreq-wantWg) > 1e-4*wantWg {
				t.Errorf("PM = %g @ %g, want %g @ %g", mr.PhaseMargin, mr.WgFreq, wantPM, wantWg)
			}
			if mr.PhaseMargin <= -180 || mr.PhaseMargin > 180 {
				t.Errorf("PM = %g outside (-180,180]", mr.PhaseMargin)
			}
			off := math.Mod(c.phase(mr.WpFreq)+180, 360)
			if math.Min(math.Abs(off), 360-math.Abs(off)) > 1e-2 {
				t.Errorf("angle at WpFreq = %g, not -180 mod 360", c.phase(mr.WpFreq))
			}
			if gm := -20 * math.Log10(cmplx.Abs(c.resp(mr.WpFreq))); math.Abs(gm-mr.GainMargin) > 1e-3 {
				t.Errorf("GM = %g, -20log|L(WpFreq)| = %g", mr.GainMargin, gm)
			}
		})
	}
}

func TestFRDMargin_DelayRepro(t *testing.T) {
	sys, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{2}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{30}
	f, err := sys.FRD(logspace(-2, math.Log10(5), 20000))
	if err != nil {
		t.Fatal(err)
	}
	mr, err := FRDMargin(f)
	if err != nil {
		t.Fatal(err)
	}
	wg := math.Sqrt(3)
	wantPM := math.Mod(180+(-math.Atan(wg)-30*wg)*180/math.Pi, 360)
	if wantPM <= -180 {
		wantPM += 360
	}
	if math.Abs(mr.PhaseMargin-wantPM) > 1e-3 || math.Abs(mr.WgFreq-wg) > 1e-4 {
		t.Errorf("PM = %g @ %g, want %g @ %g", mr.PhaseMargin, mr.WgFreq, wantPM, wg)
	}
}

func TestNewFRDRejectsNonFiniteData(t *testing.T) {
	resp := [][][]complex128{{{1}}, {{2}}}
	for _, w := range []float64{math.NaN(), math.Inf(1)} {
		if _, err := NewFRD(resp, []float64{1, w}, 0); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("NewFRD omega=%v err = %v", w, err)
		}
	}
	for _, dt := range []float64{math.NaN(), math.Inf(1)} {
		if _, err := NewFRD(resp, []float64{1, 2}, dt); !errors.Is(err, ErrInvalidSampleTime) {
			t.Errorf("NewFRD dt=%v err = %v", dt, err)
		}
	}
}

func TestFRDRejectsPlaceholderData(t *testing.T) {
	if _, err := NewFRD(nil, nil, 0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("empty: err = %v, want ErrInvalidArgument", err)
	}
	if _, err := NewFRD([][][]complex128{{{1}}, {{2}}}, []float64{1, 1}, 0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("duplicate omega: err = %v, want ErrInvalidArgument", err)
	}
	nan := complex(math.NaN(), 0)
	if _, err := NewFRD([][][]complex128{{{nan}}, {{nan}}}, []float64{1, 2}, 0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("NaN response: err = %v, want ErrInvalidArgument", err)
	}
	f, err := NewFRD([][][]complex128{{{1, 2}}, {{3, 4}}, {{5, 6}}}, []float64{1, 2, 3}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.SelectFrequencyRange(5, 6); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("empty range: err = %v, want ErrInvalidArgument", err)
	}
	if _, err := f.SelectFrequencies(nil); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("no indices: err = %v, want ErrInvalidArgument", err)
	}
	if _, err := f.SelectFrequencies([]int{3}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("index out of range: err = %v, want ErrInvalidArgument", err)
	}
	if _, err := f.MapResponse(nil); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil mapper: err = %v, want ErrInvalidArgument", err)
	}

	hand := &FRD{Response: [][][]complex128{{}}, Omega: []float64{1}}
	if p, m := hand.Dims(); p != 0 || m != 0 {
		t.Errorf("hand-built Dims = %d, %d", p, m)
	}
	if err := hand.Validate(); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("hand-built Validate: err = %v", err)
	}
	if _, err := hand.Sigma(); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("hand-built Sigma: err = %v", err)
	}
	if _, err := hand.PeakGain(); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("hand-built PeakGain: err = %v", err)
	}
	var nilFRD *FRD
	if err := nilFRD.Validate(); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil Validate: err = %v", err)
	}
	if err := f.Validate(); err != nil {
		t.Errorf("valid FRD: %v", err)
	}
}

func TestFRDErrorSentinels(t *testing.T) {
	mimo, err := NewFRD([][][]complex128{{{1, 2}}, {{3, 4}}}, []float64{1, 2}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := mimo.Nyquist(); !errors.Is(err, ErrNotSISO) {
		t.Errorf("Nyquist MIMO: err = %v, want ErrNotSISO", err)
	}
	if _, err := FRDMargin(mimo); !errors.Is(err, ErrNotSISO) {
		t.Errorf("FRDMargin MIMO: err = %v, want ErrNotSISO", err)
	}
	one, err := NewFRD([][][]complex128{{{1}}}, []float64{1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := FRDMargin(one); !errors.Is(err, ErrInsufficientData) {
		t.Errorf("FRDMargin one point: err = %v, want ErrInsufficientData", err)
	}
	g, err := NewFRD([][][]complex128{{{1}}, {{0.5i}}}, []float64{1, 2}, 0)
	if err != nil {
		t.Fatal(err)
	}
	k, err := NewFRD([][][]complex128{{{1}}, {{1}}}, []float64{1, 2}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := FRDFeedback(g, k, 1); !errors.Is(err, ErrAlgebraicLoop) {
		t.Errorf("FRDFeedback singular: err = %v, want ErrAlgebraicLoop", err)
	}
	if _, err := FRDFeedback(g, k, 0.5); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("FRDFeedback sign 0.5: err = %v, want ErrInvalidArgument", err)
	}
	if _, err := FRDSeries(g, mimo); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("FRDSeries dims: err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := FRDParallel(g, mimo); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("FRDParallel dims: err = %v, want ErrDimensionMismatch", err)
	}
	other, err := NewFRD([][][]complex128{{{1}}, {{1}}}, []float64{1, 3}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := FRDParallel(g, other); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("FRDParallel grid mismatch: err = %v, want ErrInvalidArgument", err)
	}
}
