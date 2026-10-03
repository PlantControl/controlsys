package controlsys

import (
	"encoding/json"
	"math"
	"math/cmplx"
	"os"
	"testing"
)

type matlabFixturePolynomial []float64

func (p *matlabFixturePolynomial) UnmarshalJSON(data []byte) error {
	var values []float64
	if len(data) > 0 && data[0] == '[' {
		if err := json.Unmarshal(data, &values); err != nil {
			return err
		}
	} else {
		var value float64
		if err := json.Unmarshal(data, &value); err != nil {
			return err
		}
		values = []float64{value}
	}
	*p = values
	return nil
}

type matlabLSFixture struct {
	MatlabVersion        string
	ControlVersion       string
	CoefficientTolerance float64
	ResponseTolerance    float64
	Cases                []struct {
		Name              string
		SourceNumerator   matlabFixturePolynomial
		SourceDenominator matlabFixturePolynomial
		SampleTime        float64
		FitOrder          int
		Numerator         matlabFixturePolynomial
		Denominator       matlabFixturePolynomial
		Frequencies       []float64
		ResponseReal      []float64
		ResponseImag      []float64
	}
}

func TestMATLABLeastSquaresReference(t *testing.T) {
	path := os.Getenv("CONTROLSYS_MATLAB_LS_FIXTURES")
	if path == "" {
		t.Skip("MATLAB reference unavailable; generate testdata/conversion/generate_matlab_least_squares.m and set CONTROLSYS_MATLAB_LS_FIXTURES")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var reference matlabLSFixture
	if err := json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}
	if len(reference.Cases) == 0 || reference.MatlabVersion == "" || reference.ControlVersion == "" {
		t.Fatal("fixture lacks MATLAB provenance or cases")
	}
	if !(reference.CoefficientTolerance > 0 && reference.CoefficientTolerance <= 1e-6 && reference.ResponseTolerance > 0 && reference.ResponseTolerance <= 1e-6) {
		t.Fatal("invalid parity tolerances")
	}
	t.Logf("MATLAB %s; Control System Toolbox %s", reference.MatlabVersion, reference.ControlVersion)
	for _, fixture := range reference.Cases {
		t.Run(fixture.Name, func(t *testing.T) {
			if len(fixture.Frequencies) == 0 || len(fixture.ResponseReal) != len(fixture.Frequencies) || len(fixture.ResponseImag) != len(fixture.Frequencies) {
				t.Fatal("invalid response fixture")
			}
			source := conversionSISO(t, fixture.SourceNumerator, fixture.SourceDenominator, 0)
			result, err := source.DiscretizeLeastSquares(fixture.SampleTime, fixture.FitOrder)
			if err != nil {
				t.Fatal(err)
			}
			transfer, err := result.Sys.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			compareConversionCoefficients(t, transfer.TF.Num[0][0], fixture.Numerator, reference.CoefficientTolerance)
			compareConversionCoefficients(t, transfer.TF.Den[0], fixture.Denominator, reference.CoefficientTolerance)
			for i, w := range fixture.Frequencies {
				got, err := result.Sys.EvalFr(cmplx.Exp(complex(0, w*fixture.SampleTime)))
				if err != nil {
					t.Fatal(err)
				}
				want := complex(fixture.ResponseReal[i], fixture.ResponseImag[i])
				if cmplx.Abs(got[0][0]-want) > reference.ResponseTolerance*math.Max(1, cmplx.Abs(want)) {
					t.Fatalf("w=%g got %v want %v", w, got[0][0], want)
				}
			}
		})
	}
}
