package controlsys

import (
	"fmt"
	"math/cmplx"
)

type sampledComplexResponse struct {
	data   []complex128
	layout sampledResponseLayout
}

type sampledScalarResponse struct {
	data   []float64
	layout sampledResponseLayout
}

type sampledResponseLayout struct {
	omega []float64
	p, m  int
}

type sampledComplexGridResponse struct {
	response [][][]complex128
	omega    []float64
	p, m     int
}

func newSampledComplexResponse(data []complex128, omega []float64, p, m int) sampledComplexResponse {
	return sampledComplexResponse{data: data, layout: newSampledResponseLayout(omega, p, m)}
}

func newSampledScalarResponse(data []float64, omega []float64, p, m int) sampledScalarResponse {
	return sampledScalarResponse{data: data, layout: newSampledResponseLayout(omega, p, m)}
}

func newSampledResponseLayout(omega []float64, p, m int) sampledResponseLayout {
	return sampledResponseLayout{omega: omega, p: p, m: m}
}

func (l sampledResponseLayout) offset(freq, output, input int) int {
	return freq*l.p*l.m + output*l.m + input
}

func (l sampledResponseLayout) blockOffset(freq int) int {
	return freq * l.p * l.m
}

func (r sampledComplexResponse) offset(freq, output, input int) int {
	return r.layout.offset(freq, output, input)
}

func (r sampledComplexResponse) at(freq, output, input int) complex128 {
	return r.data[r.offset(freq, output, input)]
}

func (r sampledComplexResponse) blockOffset(freq int) int {
	return r.layout.blockOffset(freq)
}

func (r sampledComplexResponse) singularValues(dst []float64, ws *complexSVDWorkspace, freq int) error {
	if r.layout.p == 1 && r.layout.m == 1 {
		dst[0] = cmplx.Abs(r.data[freq])
		return nil
	}
	if err := ws.singularValuesFromFlat(dst, r.data, r.blockOffset(freq), r.layout.p, r.layout.m); err != nil {
		return fmt.Errorf("omega=%g: %w", r.layout.omega[freq], err)
	}
	return nil
}

func (r sampledScalarResponse) at(freq, output, input int) float64 {
	return r.data[r.layout.offset(freq, output, input)]
}

func (r sampledScalarResponse) set(freq, output, input int, value float64) {
	r.data[r.layout.offset(freq, output, input)] = value
}

func newSampledComplexGridResponse(response [][][]complex128, omega []float64, p, m int) sampledComplexGridResponse {
	return sampledComplexGridResponse{response: response, omega: omega, p: p, m: m}
}

func (r sampledComplexGridResponse) singularValues(dst []float64, ws *complexSVDWorkspace, freq int) error {
	if r.p == 1 && r.m == 1 {
		dst[0] = cmplx.Abs(r.response[freq][0][0])
		return nil
	}
	if err := ws.singularValuesFromNested(dst, r.response[freq], r.p, r.m); err != nil {
		return fmt.Errorf("omega=%g: %w", r.omega[freq], err)
	}
	return nil
}
