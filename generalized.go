package controlsys

import (
	"fmt"
	"maps"
)

type NumericBlock interface {
	CurrentSystem() (*System, error)
}

type TunableBlock interface {
	NumericBlock
	FreeParameters() []*TunableReal
	SampleBlock(map[string]float64) (TunableBlock, error)
}

type fixedSystemBlock struct {
	sys *System
}

func (b fixedSystemBlock) CurrentSystem() (*System, error) {
	if b.sys == nil {
		return nil, fmt.Errorf("fixed system block is nil: %w", ErrDimensionMismatch)
	}
	return b.sys.Copy(), nil
}

type GeneralizedModel struct {
	name           string
	block          NumericBlock
	inputName      []string
	outputName     []string
	analysisPoints map[string]AnalysisPoint
}

// AnalysisPointLocation identifies the signal where a feedback loop is broken.
type AnalysisPointLocation uint8

const (
	AnalysisPointUnspecified AnalysisPointLocation = iota
	AnalysisPointPlantOutput
	AnalysisPointPlantInput
)

// AnalysisPoint binds a name to a loop location.
type AnalysisPoint struct {
	Name     string
	Location AnalysisPointLocation
}

func NewGeneralizedModel(name string, block any) (*GeneralizedModel, error) {
	if name == "" {
		return nil, fmt.Errorf("NewGeneralizedModel: name is empty: %w", ErrDimensionMismatch)
	}
	numeric, err := numericBlockFromAny(block)
	if err != nil {
		return nil, err
	}
	return &GeneralizedModel{name: name, block: numeric, analysisPoints: make(map[string]AnalysisPoint)}, nil
}

func numericBlockFromAny(block any) (NumericBlock, error) {
	switch b := block.(type) {
	case NumericBlock:
		return b, nil
	case *System:
		return fixedSystemBlock{sys: b}, nil
	default:
		return nil, fmt.Errorf("unsupported generalized block %T: %w", block, ErrDimensionMismatch)
	}
}

func (g *GeneralizedModel) SetInputName(names ...string) {
	g.inputName = copyStringSlice(names)
}

func (g *GeneralizedModel) SetOutputName(names ...string) {
	g.outputName = copyStringSlice(names)
}

func (g *GeneralizedModel) InsertAnalysisPoint(name string) {
	if g.analysisPoints == nil {
		g.analysisPoints = make(map[string]AnalysisPoint)
	}
	g.analysisPoints[name] = AnalysisPoint{Name: name}
}

func (g *GeneralizedModel) HasAnalysisPoint(name string) bool {
	if g == nil {
		return false
	}
	_, ok := g.analysisPoints[name]
	return ok
}

func (g *GeneralizedModel) AnalysisPoint(name string) (AnalysisPoint, error) {
	if g == nil {
		return AnalysisPoint{}, fmt.Errorf("GeneralizedModel.AnalysisPoint: nil model: %w", ErrDimensionMismatch)
	}
	ap, ok := g.analysisPoints[name]
	if !ok {
		return AnalysisPoint{}, fmt.Errorf("%q: %w", name, ErrSignalNotFound)
	}
	return ap, nil
}

func (g *GeneralizedModel) CurrentSystem() (*System, error) {
	if g == nil || g.block == nil {
		return nil, fmt.Errorf("GeneralizedModel.CurrentSystem: nil model: %w", ErrDimensionMismatch)
	}
	sys, err := g.block.CurrentSystem()
	if err != nil {
		return nil, err
	}
	if g.inputName != nil {
		sys.InputName = copyStringSlice(g.inputName)
	}
	if g.outputName != nil {
		sys.OutputName = copyStringSlice(g.outputName)
	}
	return sys, nil
}

type GeneralizedClosedLoop struct {
	name                 string
	plant                *System
	controller           NumericBlock
	tunableController    TunableBlock
	analysisPoints       map[string]AnalysisPoint
	primaryAnalysisPoint string
}

func NewGeneralizedClosedLoop(name string, plant *System, controller any, analysisPoint string) (*GeneralizedClosedLoop, error) {
	ctrl, err := numericBlockFromAny(controller)
	if err != nil {
		return nil, err
	}
	if plant == nil {
		return nil, fmt.Errorf("NewGeneralizedClosedLoop: nil plant: %w", ErrDimensionMismatch)
	}
	if analysisPoint == "" {
		return nil, fmt.Errorf("NewGeneralizedClosedLoop: analysis point is empty: %w", ErrDimensionMismatch)
	}
	g := &GeneralizedClosedLoop{
		name:                 name,
		plant:                plant.Copy(),
		controller:           ctrl,
		analysisPoints:       make(map[string]AnalysisPoint),
		primaryAnalysisPoint: analysisPoint,
	}
	if tunable, ok := controller.(TunableBlock); ok {
		g.tunableController = tunable
	}
	g.analysisPoints[analysisPoint] = AnalysisPoint{Name: analysisPoint, Location: AnalysisPointPlantOutput}
	return g, nil
}

// InsertAnalysisPoint binds name to the plant-input or plant-output loop break.
func (g *GeneralizedClosedLoop) InsertAnalysisPoint(name string, location AnalysisPointLocation) error {
	if g == nil {
		return fmt.Errorf("GeneralizedClosedLoop.InsertAnalysisPoint: nil model: %w", ErrDimensionMismatch)
	}
	if name == "" {
		return fmt.Errorf("GeneralizedClosedLoop.InsertAnalysisPoint: name is empty: %w", ErrDimensionMismatch)
	}
	if location != AnalysisPointPlantOutput && location != AnalysisPointPlantInput {
		return fmt.Errorf("GeneralizedClosedLoop.InsertAnalysisPoint: invalid location %d: %w", location, ErrDimensionMismatch)
	}
	if g.analysisPoints == nil {
		g.analysisPoints = make(map[string]AnalysisPoint)
	}
	g.analysisPoints[name] = AnalysisPoint{Name: name, Location: location}
	return nil
}

// withSampledController returns a model bound to sampled without sharing any
// mutable state with the receiver; the plant is shared because it is never
// mutated after construction.
func (g *GeneralizedClosedLoop) withSampledController(sampled TunableBlock) *GeneralizedClosedLoop {
	return &GeneralizedClosedLoop{
		name:                 g.name,
		plant:                g.plant,
		controller:           sampled,
		tunableController:    sampled,
		analysisPoints:       maps.Clone(g.analysisPoints),
		primaryAnalysisPoint: g.primaryAnalysisPoint,
	}
}

func (g *GeneralizedClosedLoop) AnalysisPoint(name string) (AnalysisPoint, error) {
	if g == nil {
		return AnalysisPoint{}, fmt.Errorf("GeneralizedClosedLoop.AnalysisPoint: nil model: %w", ErrDimensionMismatch)
	}
	ap, ok := g.analysisPoints[name]
	if !ok {
		return AnalysisPoint{}, fmt.Errorf("%q: %w", name, ErrSignalNotFound)
	}
	return ap, nil
}

func (g *GeneralizedClosedLoop) OpenLoop(name string) (*System, error) {
	point, err := g.AnalysisPoint(name)
	if err != nil {
		return nil, err
	}
	controller, err := g.controller.CurrentSystem()
	if err != nil {
		return nil, err
	}
	switch point.Location {
	case AnalysisPointPlantOutput:
		return Series(controller, g.plant)
	case AnalysisPointPlantInput:
		return Series(g.plant, controller)
	default:
		return nil, fmt.Errorf("GeneralizedClosedLoop.OpenLoop: analysis point %q has no loop location: %w", name, ErrDimensionMismatch)
	}
}

func (g *GeneralizedClosedLoop) ClosedLoop(name string) (*System, error) {
	return g.ComplementarySensitivity(name)
}

func (g *GeneralizedClosedLoop) ComplementarySensitivity(name string) (*System, error) {
	loop, err := g.OpenLoop(name)
	if err != nil {
		return nil, err
	}
	return Feedback(loop, nil, -1)
}

func (g *GeneralizedClosedLoop) Sensitivity(name string) (*System, error) {
	loop, err := g.OpenLoop(name)
	if err != nil {
		return nil, err
	}
	_, inputs, outputs := loop.Dims()
	if inputs != outputs {
		return nil, fmt.Errorf("GeneralizedClosedLoop.Sensitivity: loop at %q is %dx%d: %w", name, outputs, inputs, ErrDimensionMismatch)
	}
	eye, err := makeIdentityGain(outputs, loop.Dt)
	if err != nil {
		return nil, fmt.Errorf("GeneralizedClosedLoop.Sensitivity: %w", err)
	}
	return Feedback(eye, loop, -1)
}

func (g *GeneralizedClosedLoop) primaryAnalysisPointName() string {
	if g == nil {
		return ""
	}
	if g.primaryAnalysisPoint != "" {
		return g.primaryAnalysisPoint
	}
	return firstAnalysisPointName(g.analysisPoints)
}
