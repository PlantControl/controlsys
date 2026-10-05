package controlsys

import (
	"fmt"
	"maps"
)

// NumericBlock is a model block whose current value is a state-space model,
// like a MATLAB Control Design Block; FixedBlock wraps a fixed *System.
type NumericBlock interface {
	CurrentSystem() (*System, error)
}

// TunableBlock is a NumericBlock with free parameters that Systune adjusts;
// SampleBlock returns a copy with the named parameters set.
type TunableBlock interface {
	NumericBlock
	FreeParameters() []*TunableReal
	SampleBlock(map[string]float64) (TunableBlock, error)
}

type fixedSystemBlock struct {
	sys *System
}

// FixedBlock wraps a fixed model as a NumericBlock. The model is copied, so
// later changes to sys do not affect the block. A nil or invalid sys returns
// ErrInvalidArgument.
func FixedBlock(sys *System) (NumericBlock, error) {
	if err := requireSystem("FixedBlock", sys); err != nil {
		return nil, err
	}
	return fixedSystemBlock{sys: sys.Copy()}, nil
}

func (b fixedSystemBlock) CurrentSystem() (*System, error) {
	return b.sys.Copy(), nil
}

// GeneralizedModel is a named NumericBlock with optional signal names and
// analysis points, a minimal counterpart of a MATLAB genss model.
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
	// AnalysisPointUnspecified marks a point with no loop location, as
	// GeneralizedModel records; it cannot break a closed loop.
	AnalysisPointUnspecified AnalysisPointLocation = iota
	// AnalysisPointPlantOutput breaks the loop at the plant output.
	AnalysisPointPlantOutput
	// AnalysisPointPlantInput breaks the loop at the plant input.
	AnalysisPointPlantInput
)

// AnalysisPoint binds a name to a loop location.
type AnalysisPoint struct {
	Name     string
	Location AnalysisPointLocation
}

// NewGeneralizedModel wraps block under name; an empty name or nil block
// returns ErrInvalidArgument.
func NewGeneralizedModel(name string, block NumericBlock) (*GeneralizedModel, error) {
	if name == "" {
		return nil, fmt.Errorf("NewGeneralizedModel: name is empty: %w", ErrInvalidArgument)
	}
	if block == nil {
		return nil, fmt.Errorf("NewGeneralizedModel: block is nil: %w", ErrInvalidArgument)
	}
	return &GeneralizedModel{name: name, block: block, analysisPoints: make(map[string]AnalysisPoint)}, nil
}

// SetInputName records input names applied by CurrentSystem, which checks
// them against the block's inputs as System.SetInputName does.
func (g *GeneralizedModel) SetInputName(names ...string) error {
	if g == nil {
		return fmt.Errorf("GeneralizedModel.SetInputName: nil model: %w", ErrInvalidArgument)
	}
	g.inputName = copyStringSlice(names)
	return nil
}

// SetOutputName records output names applied by CurrentSystem, which checks
// them against the block's outputs as System.SetOutputName does.
func (g *GeneralizedModel) SetOutputName(names ...string) error {
	if g == nil {
		return fmt.Errorf("GeneralizedModel.SetOutputName: nil model: %w", ErrInvalidArgument)
	}
	g.outputName = copyStringSlice(names)
	return nil
}

// InsertAnalysisPoint records a named analysis point without a loop
// location; an empty name or nil model returns ErrInvalidArgument.
func (g *GeneralizedModel) InsertAnalysisPoint(name string) error {
	if g == nil {
		return fmt.Errorf("GeneralizedModel.InsertAnalysisPoint: nil model: %w", ErrInvalidArgument)
	}
	if name == "" {
		return fmt.Errorf("GeneralizedModel.InsertAnalysisPoint: name is empty: %w", ErrInvalidArgument)
	}
	if g.analysisPoints == nil {
		g.analysisPoints = make(map[string]AnalysisPoint)
	}
	g.analysisPoints[name] = AnalysisPoint{Name: name}
	return nil
}

// HasAnalysisPoint reports whether name was inserted; false for a nil model.
func (g *GeneralizedModel) HasAnalysisPoint(name string) bool {
	if g == nil {
		return false
	}
	_, ok := g.analysisPoints[name]
	return ok
}

// AnalysisPoint returns the analysis point inserted as name, or
// ErrSignalNotFound.
func (g *GeneralizedModel) AnalysisPoint(name string) (AnalysisPoint, error) {
	if g == nil {
		return AnalysisPoint{}, fmt.Errorf("GeneralizedModel.AnalysisPoint: nil model: %w", ErrInvalidArgument)
	}
	ap, ok := g.analysisPoints[name]
	if !ok {
		return AnalysisPoint{}, fmt.Errorf("GeneralizedModel.AnalysisPoint: analysis point %q: %w", name, ErrSignalNotFound)
	}
	return ap, nil
}

// CurrentSystem returns the block's current model with the recorded signal
// names, like MATLAB getValue; a name count that does not match the model
// returns ErrDimensionMismatch.
func (g *GeneralizedModel) CurrentSystem() (*System, error) {
	if g == nil || g.block == nil {
		return nil, fmt.Errorf("GeneralizedModel.CurrentSystem: nil model: %w", ErrInvalidArgument)
	}
	sys, err := g.block.CurrentSystem()
	if err != nil {
		return nil, fmt.Errorf("GeneralizedModel.CurrentSystem: %w", err)
	}
	if g.inputName != nil {
		if err := sys.SetInputName(g.inputName...); err != nil {
			return nil, fmt.Errorf("GeneralizedModel.CurrentSystem: %w", err)
		}
	}
	if g.outputName != nil {
		if err := sys.SetOutputName(g.outputName...); err != nil {
			return nil, fmt.Errorf("GeneralizedModel.CurrentSystem: %w", err)
		}
	}
	return sys, nil
}

// GeneralizedClosedLoop is the negative-feedback loop of a fixed plant and a
// (possibly tunable) controller with named analysis points, the model
// Systune tunes.
type GeneralizedClosedLoop struct {
	name                 string
	plant                *System
	controller           NumericBlock
	tunableController    TunableBlock
	analysisPoints       map[string]AnalysisPoint
	primaryAnalysisPoint string
}

// NewGeneralizedClosedLoop builds the loop of plant (copied) and controller
// with analysisPoint at the plant output. A nil plant or controller and an
// empty analysis point name return ErrInvalidArgument.
func NewGeneralizedClosedLoop(name string, plant *System, controller NumericBlock, analysisPoint string) (*GeneralizedClosedLoop, error) {
	if controller == nil {
		return nil, fmt.Errorf("NewGeneralizedClosedLoop: controller is nil: %w", ErrInvalidArgument)
	}
	if err := requireSystem("NewGeneralizedClosedLoop", plant); err != nil {
		return nil, err
	}
	if analysisPoint == "" {
		return nil, fmt.Errorf("NewGeneralizedClosedLoop: analysis point is empty: %w", ErrInvalidArgument)
	}
	g := &GeneralizedClosedLoop{
		name:                 name,
		plant:                plant.Copy(),
		controller:           controller,
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
		return fmt.Errorf("GeneralizedClosedLoop.InsertAnalysisPoint: nil model: %w", ErrInvalidArgument)
	}
	if name == "" {
		return fmt.Errorf("GeneralizedClosedLoop.InsertAnalysisPoint: name is empty: %w", ErrInvalidArgument)
	}
	if location != AnalysisPointPlantOutput && location != AnalysisPointPlantInput {
		return fmt.Errorf("GeneralizedClosedLoop.InsertAnalysisPoint: invalid location %d: %w", location, ErrInvalidArgument)
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

// AnalysisPoint returns the analysis point registered as name, or
// ErrSignalNotFound.
func (g *GeneralizedClosedLoop) AnalysisPoint(name string) (AnalysisPoint, error) {
	ap, err := g.analysisPoint(name)
	if err != nil {
		return AnalysisPoint{}, fmt.Errorf("GeneralizedClosedLoop.AnalysisPoint: %w", err)
	}
	return ap, nil
}

func (g *GeneralizedClosedLoop) analysisPoint(name string) (AnalysisPoint, error) {
	if g == nil {
		return AnalysisPoint{}, fmt.Errorf("nil model: %w", ErrInvalidArgument)
	}
	ap, ok := g.analysisPoints[name]
	if !ok {
		return AnalysisPoint{}, fmt.Errorf("analysis point %q: %w", name, ErrSignalNotFound)
	}
	return ap, nil
}

// OpenLoop returns the loop transfer broken at the named analysis point,
// L = P·C at the plant output and L = C·P at the plant input, for the
// negative-feedback loop that ComplementarySensitivity and Sensitivity close.
func (g *GeneralizedClosedLoop) OpenLoop(name string) (*System, error) {
	loop, err := g.openLoop(name)
	if err != nil {
		return nil, fmt.Errorf("GeneralizedClosedLoop.OpenLoop: %w", err)
	}
	return loop, nil
}

func (g *GeneralizedClosedLoop) openLoop(name string) (*System, error) {
	point, err := g.analysisPoint(name)
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
		return nil, fmt.Errorf("analysis point %q has no loop location: %w", name, ErrInvalidArgument)
	}
}

// ClosedLoop is ComplementarySensitivity.
func (g *GeneralizedClosedLoop) ClosedLoop(name string) (*System, error) {
	t, err := g.complementarySensitivity(name)
	if err != nil {
		return nil, fmt.Errorf("GeneralizedClosedLoop.ClosedLoop: %w", err)
	}
	return t, nil
}

// ComplementarySensitivity returns T = L(I + L)⁻¹ for the open loop L at
// the named analysis point, like MATLAB getCompSensitivity.
func (g *GeneralizedClosedLoop) ComplementarySensitivity(name string) (*System, error) {
	t, err := g.complementarySensitivity(name)
	if err != nil {
		return nil, fmt.Errorf("GeneralizedClosedLoop.ComplementarySensitivity: %w", err)
	}
	return t, nil
}

func (g *GeneralizedClosedLoop) complementarySensitivity(name string) (*System, error) {
	loop, err := g.openLoop(name)
	if err != nil {
		return nil, err
	}
	return Feedback(loop, nil, -1)
}

// Sensitivity returns S = (I + L)⁻¹ for the square open loop L at the named
// analysis point, like MATLAB getSensitivity.
func (g *GeneralizedClosedLoop) Sensitivity(name string) (*System, error) {
	loop, err := g.openLoop(name)
	if err != nil {
		return nil, fmt.Errorf("GeneralizedClosedLoop.Sensitivity: %w", err)
	}
	_, inputs, outputs := loop.Dims()
	if inputs != outputs {
		return nil, fmt.Errorf("GeneralizedClosedLoop.Sensitivity: loop at %q is %dx%d: %w", name, outputs, inputs, ErrDimensionMismatch)
	}
	eye, err := makeIdentityGain(outputs, loop.Dt)
	if err != nil {
		return nil, fmt.Errorf("GeneralizedClosedLoop.Sensitivity: %w", err)
	}
	s, err := Feedback(eye, loop, -1)
	if err != nil {
		return nil, fmt.Errorf("GeneralizedClosedLoop.Sensitivity: %w", err)
	}
	return s, nil
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

func firstAnalysisPointName(points map[string]AnalysisPoint) string {
	first := ""
	for name := range points {
		if first == "" || name < first {
			first = name
		}
	}
	return first
}
