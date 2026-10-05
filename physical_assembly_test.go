package controlsys

import (
	"errors"
	"math/cmplx"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestPhysicalAssemblyConnectedPortsChangeTransferBehavior(t *testing.T) {
	left := physicalCoupledComponent(t, "left", 1)
	right := physicalCoupledComponent(t, "right", 2)
	assembled, err := AssemblePhysical("pair", []PhysicalComponent{left, right}, []PhysicalConnection{
		{FromComponent: "left", FromPort: "mount", ToComponent: "right", ToPort: "mount"},
	})
	if err != nil {
		t.Fatalf("AssemblePhysical: %v", err)
	}
	if n, m, p := assembled.Dims(); n != 4 || m != 2 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (4,2,2)", n, m, p)
	}
	if !assembled.IsDescriptor() {
		t.Fatal("connected physical assembly must retain algebraic port constraints")
	}
	response, err := assembled.FreqResponse([]float64{1})
	if err != nil {
		t.Fatalf("FreqResponse: %v", err)
	}
	want := 1 / complex(3, 2)
	if got := response.At(0, 0, 1); cmplx.Abs(got-want) > 1e-12 {
		t.Fatalf("cross response = %v, want %v", got, want)
	}
	if cmplx.Abs(response.At(0, 0, 1)) == 0 {
		t.Fatal("connected assembly retained block-diagonal transfer behavior")
	}
	if !sameStrings(assembled.InputName, []string{"left.force", "right.force"}) {
		t.Fatalf("input names = %v", assembled.InputName)
	}
	if assembled.Notes != "pair" {
		t.Fatalf("assembly notes = %q, want pair", assembled.Notes)
	}
}

func TestPhysicalAssemblyGroundingIntroducesReactionConstraint(t *testing.T) {
	component := physicalCoupledComponent(t, "mass", 1)
	assembled, err := AssemblePhysical("grounded", []PhysicalComponent{component}, []PhysicalConnection{
		{FromComponent: "mass", FromPort: "mount", Grounded: true},
	})
	if err != nil {
		t.Fatalf("AssemblePhysical: %v", err)
	}
	if n, m, p := assembled.Dims(); n != 2 || m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (2,1,1)", n, m, p)
	}
	response, err := assembled.FreqResponse([]float64{0, 1, 10})
	if err != nil {
		t.Fatalf("FreqResponse: %v", err)
	}
	for k := range response.NFreq {
		if got := cmplx.Abs(response.At(k, 0, 0)); got > 1e-12 {
			t.Fatalf("grounded response[%d] = %g, want zero", k, got)
		}
	}
	if assembled.E.At(1, 1) != 0 || assembled.A.At(1, 0) != 1 || assembled.A.At(0, 1) != 1 {
		t.Fatalf("ground reaction descriptor equations are incorrect: E=%v A=%v", mat.Formatted(assembled.E), mat.Formatted(assembled.A))
	}
}

func TestPhysicalAssemblyBuildsMultiPortNode(t *testing.T) {
	components := []PhysicalComponent{
		physicalCoupledComponent(t, "one", 1),
		physicalCoupledComponent(t, "two", 2),
		physicalCoupledComponent(t, "three", 3),
	}
	assembled, err := AssemblePhysical("node", components, []PhysicalConnection{
		{FromComponent: "one", FromPort: "mount", ToComponent: "two", ToPort: "mount"},
		{FromComponent: "two", FromPort: "mount", ToComponent: "three", ToPort: "mount"},
	})
	if err != nil {
		t.Fatalf("AssemblePhysical: %v", err)
	}
	if n, m, p := assembled.Dims(); n != 6 || m != 3 || p != 3 {
		t.Fatalf("dims = (%d,%d,%d), want (6,3,3)", n, m, p)
	}
	response, err := assembled.FreqResponse([]float64{1})
	if err != nil {
		t.Fatal(err)
	}
	want := 1 / complex(6, 3)
	if got := response.At(0, 0, 2); cmplx.Abs(got-want) > 1e-12 {
		t.Fatalf("three-component cross response = %v, want %v", got, want)
	}
}

func TestPhysicalAssemblyPreservesUnconnectedDescriptorComponent(t *testing.T) {
	component := physicalDescriptorComponent(t, "mass")
	assembled, err := AssemblePhysical("unconnected", []PhysicalComponent{component}, nil)
	if err != nil {
		t.Fatalf("AssemblePhysical: %v", err)
	}
	if !assembled.IsDescriptor() || !matEqual(assembled.A, component.System.A, 1e-12) || !matEqual(assembled.E, component.System.E, 1e-12) {
		t.Fatalf("unconnected descriptor component was not preserved")
	}
	if !sameStrings(assembled.StateName, []string{"mass.x1", "mass.x2"}) {
		t.Fatalf("state names = %v", assembled.StateName)
	}
}

func TestPhysicalAssemblyOwnsConnectedResult(t *testing.T) {
	left := physicalCoupledComponent(t, "left", 1)
	right := physicalCoupledComponent(t, "right", 2)
	assembled, err := AssemblePhysical("pair", []PhysicalComponent{left, right}, []PhysicalConnection{
		{FromComponent: "left", FromPort: "mount", ToComponent: "right", ToPort: "mount"},
	})
	if err != nil {
		t.Fatal(err)
	}

	assembledA := assembled.A.At(0, 0)
	assembledInputName := assembled.InputName[0]
	left.System.A.Set(0, 0, 99)
	left.System.InputName[0] = "changed"
	left.Ports[0].Name = "changed"
	left.Ports[0].Input[0] = 0
	if assembled.A.At(0, 0) != assembledA || assembled.InputName[0] != assembledInputName {
		t.Fatal("source mutation changed assembled result")
	}

	sourceA := left.System.A.At(0, 0)
	sourceInputName := left.System.InputName[0]
	assembled.A.Set(0, 0, -123)
	assembled.InputName[0] = "result.changed"
	if left.System.A.At(0, 0) != sourceA || left.System.InputName[0] != sourceInputName {
		t.Fatal("assembled result mutation changed source component")
	}
}

func TestPhysicalAssemblyImplicitPortsSkipExplicitBindings(t *testing.T) {
	explicit := PhysicalPort{Name: "external", Kind: PhysicalPortDisplacement, Dimension: 1, Input: []int{0}, Output: []int{0}}
	implicit := PhysicalPort{Name: "mount", Kind: PhysicalPortDisplacement, Dimension: 2}
	for _, test := range []struct {
		name  string
		ports []PhysicalPort
	}{
		{name: "explicit first", ports: []PhysicalPort{explicit, implicit}},
		{name: "explicit last", ports: []PhysicalPort{implicit, explicit}},
	} {
		t.Run(test.name, func(t *testing.T) {
			component := physicalMixedBindingComponent(t, "mixed", test.ports)
			assembled, err := AssemblePhysical("grounded", []PhysicalComponent{component}, []PhysicalConnection{
				{FromComponent: "mixed", FromPort: "mount", Grounded: true},
			})
			if err != nil {
				t.Fatalf("AssemblePhysical: %v", err)
			}
			if n, m, p := assembled.Dims(); n != 3 || m != 1 || p != 1 {
				t.Fatalf("dims = (%d,%d,%d), want (3,1,1)", n, m, p)
			}
			if !sameStrings(assembled.InputName, []string{"mixed.external.force"}) {
				t.Fatalf("input names = %v", assembled.InputName)
			}
			if !sameStrings(assembled.OutputName, []string{"mixed.external.position"}) {
				t.Fatalf("output names = %v", assembled.OutputName)
			}
		})
	}
}

func TestPhysicalAssemblyRejectsInvalidTopologies(t *testing.T) {
	left := physicalCoupledComponent(t, "left", 1)
	right := physicalCoupledComponent(t, "right", 2)
	right.Ports[0].Kind = PhysicalPortEffort
	connection := PhysicalConnection{FromComponent: "left", FromPort: "mount", ToComponent: "right", ToPort: "mount"}
	if _, err := AssemblePhysical("bad", []PhysicalComponent{left, right}, []PhysicalConnection{connection}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("incompatible port err = %v, want ErrDimensionMismatch", err)
	}
	right.Ports[0].Kind = PhysicalPortDisplacement
	if _, err := AssemblePhysical("duplicate", []PhysicalComponent{left, right}, []PhysicalConnection{connection, connection}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("duplicate connection err = %v, want ErrInvalidArgument", err)
	}
	badMapping := left
	badMapping.Ports = copyPhysicalPorts(left.Ports)
	badMapping.Ports[0].Input = []int{4}
	if _, err := AssemblePhysical("mapping", []PhysicalComponent{badMapping}, []PhysicalConnection{{FromComponent: "left", FromPort: "mount", Grounded: true}}); !errors.Is(err, ErrInvalidArgument) || !strings.HasPrefix(err.Error(), "AssemblePhysical: ") {
		t.Fatalf("bad channel mapping err = %v, want ErrInvalidArgument", err)
	}
	if _, err := NewPhysicalComponent("left", left.System, badMapping.Ports); !errors.Is(err, ErrInvalidArgument) || !strings.HasPrefix(err.Error(), "NewPhysicalComponent: ") {
		t.Fatalf("constructor bad mapping err = %v, want ErrInvalidArgument", err)
	}
	if _, err := NewPhysicalComponent("nil", nil, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil system err = %v, want ErrInvalidArgument", err)
	}
	if _, err := NewPhysicalComponent("", left.System, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("empty name err = %v, want ErrInvalidArgument", err)
	}
	if _, err := NewPhysicalComponent("dup", left.System, []PhysicalPort{{Name: "a", Dimension: 1}, {Name: "a", Dimension: 1}}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("duplicate port err = %v, want ErrInvalidArgument", err)
	}
	if _, err := NewPhysicalComponent("short", left.System, []PhysicalPort{{Name: "a", Dimension: 3}}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("too few channels err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := AssemblePhysical("", []PhysicalComponent{left}, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("empty assembly name err = %v, want ErrInvalidArgument", err)
	}
	delayed := physicalCoupledComponent(t, "delayed", 1)
	delayed.System.InputDelay = []float64{0.5, 0}
	if _, err := AssemblePhysical("delay", []PhysicalComponent{left, delayed}, []PhysicalConnection{{FromComponent: "left", FromPort: "mount", ToComponent: "delayed", ToPort: "mount"}}); !errors.Is(err, ErrDelayUnsupported) {
		t.Fatalf("delayed err = %v, want ErrDelayUnsupported", err)
	}
}

func physicalCoupledComponent(t *testing.T, name string, decay float64) PhysicalComponent {
	t.Helper()
	sys, err := New(
		mat.NewDense(1, 1, []float64{-decay}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(2, 2, nil),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputName = []string{"force", "mount.force"}
	sys.OutputName = []string{"position", "mount.position"}
	sys.StateName = []string{"position"}
	return mustPhysicalComponent(t, name, sys, []PhysicalPort{{
		Name: "mount", Kind: PhysicalPortDisplacement, Dimension: 1, Input: []int{1}, Output: []int{1},
	}})
}

func mustPhysicalComponent(t *testing.T, name string, sys *System, ports []PhysicalPort) PhysicalComponent {
	t.Helper()
	c, err := NewPhysicalComponent(name, sys, ports)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func physicalDescriptorComponent(t *testing.T, name string) PhysicalComponent {
	t.Helper()
	sys, err := NewDescriptor(
		mat.NewDense(2, 2, []float64{-1, 0.5, -2, -3}),
		mat.NewDense(2, 1, []float64{1, -1}),
		mat.NewDense(1, 2, []float64{2, -0.5}),
		mat.NewDense(1, 1, nil),
		mat.NewDense(2, 2, []float64{2, 0, 0.1, 3}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputName = []string{"force"}
	sys.OutputName = []string{"position"}
	sys.StateName = []string{"x1", "x2"}
	return mustPhysicalComponent(t, name, sys, []PhysicalPort{{Name: "mount", Kind: PhysicalPortDisplacement, Dimension: 1}})
}

func physicalMixedBindingComponent(t *testing.T, name string, ports []PhysicalPort) PhysicalComponent {
	t.Helper()
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 3, []float64{1, 2, 3}),
		mat.NewDense(3, 1, []float64{1, 2, 3}),
		mat.NewDense(3, 3, nil),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputName = []string{"external.force", "mount.force[0]", "mount.force[1]"}
	sys.OutputName = []string{"external.position", "mount.position[0]", "mount.position[1]"}
	return mustPhysicalComponent(t, name, sys, ports)
}
