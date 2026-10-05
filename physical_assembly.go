package controlsys

import (
	"fmt"
	"sort"

	"plantcontrol.org/v1/gonum/mat"
)

// PhysicalPortKind is the physical domain of a port; only ports of the same
// kind and dimension can be connected.
type PhysicalPortKind int

const (
	// PhysicalPortDisplacement is a mechanical port whose across variable is
	// a displacement and whose through variable a force.
	PhysicalPortDisplacement PhysicalPortKind = iota
	// PhysicalPortEffort is a port whose across variable is an effort (for
	// example a voltage or pressure) and whose through variable a flow.
	PhysicalPortEffort
)

// PhysicalPort binds Dimension input channels (through variables) and
// Dimension output channels (across variables) of a component to a named
// port. Input and Output are 0-based channel indices; leaving both empty
// claims the lowest unused channels in order.
type PhysicalPort struct {
	Name      string
	Kind      PhysicalPortKind
	Dimension int
	Input     []int
	Output    []int
}

// PhysicalComponent is a named model with physical ports; build it with
// NewPhysicalComponent.
type PhysicalComponent struct {
	Name   string
	System *System
	Ports  []PhysicalPort
}

// PhysicalConnection joins port FromComponent.FromPort to ToComponent.ToPort,
// or, with Grounded, grounds the From port (To* are then ignored).
type PhysicalConnection struct {
	FromComponent string
	FromPort      string
	ToComponent   string
	ToPort        string
	Grounded      bool
}

// NewPhysicalComponent returns a component holding copies of sys and ports.
// An empty name, a nil or invalid sys, and invalid ports (empty or duplicate
// name, non-positive dimension, unknown kind, out-of-range or reused
// channels) return ErrInvalidArgument; channel-count mismatches return
// ErrDimensionMismatch.
func NewPhysicalComponent(name string, sys *System, ports []PhysicalPort) (PhysicalComponent, error) {
	if err := requireSystem("NewPhysicalComponent", sys); err != nil {
		return PhysicalComponent{}, err
	}
	component := PhysicalComponent{Name: name, System: sys.Copy(), Ports: copyPhysicalPorts(ports)}
	if _, err := newPhysicalAssemblyPlan(name, []PhysicalComponent{component}, nil); err != nil {
		return PhysicalComponent{}, fmt.Errorf("NewPhysicalComponent: %w", err)
	}
	return component, nil
}

// AssemblePhysical connects components into one descriptor model named name,
// imposing equal across variables and zero net through variable at every
// node. Unconnected channels remain external inputs and outputs; without
// connections the components are appended. Invalid names, components,
// ports or topologies return ErrInvalidArgument, incompatible channel counts
// ErrDimensionMismatch, and connected delayed components
// ErrDelayUnsupported.
func AssemblePhysical(name string, components []PhysicalComponent, connections []PhysicalConnection) (*System, error) {
	plan, err := newPhysicalAssemblyPlan(name, components, connections)
	if err != nil {
		return nil, fmt.Errorf("AssemblePhysical: %w", err)
	}
	sys, err := plan.assemble()
	if err != nil {
		return nil, fmt.Errorf("AssemblePhysical: %w", err)
	}
	return sys, nil
}

type physicalAssemblyPlan struct {
	name             string
	components       []PhysicalComponent
	ports            []physicalPortBinding
	groups           []physicalPortGroup
	stateOffsets     []int
	inputOffsets     []int
	outputOffsets    []int
	totalStates      int
	totalInputs      int
	totalOutputs     int
	internalInputs   []int
	internalOutputs  []int
	externalInputs   []int
	externalOutputs  []int
	internalInputPos map[int]int
}

type physicalPortBinding struct {
	key       string
	component int
	port      PhysicalPort
	inputs    []int
	outputs   []int
}

type physicalPortGroup struct {
	ports    []int
	grounded bool
}

func newPhysicalAssemblyPlan(name string, components []PhysicalComponent, connections []PhysicalConnection) (*physicalAssemblyPlan, error) {
	if name == "" {
		return nil, fmt.Errorf("empty name: %w", ErrInvalidArgument)
	}
	if len(components) == 0 {
		return nil, fmt.Errorf("no components: %w", ErrInvalidArgument)
	}
	plan := &physicalAssemblyPlan{
		name:          name,
		components:    components,
		stateOffsets:  make([]int, len(components)),
		inputOffsets:  make([]int, len(components)),
		outputOffsets: make([]int, len(components)),
	}
	byComponent := make(map[string]int, len(components))
	for i, component := range components {
		if component.Name == "" || component.System == nil {
			return nil, fmt.Errorf("component %d has an empty name or nil system: %w", i, ErrInvalidArgument)
		}
		if _, exists := byComponent[component.Name]; exists {
			return nil, fmt.Errorf("duplicate component %q: %w", component.Name, ErrInvalidArgument)
		}
		if err := component.System.Validate(); err != nil {
			return nil, fmt.Errorf("component %q: %w", component.Name, err)
		}
		if i > 0 && component.System.Dt != components[0].System.Dt {
			return nil, fmt.Errorf("component %q has sample time %g, want %g: %w", component.Name, component.System.Dt, components[0].System.Dt, ErrDomainMismatch)
		}
		plan.stateOffsets[i] = plan.totalStates
		plan.inputOffsets[i] = plan.totalInputs
		plan.outputOffsets[i] = plan.totalOutputs
		n, m, p := component.System.Dims()
		plan.totalStates += n
		plan.totalInputs += m
		plan.totalOutputs += p
		byComponent[component.Name] = i
	}
	portIndex, err := plan.bindPorts()
	if err != nil {
		return nil, err
	}
	if err := plan.bindConnections(connections, portIndex); err != nil {
		return nil, err
	}
	return plan, nil
}

func (p *physicalAssemblyPlan) bindPorts() (map[string]int, error) {
	portIndex := make(map[string]int)
	for componentIndex, component := range p.components {
		_, m, outputs := component.System.Dims()
		usedInputs := make([]bool, m)
		usedOutputs := make([]bool, outputs)
		seenPorts := make(map[string]struct{}, len(component.Ports))
		for _, port := range component.Ports {
			if port.Name == "" || port.Dimension <= 0 {
				return nil, fmt.Errorf("port %q on %q needs a name and positive dimension: %w", port.Name, component.Name, ErrInvalidArgument)
			}
			if port.Kind != PhysicalPortDisplacement && port.Kind != PhysicalPortEffort {
				return nil, fmt.Errorf("invalid port kind %d on %s.%s: %w", port.Kind, component.Name, port.Name, ErrInvalidArgument)
			}
			key := component.Name + "." + port.Name
			if _, exists := seenPorts[key]; exists {
				return nil, fmt.Errorf("duplicate port %q: %w", key, ErrInvalidArgument)
			}
			seenPorts[key] = struct{}{}
			if len(port.Input) == 0 && len(port.Output) == 0 {
				continue
			}
			if len(port.Input) != port.Dimension || len(port.Output) != port.Dimension {
				return nil, fmt.Errorf("port %q must bind %d input and output channels: %w", key, port.Dimension, ErrDimensionMismatch)
			}
			if err := reservePhysicalChannels(usedInputs, port.Input, "input", key); err != nil {
				return nil, err
			}
			if err := reservePhysicalChannels(usedOutputs, port.Output, "output", key); err != nil {
				return nil, err
			}
		}
		for _, port := range component.Ports {
			key := component.Name + "." + port.Name
			inputs := copyIntSlice(port.Input)
			outputsForPort := copyIntSlice(port.Output)
			if len(inputs) == 0 && len(outputsForPort) == 0 {
				inputs = claimUnusedPhysicalChannels(usedInputs, port.Dimension)
				if len(inputs) != port.Dimension {
					return nil, fmt.Errorf("not enough unused input channels for %q: %w", key, ErrDimensionMismatch)
				}
				outputsForPort = claimUnusedPhysicalChannels(usedOutputs, port.Dimension)
				if len(outputsForPort) != port.Dimension {
					return nil, fmt.Errorf("not enough unused output channels for %q: %w", key, ErrDimensionMismatch)
				}
			}
			binding := physicalPortBinding{key: key, component: componentIndex, port: port, inputs: make([]int, port.Dimension), outputs: make([]int, port.Dimension)}
			for i := range port.Dimension {
				binding.inputs[i] = p.inputOffsets[componentIndex] + inputs[i]
				binding.outputs[i] = p.outputOffsets[componentIndex] + outputsForPort[i]
			}
			portIndex[key] = len(p.ports)
			p.ports = append(p.ports, binding)
		}
	}
	return portIndex, nil
}

func (p *physicalAssemblyPlan) bindConnections(connections []PhysicalConnection, portIndex map[string]int) error {
	parent := make([]int, len(p.ports))
	for i := range parent {
		parent[i] = i
	}
	active := make([]bool, len(p.ports))
	grounded := make([]bool, len(p.ports))
	seen := make(map[string]bool)
	for _, connection := range connections {
		from, err := physicalPortIndex(portIndex, connection.FromComponent, connection.FromPort)
		if err != nil {
			return err
		}
		active[from] = true
		if connection.Grounded {
			key := "ground:" + p.ports[from].key
			if seen[key] {
				return fmt.Errorf("duplicate grounding of %s: %w", p.ports[from].key, ErrInvalidArgument)
			}
			seen[key] = true
			grounded[from] = true
			continue
		}
		to, err := physicalPortIndex(portIndex, connection.ToComponent, connection.ToPort)
		if err != nil {
			return err
		}
		if from == to {
			return fmt.Errorf("port %s cannot connect to itself: %w", p.ports[from].key, ErrInvalidArgument)
		}
		fromPort, toPort := p.ports[from].port, p.ports[to].port
		if fromPort.Kind != toPort.Kind || fromPort.Dimension != toPort.Dimension {
			return fmt.Errorf("incompatible ports %s and %s: %w", p.ports[from].key, p.ports[to].key, ErrDimensionMismatch)
		}
		edge := normalizedPhysicalEdge(p.ports[from].key, p.ports[to].key)
		if seen[edge] {
			return fmt.Errorf("duplicate connection %s: %w", edge, ErrInvalidArgument)
		}
		seen[edge] = true
		active[to] = true
		unionPhysicalPorts(parent, from, to)
	}
	groups := make(map[int]*physicalPortGroup)
	for i := range p.ports {
		if !active[i] {
			continue
		}
		root := findPhysicalPort(parent, i)
		group := groups[root]
		if group == nil {
			group = &physicalPortGroup{}
			groups[root] = group
		}
		group.ports = append(group.ports, i)
		group.grounded = group.grounded || grounded[i]
	}
	roots := make([]int, 0, len(groups))
	for root := range groups {
		roots = append(roots, root)
	}
	sort.Ints(roots)
	for _, root := range roots {
		group := groups[root]
		if !group.grounded && len(group.ports) < 2 {
			return fmt.Errorf("ungrounded node has fewer than two ports: %w", ErrInvalidArgument)
		}
		p.groups = append(p.groups, *group)
		for _, portIndex := range group.ports {
			p.internalInputs = append(p.internalInputs, p.ports[portIndex].inputs...)
			p.internalOutputs = append(p.internalOutputs, p.ports[portIndex].outputs...)
		}
	}
	p.internalInputPos = make(map[int]int, len(p.internalInputs))
	for position, channel := range p.internalInputs {
		p.internalInputPos[channel] = position
	}
	p.externalInputs = complementChannels(p.totalInputs, p.internalInputs)
	p.externalOutputs = complementChannels(p.totalOutputs, p.internalOutputs)
	return nil
}

func (p *physicalAssemblyPlan) assemble() (*System, error) {
	if len(p.groups) == 0 {
		return p.appendComponents()
	}
	for _, component := range p.components {
		if component.System.HasDelay() {
			return nil, fmt.Errorf("connected delayed components are not supported: %w", ErrDelayUnsupported)
		}
	}
	a, b, c, d, e := p.aggregateMatrices()
	q := len(p.internalInputs)
	n := p.totalStates
	nAugmented := n + q
	aAugmented := newDense(nAugmented, nAugmented)
	eAugmented := newDense(nAugmented, nAugmented)
	bAugmented := newDense(nAugmented, len(p.externalInputs))
	setBlock(aAugmented, 0, 0, a)
	setBlock(eAugmented, 0, 0, e)
	aAugmentedRaw := aAugmented.RawMatrix()
	bAugmentedRaw := bAugmented.RawMatrix()
	bRaw := b.RawMatrix()
	for state := range n {
		for position, channel := range p.internalInputs {
			aAugmentedRaw.Data[state*aAugmentedRaw.Stride+n+position] = bRaw.Data[state*bRaw.Stride+channel]
		}
		for position, channel := range p.externalInputs {
			bAugmentedRaw.Data[state*bAugmentedRaw.Stride+position] = bRaw.Data[state*bRaw.Stride+channel]
		}
	}
	constraint := 0
	for _, group := range p.groups {
		dimension := p.ports[group.ports[0]].port.Dimension
		if group.grounded {
			for _, portIndex := range group.ports {
				for coordinate := range dimension {
					p.addAcrossConstraint(aAugmented, bAugmented, c, d, n+constraint, portIndex, coordinate, 1)
					constraint++
				}
			}
			continue
		}
		reference := group.ports[0]
		for _, portIndex := range group.ports[1:] {
			for coordinate := range dimension {
				row := n + constraint
				p.addAcrossConstraint(aAugmented, bAugmented, c, d, row, portIndex, coordinate, 1)
				p.addAcrossConstraint(aAugmented, bAugmented, c, d, row, reference, coordinate, -1)
				constraint++
			}
		}
		for coordinate := range dimension {
			row := n + constraint
			for _, portIndex := range group.ports {
				channel := p.ports[portIndex].inputs[coordinate]
				column := n + p.internalInputPos[channel]
				aAugmentedRaw.Data[row*aAugmentedRaw.Stride+column]++
			}
			constraint++
		}
	}
	if constraint != q {
		return nil, fmt.Errorf("generated %d constraints for %d internal variables: %w", constraint, q, ErrDimensionMismatch)
	}
	cAugmented := newDense(len(p.externalOutputs), nAugmented)
	dAugmented := newDense(len(p.externalOutputs), len(p.externalInputs))
	cRaw := c.RawMatrix()
	dRaw := d.RawMatrix()
	cAugmentedRaw := cAugmented.RawMatrix()
	dAugmentedRaw := dAugmented.RawMatrix()
	for row, output := range p.externalOutputs {
		for state := range n {
			cAugmentedRaw.Data[row*cAugmentedRaw.Stride+state] = cRaw.Data[output*cRaw.Stride+state]
		}
		for position, input := range p.internalInputs {
			cAugmentedRaw.Data[row*cAugmentedRaw.Stride+n+position] = dRaw.Data[output*dRaw.Stride+input]
		}
		for column, input := range p.externalInputs {
			dAugmentedRaw.Data[row*dAugmentedRaw.Stride+column] = dRaw.Data[output*dRaw.Stride+input]
		}
	}
	var constructorB, constructorC, constructorD *mat.Dense
	if len(p.externalInputs) > 0 {
		constructorB = bAugmented
	}
	if len(p.externalOutputs) > 0 {
		constructorC = cAugmented
	}
	if len(p.externalInputs) > 0 && len(p.externalOutputs) > 0 {
		constructorD = dAugmented
	}
	result, err := newDescriptorOwned(aAugmented, constructorB, constructorC, constructorD, eAugmented, p.components[0].System.Dt)
	if err != nil {
		return nil, err
	}
	result.InputName = selectedPhysicalNames(p.aggregateInputNames(), p.externalInputs)
	result.OutputName = selectedPhysicalNames(p.aggregateOutputNames(), p.externalOutputs)
	result.StateName = append(p.aggregateStateNames(), p.internalStateNames()...)
	result.Notes = p.name
	return result, nil
}

func (p *physicalAssemblyPlan) addAcrossConstraint(a, b, c, d *mat.Dense, row, portIndex, coordinate int, factor float64) {
	output := p.ports[portIndex].outputs[coordinate]
	aRaw := a.RawMatrix()
	bRaw := b.RawMatrix()
	cRaw := c.RawMatrix()
	dRaw := d.RawMatrix()
	for state := range p.totalStates {
		aRaw.Data[row*aRaw.Stride+state] += factor * cRaw.Data[output*cRaw.Stride+state]
	}
	for position, input := range p.internalInputs {
		column := p.totalStates + position
		aRaw.Data[row*aRaw.Stride+column] += factor * dRaw.Data[output*dRaw.Stride+input]
	}
	for position, input := range p.externalInputs {
		bRaw.Data[row*bRaw.Stride+position] += factor * dRaw.Data[output*dRaw.Stride+input]
	}
}

func (p *physicalAssemblyPlan) aggregateMatrices() (a, b, c, d, e *mat.Dense) {
	a = newDense(p.totalStates, p.totalStates)
	b = newDense(p.totalStates, p.totalInputs)
	c = newDense(p.totalOutputs, p.totalStates)
	d = newDense(p.totalOutputs, p.totalInputs)
	e = newDense(p.totalStates, p.totalStates)
	eRaw := e.RawMatrix()
	for i, component := range p.components {
		n, _, _ := component.System.Dims()
		stateOffset := p.stateOffsets[i]
		inputOffset := p.inputOffsets[i]
		outputOffset := p.outputOffsets[i]
		setBlock(a, stateOffset, stateOffset, component.System.A)
		setBlock(b, stateOffset, inputOffset, component.System.B)
		setBlock(c, outputOffset, stateOffset, component.System.C)
		setBlock(d, outputOffset, inputOffset, component.System.D)
		if component.System.E == nil {
			for state := range n {
				eRaw.Data[(stateOffset+state)*eRaw.Stride+stateOffset+state] = 1
			}
		} else {
			setBlock(e, stateOffset, stateOffset, component.System.E)
		}
	}
	return a, b, c, d, e
}

func (p *physicalAssemblyPlan) appendComponents() (*System, error) {
	assembled := p.components[0].System.Copy()
	prefixSystemMetadata(assembled, p.components[0].Name)
	for _, component := range p.components[1:] {
		next := component.System.Copy()
		prefixSystemMetadata(next, component.Name)
		var err error
		assembled, err = Append(assembled, next)
		if err != nil {
			return nil, err
		}
	}
	assembled.Notes = p.name
	return assembled, nil
}

func (p *physicalAssemblyPlan) aggregateInputNames() []string {
	names := make([]string, 0, p.totalInputs)
	for _, component := range p.components {
		_, m, _ := component.System.Dims()
		names = append(names, physicalSignalNames(component.System.InputName, m, component.Name, "input")...)
	}
	return names
}

func (p *physicalAssemblyPlan) aggregateOutputNames() []string {
	names := make([]string, 0, p.totalOutputs)
	for _, component := range p.components {
		_, _, outputs := component.System.Dims()
		names = append(names, physicalSignalNames(component.System.OutputName, outputs, component.Name, "output")...)
	}
	return names
}

func (p *physicalAssemblyPlan) aggregateStateNames() []string {
	names := make([]string, 0, p.totalStates)
	for _, component := range p.components {
		n, _, _ := component.System.Dims()
		names = append(names, physicalSignalNames(component.System.StateName, n, component.Name, "state")...)
	}
	return names
}

func (p *physicalAssemblyPlan) internalStateNames() []string {
	names := make([]string, 0, len(p.internalInputs))
	for _, group := range p.groups {
		for _, portIndex := range group.ports {
			binding := p.ports[portIndex]
			for coordinate := range binding.port.Dimension {
				names = append(names, fmt.Sprintf("%s.%s.%s.through[%d]", p.name, p.components[binding.component].Name, binding.port.Name, coordinate))
			}
		}
	}
	return names
}

func copyPhysicalPorts(ports []PhysicalPort) []PhysicalPort {
	out := make([]PhysicalPort, len(ports))
	for i, port := range ports {
		out[i] = port
		out[i].Input = copyIntSlice(port.Input)
		out[i].Output = copyIntSlice(port.Output)
	}
	return out
}

func physicalPortIndex(index map[string]int, component, port string) (int, error) {
	key := component + "." + port
	value, ok := index[key]
	if !ok {
		return 0, fmt.Errorf("port %q not found: %w", key, ErrSignalNotFound)
	}
	return value, nil
}

func normalizedPhysicalEdge(a, b string) string {
	if a > b {
		a, b = b, a
	}
	return a + "<->" + b
}

func findPhysicalPort(parent []int, value int) int {
	for parent[value] != value {
		parent[value] = parent[parent[value]]
		value = parent[value]
	}
	return value
}

func unionPhysicalPorts(parent []int, a, b int) {
	aRoot := findPhysicalPort(parent, a)
	bRoot := findPhysicalPort(parent, b)
	if aRoot != bRoot {
		if aRoot > bRoot {
			aRoot, bRoot = bRoot, aRoot
		}
		parent[bRoot] = aRoot
	}
}

func complementChannels(total int, internal []int) []int {
	used := make([]bool, total)
	for _, channel := range internal {
		used[channel] = true
	}
	external := make([]int, 0, total-len(internal))
	for channel := range total {
		if !used[channel] {
			external = append(external, channel)
		}
	}
	return external
}

func selectedPhysicalNames(names []string, channels []int) []string {
	selected := make([]string, len(channels))
	for i, channel := range channels {
		selected[i] = names[channel]
	}
	return selected
}

func physicalSignalNames(names []string, count int, prefix, kind string) []string {
	out := make([]string, count)
	for i := range count {
		name := fmt.Sprintf("%s%d", kind, i+1)
		if i < len(names) && names[i] != "" {
			name = names[i]
		}
		out[i] = prefix + "." + name
	}
	return out
}

func reservePhysicalChannels(used []bool, channels []int, signal, key string) error {
	for _, channel := range channels {
		if channel < 0 || channel >= len(used) || used[channel] {
			return fmt.Errorf("invalid or reused %s channel %d on %q: %w", signal, channel, key, ErrInvalidArgument)
		}
		used[channel] = true
	}
	return nil
}

func claimUnusedPhysicalChannels(used []bool, count int) []int {
	channels := make([]int, 0, count)
	for channel, reserved := range used {
		if reserved {
			continue
		}
		used[channel] = true
		channels = append(channels, channel)
		if len(channels) == count {
			break
		}
	}
	return channels
}

func prefixSystemMetadata(sys *System, prefix string) {
	for i, name := range sys.InputName {
		sys.InputName[i] = prefix + "." + name
	}
	for i, name := range sys.OutputName {
		sys.OutputName[i] = prefix + "." + name
	}
	for i, name := range sys.StateName {
		sys.StateName[i] = prefix + "." + name
	}
}
