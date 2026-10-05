package controlsys

type signalMetadata struct {
	input  []string
	output []string
	state  []string
	notes  string
}

type frdMetadata struct {
	input  []string
	output []string
}

func metadataFromSystem(sys *System) signalMetadata {
	return signalMetadata{
		input:  sys.InputName,
		output: sys.OutputName,
		state:  sys.StateName,
		notes:  sys.Notes,
	}
}

func (md signalMetadata) applyAll(sys *System) {
	sys.InputName = copyStringSlice(md.input)
	sys.OutputName = copyStringSlice(md.output)
	sys.StateName = copyStringSlice(md.state)
	sys.Notes = md.notes
}

func (md signalMetadata) applyIO(sys *System) {
	sys.InputName = copyStringSlice(md.input)
	sys.OutputName = copyStringSlice(md.output)
}

func (md signalMetadata) applyAllOwned(sys *System) {
	sys.InputName = md.input
	sys.OutputName = md.output
	sys.StateName = md.state
	sys.Notes = md.notes
}

func (md signalMetadata) applyIOOwned(sys *System) {
	sys.InputName = md.input
	sys.OutputName = md.output
}

func (md signalMetadata) selectIO(inputs, outputs []int) signalMetadata {
	return signalMetadata{
		input:  selectStringSlice(md.input, inputs),
		output: selectStringSlice(md.output, outputs),
		state:  copyStringSlice(md.state),
		notes:  md.notes,
	}
}

func seriesMetadata(left, right *System, n1, n2 int) signalMetadata {
	return signalMetadata{
		input:  copyStringSlice(left.InputName),
		output: copyStringSlice(right.OutputName),
		state:  concatStringSlices([][]string{left.StateName, right.StateName}, []int{n1, n2}),
	}
}

func controllerMetadata(measurementNames, controlNames []string) signalMetadata {
	return signalMetadata{
		input:  measurementNames,
		output: controlNames,
	}
}

func parallelMetadata(left, right *System, n1, n2 int) signalMetadata {
	return signalMetadata{
		input:  copyStringSlice(left.InputName),
		output: copyStringSlice(left.OutputName),
		state:  concatStringSlices([][]string{left.StateName, right.StateName}, []int{n1, n2}),
	}
}

func lftVisibleMetadata(sys *System, inputs, outputs int) signalMetadata {
	md := signalMetadata{notes: sys.Notes}
	if len(sys.InputName) >= inputs {
		md.input = copyStringSlice(sys.InputName[:inputs])
	}
	if len(sys.OutputName) >= outputs {
		md.output = copyStringSlice(sys.OutputName[:outputs])
	}
	return md
}

func frdSeriesMetadata(left, right *FRD) frdMetadata {
	return frdMetadata{
		input:  copyStringSlice(left.InputName),
		output: copyStringSlice(right.OutputName),
	}
}

func frdParallelMetadata(left *FRD) frdMetadata {
	return frdMetadata{
		input:  copyStringSlice(left.InputName),
		output: copyStringSlice(left.OutputName),
	}
}

func frdFeedbackMetadata(plant *FRD) frdMetadata {
	return frdMetadata{
		input:  copyStringSlice(plant.InputName),
		output: copyStringSlice(plant.OutputName),
	}
}
