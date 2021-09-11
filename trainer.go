package main

type (
	Trainer struct {
		Net    Network
		Epochs int
	}
)

// TODO: Implement me
func (t *Trainer) getSample() Sample {
	return Sample{}
}

func (t *Trainer) Train() {
	sample := t.getSample()
	// Study
	t.Net.ForwardPropagate(sample.Input)
	// Teach
	errors := t.Net.FindErrors(sample.EvalTarget, sample.WDLTarget)
	// Learn
	t.Net.BackPropagate(errors)
}
