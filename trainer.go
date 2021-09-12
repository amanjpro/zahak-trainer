package main

import (
	"fmt"
	"math/rand"
	"time"
)

type (
	Sample struct {
		Inputs []int
	}

	Trainer struct {
		Net     Network
		Dataset []Data
		Epochs  int
	}
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func (t *Trainer) getSample() Sample {
	sampleSize := len(t.Dataset) / t.Epochs
	var dummy struct{}
	seen := make(map[int]struct{}, sampleSize)
	sample := make([]int, sampleSize)
	chosen := 0
	for chosen < sampleSize {
		candidate := rand.Intn(len(t.Dataset))
		if _, ok := seen[candidate]; !ok {
			seen[candidate] = dummy
			sample[chosen] = candidate
			chosen += 1
		}
	}

	return Sample{
		Inputs: sample,
	}
}

func (t *Trainer) Train(path string) {
	for epoch := 0; epoch < t.Epochs; epoch++ {
		sample := t.getSample()
		fmt.Printf("Started Epoch %d at %s\n", epoch, time.Now().String())
		for _, index := range sample.Inputs {
			data := t.Dataset[index]
			// Study
			t.Net.ForwardPropagate(t.Net.CreateInput(data.Input))
			// Teach
			errors := t.Net.FindErrors(data.Score, data.Outcome)
			// Learn
			t.Net.BackPropagate(errors)
		}
		fmt.Printf("Finished Epoch %d at %s\n", epoch, time.Now().String())
		fmt.Printf("Storing This Epoch %d network\n", epoch)
		t.Net.Save(path)
		fmt.Printf("Stored This Epoch %d's network\n", epoch)
	}
}
