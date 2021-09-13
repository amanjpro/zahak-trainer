package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"
)

type (
	Sample struct {
		Inputs []int
	}

	Trainer struct {
		Net     Network
		Dataset []*Data
		Epochs  int
	}
)

var (
	SigmoidScale float32 = 2.5 / 1024
	LearningRate float32 = 0.01
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func NewTrainer(net Network, dataset []*Data, epochs int) *Trainer {
	return &Trainer{
		Net:     net,
		Dataset: dataset,
		Epochs:  epochs,
	}
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
		startTime := time.Now()
		fmt.Printf("Started Epoch %d at %s\n", epoch, startTime.String())
		fmt.Printf("Number of samples: %d\n", len(sample.Inputs))
		totalCost := float32(0)
		totalValidation := float32(0)
		for _, index := range sample.Inputs {
			data := t.Dataset[index]
			input := t.Net.CreateInput(data.Input)
			totalCost += t.Net.Train(input, data.Score, data.Outcome)
			cost := CalculateCost(t.Net.Activations[len(t.Net.Activations)-1], data.Score, data.Outcome)
			totalValidation += cost.Data[0]
		}
		fmt.Printf("Finished Epoch %d at %s, elapsed time %s\n", epoch, time.Now().String(), time.Since(startTime).String())
		fmt.Printf("Storing This Epoch %d network\n", epoch)
		t.Net.Save(fmt.Sprintf("%s%cepoch-%d.nnue", path, os.PathSeparator, epoch))
		fmt.Printf("Stored This Epoch %d's network\n", epoch)
		fmt.Printf("Current cost is: %f\n", totalCost/float32(len(sample.Inputs)))
		fmt.Printf("Current validation is: %f\n", totalValidation/float32(len(sample.Inputs)))
	}
}
