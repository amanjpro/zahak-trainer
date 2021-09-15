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
		Net        Network
		Training   *[]Data
		Validation *[]Data
		Epochs     int
	}
)

var (
	SigmoidScale float32 = 2.5 / 1024
	LearningRate float32 = 0.01
	BatchSize            = 16384
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func NewTrainer(net Network, dataset *[]Data, epochs int) *Trainer {
	upperEnd := 80 * len(*dataset) / 100
	training := (*dataset)[:upperEnd]
	validation := (*dataset)[upperEnd:]
	return &Trainer{
		Net:        net,
		Training:   &training,
		Validation: &validation,
		Epochs:     epochs,
	}
}

func (t *Trainer) PrintCost() {
	totalCost := float32(0)
	for i := 0; i < len(*t.Validation); i++ {
		data := (*t.Training)[i]
		predicted := t.Net.Predict(data.Input)
		totalCost += CalculateCostGradient(predicted, data.Score, data.Outcome) * SigmoidPrime(predicted)
	}
	fmt.Printf("Current cost is: %f\n", totalCost/float32(len(*t.Validation)))
}

func (t *Trainer) Train(path string) {
	// inputs := t.Net.Topology.Inputs
	for epoch := 0; epoch < t.Epochs; epoch++ {
		// sample := t.getSample()
		startTime := time.Now()
		fmt.Printf("Started Epoch %d at %s\n", epoch, startTime.String())
		fmt.Printf("Number of samples: %d\n", len(*t.Training))
		// totalValidation := float32(0)
		trainedSamplesInBatch := 0
		for i := 0; i < len(*t.Training); i++ {
			if trainedSamplesInBatch == BatchSize {
				trainedSamplesInBatch = 0
				t.Net.ApplyGradients()
			}
			trainedSamplesInBatch += 1
			data := (*t.Training)[i]
			t.Net.Train(data.Input, data.Score, data.Outcome)
			if i%4048 == 0 {
				speed := float64(i) / time.Since(startTime).Seconds()
				fmt.Printf("\rTrained on %d samples [ %f samples / second ]", i, speed)
			}
		}
		fmt.Printf("Finished Epoch %d at %s, elapsed time %s\n", epoch, time.Now().String(), time.Since(startTime).String())
		fmt.Printf("Storing This Epoch %d network\n", epoch)
		t.Net.Save(fmt.Sprintf("%s%cepoch-%d.nnue", path, os.PathSeparator, epoch))
		fmt.Printf("Stored This Epoch %d's network\n", epoch)
		t.PrintCost()
	}
}
