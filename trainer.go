package main

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"time"
)

type (
	Sample struct {
		Inputs []int
	}

	Trainer struct {
		Nets            []*Network
		Training        *[]Data
		Validation      *[]Data
		Epochs          int
		ValidationCosts []float32
		TrainingCosts   []float32
	}
)

var (
	SigmoidScale    float32 = 2.5 / 1024
	LearningRate    float32 = 0.01
	NumberOfThreads         = runtime.NumCPU()
	BatchSize               = 16384
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func NewTrainer(net Network, dataset *[]Data, epochs int) *Trainer {
	upperEnd := 80 * len(*dataset) / 100
	training := (*dataset)[:upperEnd]
	validation := (*dataset)[upperEnd:]
	networks := make([]*Network, NumberOfThreads)
	for i := 0; i < len(networks); i++ {
		networks[i] = net.Copy()
	}
	return &Trainer{
		Nets:            networks,
		Training:        &training,
		Validation:      &validation,
		Epochs:          epochs,
		TrainingCosts:   make([]float32, epochs),
		ValidationCosts: make([]float32, epochs),
	}
}

func (t *Trainer) CopyNets() {
	for i := 1; i < len(t.Nets); i++ {
		t.Nets[i] = t.Nets[0].Copy()
	}
}

func (t *Trainer) SyncGradients() {
	for i := 1; i < len(t.Nets); i++ {
		for j := 0; j < len(t.Nets[i].Activations); j++ {
			wgrad := t.Nets[i].WGradients[j]
			size := wgrad.Size()
			for k := uint32(0); k < size; k++ {
				t.Nets[0].WGradients[j].Data[k].Update(wgrad.Data[k].Value)
				t.Nets[i].WGradients[j].Data[k].Reset()
			}

			bgrad := t.Nets[i].BGradients[j]
			size = bgrad.Size()
			for k := uint32(0); k < size; k++ {
				t.Nets[0].BGradients[j].Data[k].Update(bgrad.Data[k].Value)
				t.Nets[i].BGradients[j].Data[k].Reset()
			}
		}
	}
}

func (t *Trainer) PrintCost() float32 {
	fmt.Printf("Starting the validation of the Epoch\n")
	totalCost := float32(0)

	batchSize := len(*t.Validation) / NumberOfThreads
	answer := make(chan float32)

	for i := 0; i < NumberOfThreads; i++ {
		batch := (*t.Validation)[i*batchSize : (i+1)*batchSize]
		go func(n *Network, batch []Data, answer chan float32) {
			localCost := float32(0)
			for d := 0; d < len(batch); d++ {
				data := batch[d]

				predicted := n.Predict(data.Input)
				cost := ValidationCost(predicted, data.Score, data.Outcome)

				localCost += cost
			}
			answer <- localCost
		}(t.Nets[i], batch, answer)
	}
	for i := 0; i < NumberOfThreads; i++ {
		totalCost += <-answer
	}
	averageCost := totalCost / float32(len(*t.Validation))
	fmt.Printf("Current validation cost is: %f\n", averageCost)
	return averageCost
}

func (t *Trainer) StartEpoch(startTime time.Time) float32 {
	batchEnd := BatchSize
	samples := 0
	totalCost := float32(0)
	for batchEnd < len(*t.Training) {
		newBatch := (*t.Training)[batchEnd-BatchSize : batchEnd]
		miniBatchSize := len(newBatch) / NumberOfThreads
		answers := make(chan float32)
		for i := 0; i < NumberOfThreads; i++ {
			smallBatch := newBatch[i*miniBatchSize : (i+1)*miniBatchSize]
			go func(main bool, n *Network, batch []Data, answer chan float32) {
				localCost := float32(0)
				for d := 0; d < len(batch); d++ {
					data := batch[d]
					localCost += n.Train(data.Input, data.Score, data.Outcome)
				}
				answer <- localCost
			}(i == 0, t.Nets[i], smallBatch, answers)
		}
		for i := 0; i < NumberOfThreads; i++ {
			totalCost += <-answers
			samples += miniBatchSize
		}
		speed := float64(samples) / time.Since(startTime).Seconds()
		fmt.Printf("\rTrained on %d samples [ %f samples / second ]", samples, speed)
		t.SyncGradients()
		t.Nets[0].ApplyGradients()
		t.CopyNets()
		batchEnd += BatchSize
	}

	return totalCost
}

func (t *Trainer) Train(path string) {
	for epoch := 0; epoch < t.Epochs; epoch++ {
		startTime := time.Now()
		fmt.Printf("Started Epoch %d at %s\n", epoch, startTime.String())
		fmt.Printf("Number of samples: %d\n", len(*t.Training))
		totalCost := t.StartEpoch(startTime)
		fmt.Printf("\nFinished Epoch %d at %s, elapsed time %s\n", epoch, time.Now().String(), time.Since(startTime).String())
		fmt.Printf("Storing This Epoch %d network\n", epoch+1)
		t.Nets[0].Save(fmt.Sprintf("%s%cepoch-%d.nnue", path, os.PathSeparator, epoch+1))
		fmt.Printf("Stored This Epoch %d's network\n", epoch+1)
		averageCost := totalCost / float32(len(*t.Training))
		t.ValidationCosts[epoch] = t.PrintCost()
		t.TrainingCosts[epoch] = averageCost
		fmt.Printf("Current training cost is: %f\n", averageCost)
		fmt.Println("Training and validation cost progression")
		fmt.Println("==============================================================================")
		fmt.Println("Epoch\t\t\t\tTraining Cost\t\t\t\tValidation Cost")
		for e := 0; e <= epoch; e++ {
			fmt.Printf("%d\t\t\t\t%f\t\t\t\t%f\n", e+1, t.TrainingCosts[e], t.ValidationCosts[e])
		}
		fmt.Println("==============================================================================")
	}

}
