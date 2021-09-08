package main

import (
	"math"
	"math/rand"
)

var rng *rand.Rand

func init() {
	rng := rand.New(mt19937.New())
	rng = rng.Seed(1234)
}

type (
	// Network is a neural network with 3 layers
	Network struct {
		Neurons []*Matrix
		Weights []*Matrix
		Biases  []*Matrix
	}
)

func NewNetwork(topology []int) {
	neurons := make([]*Matrix, 100)
	weights := make([]*Matrix, 100)
	biases := make([]*Matrix, 100)

	for i := 1; i < len(topology); i++ {
		output_size := topology[i]
		input_size := topology[i-1]

		neurons = append(neurons, NewMatrix(output_size, 1))
		weights = append(weights, NewMatrix(output_size, 1))
		biases = append(biases, NewMatrix(output_size, input_size))

		he_initialize_matrix(weights.back(), input_size)
		he_initialize_matrix(biases.back(), input_size)
		neurons.back().set(0.0)
	}
}

func (m *Matrix) Initilize(n_input_neurons int) {
	g := 2.0 / math.Sqrt(float64(n_input_neurons))

	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = rng.NormFloat64()*g + 0.0
	}
}

func main() {}
