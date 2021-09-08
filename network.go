package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// Network is a neural network with 3 layers
type Network struct {
	inputs        int
	hiddens       int
	outputs       int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	learningRate  float64
}

// CreateNetwork creates a neural network with random weights
func CreateNetwork(input, hidden, output int, rate float64) (net Network) {
	net = Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: rate,
	}
	net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, randomArray(net.inputs*net.hiddens, float64(net.inputs)))
	net.outputWeights = mat.NewDense(net.outputs, net.hiddens, randomArray(net.hiddens*net.outputs, float64(net.hiddens)))
	return
}

func (n *Network) SaveCheckpoint(path string) {
	err := os.MkdirAll(path, os.ModePerm)
	if err != nil {
		panic(err)
	}

	m, err := os.Create(fmt.Sprintf("%s/topology.txt", path))
	if err != nil {
		panic(err)
	}
	defer m.Close()
	m.WriteString(fmt.Sprintf("%d\n%d\n%d\n%f\n", n.inputs, n.hiddens, n.outputs, n.learningRate))

	h, err := os.Create(fmt.Sprintf("%s/hweights.model", path))
	if err != nil {
		panic(err)
	}
	defer h.Close()
	n.hiddenWeights.MarshalBinaryTo(h)

	o, err := os.Create(fmt.Sprintf("%s/oweights.model", path))
	if err != nil {
		panic(err)
	}
	defer o.Close()
	n.outputWeights.MarshalBinaryTo(o)
}

// load a neural network from file
func LoadCheckpoint(path string) Network {
	m, err := os.Open(fmt.Sprintf("%s/topology.txt", path))
	if err != nil {
		panic(err)
	}
	defer m.Close()
	scanner := bufio.NewScanner(m)
	scanner.Scan()
	inputs, err := strconv.Atoi(scanner.Text())
	if err != nil {
		panic(err)
	}
	scanner.Scan()
	hiddens, err := strconv.Atoi(scanner.Text())
	if err != nil {
		panic(err)
	}
	scanner.Scan()
	outputs, err := strconv.Atoi(scanner.Text())
	if err != nil {
		panic(err)
	}
	scanner.Scan()
	learningRate, err := strconv.ParseFloat(scanner.Text(), 64)
	if err != nil {
		panic(err)
	}

	net := CreateNetwork(inputs, hiddens, outputs, learningRate)

	h, err := os.Open(fmt.Sprintf("%s/hweights.model", path))
	if err != nil {
		panic(err)
	}
	defer h.Close()
	net.hiddenWeights.Reset()
	net.hiddenWeights.UnmarshalBinaryFrom(h)

	o, err := os.Open(fmt.Sprintf("%s/oweights.model", path))
	if err != nil {
		panic(err)
	}
	net.outputWeights.Reset()
	net.outputWeights.UnmarshalBinaryFrom(o)
	defer o.Close()

	return net
}

// Helper functions
// randomly generate a float64 array
func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		// data[i] = rand.NormFloat64() * math.Pow(v, -0.5)
		data[i] = dist.Rand()
	}
	return
}
