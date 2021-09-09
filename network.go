package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// Network is a neural network with 3 layers
type (
	Topology struct {
		Inputs        int
		Outputs       int
		HiddenLayers  int
		HiddenNeurons int
	}

	Network struct {
		Topology Topology
		Neurons  []*mat.Dense
		Weights  []*mat.Dense
		Biases   []*mat.Dense
	}
)

func NewTopology(inputs, outputs, hiddenLayers, hiddenNeurons int) Topology {
	return Topology{
		Inputs:        inputs,
		Outputs:       outputs,
		HiddenLayers:  hiddenLayers,
		HiddenNeurons: hiddenNeurons,
	}
}

// CreateNetwork creates a neural network with random weights
func CreateNetwork(topology Topology) (net Network) {
	net = Network{
		Topology: topology,
	}

	net.Neurons = make([]*mat.Dense, topology.HiddenLayers)
	net.Weights = make([]*mat.Dense, topology.HiddenLayers)
	net.Biases = make([]*mat.Dense, topology.HiddenLayers)

	for i := 0; i < topology.HiddenLayers; i++ {
		net.Neurons[i] = mat.NewDense(topology.HiddenNeurons, topology.Inputs, randomArray(topology.Inputs*topology.HiddenNeurons, float64(topology.Inputs)))
		net.Weights[i] = mat.NewDense(topology.HiddenNeurons, topology.Inputs, randomArray(topology.Inputs*topology.HiddenNeurons, float64(topology.Inputs)))
		net.Biases[i] = mat.NewDense(topology.HiddenNeurons, topology.Inputs, randomArray(topology.Inputs*topology.HiddenNeurons, float64(topology.Inputs)))
	}

	return
}

func (n *Network) SaveCheckpoint(path string) {
	err := os.MkdirAll(path, os.ModePerm)
	if err != nil {
		panic(err)
	}

	m, err := os.Create(fmt.Sprintf("%s/topology.json", path))
	if err != nil {
		panic(err)
	}
	defer m.Close()
	js, err := json.Marshal(n.Topology)
	if err != nil {
		panic(err)
	}
	m.WriteString(string(js))

	for i := 0; i < n.Topology.HiddenLayers; i++ {
		h, err := os.Create(fmt.Sprintf("%s/hidden-%d.neurons", path, i))
		if err != nil {
			panic(err)
		}
		defer h.Close()
		n.Neurons[i].MarshalBinaryTo(h)

		w, err := os.Create(fmt.Sprintf("%s/hidden-%d.weights", path, i))
		if err != nil {
			panic(err)
		}
		defer w.Close()
		n.Weights[i].MarshalBinaryTo(w)

		b, err := os.Create(fmt.Sprintf("%s/hidden-%d.biases", path, i))
		if err != nil {
			panic(err)
		}
		defer b.Close()
		n.Biases[i].MarshalBinaryTo(b)
	}
}

// load a neural network from file
func LoadCheckpoint(path string) Network {
	js, err := ioutil.ReadFile(fmt.Sprintf("%s/topology.json", path))
	if err != nil {
		panic(err)
	}
	var topology Topology
	err = json.Unmarshal([]byte(js), &topology)
	if err != nil {
		panic(err)
	}

	net := CreateNetwork(topology)

	for i := 0; i < topology.HiddenLayers; i++ {
		h, err := os.Open(fmt.Sprintf("%s/hidden-%d.neurons", path, i))
		if err != nil {
			panic(err)
		}

		defer h.Close()
		w, err := os.Open(fmt.Sprintf("%s/hidden-%d.weights", path, i))
		if err != nil {
			panic(err)
		}
		defer w.Close()

		b, err := os.Open(fmt.Sprintf("%s/hidden-%d.biases", path, i))
		if err != nil {
			panic(err)
		}
		defer b.Close()

		net.Neurons[i].Reset()
		net.Neurons[i].UnmarshalBinaryFrom(h)

		net.Weights[i].Reset()
		net.Weights[i].UnmarshalBinaryFrom(w)

		net.Biases[i].Reset()
		net.Biases[i].UnmarshalBinaryFrom(b)
	}

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
