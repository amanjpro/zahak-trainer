package main

import (
	"encoding/binary"
	"io"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// Network is a neural network with 3 layers
type (
	Topology struct {
		Inputs        uint16
		Outputs       uint16
		HiddenLayers  uint16
		HiddenNeurons uint16
	}

	Network struct {
		Topology Topology
		Neurons  []*mat.Dense
		Weights  []*mat.Dense
		Biases   []*mat.Dense
	}
)

func NewTopology(inputs, outputs, hiddenLayers, hiddenNeurons uint16) Topology {
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

	for i := uint16(0); i < topology.HiddenLayers; i++ {
		net.Neurons[i] = mat.NewDense(int(topology.HiddenNeurons), int(topology.Inputs), randomArray(topology.Inputs*topology.HiddenNeurons, float64(topology.Inputs)))
		net.Weights[i] = mat.NewDense(int(topology.HiddenNeurons), int(topology.Inputs), randomArray(topology.Inputs*topology.HiddenNeurons, float64(topology.Inputs)))
		net.Biases[i] = mat.NewDense(int(topology.HiddenNeurons), int(topology.Inputs), randomArray(topology.Inputs*topology.HiddenNeurons, float64(topology.Inputs)))
	}

	return
}

func (net *Network) CreateInputs(position *Position) []uint16 {
	input := make([]uint16, 0, 64)

	for j := 0; j < 64; j++ {
		sq := Square(j)
		piece := position.PieceAt(sq)
		if piece != NoPiece {
			index := uint16(piece)*64 + uint16(sq)
			input = append(input, index)
		}
	}

	return input
}

// Binary specification for the NNUE file:
// - All the data is stored in big-endian layout
// - All the matrices are written in row-major
// - The magic numbers:
//   - 66 (which is the ASCII code for B), uint8
//   - 90 (which is the ASCII code for Z), uint8
// - 1 The current version number, uint8
// - uint16 number to represent the number of inputs (let's call it I)
// - uint16 number to represent the number of hidden layers (let's call it L)
// - uint16 number to represent the number of neurons in each of the hidden layers (let's call it N)
// - uint16 number to represent the number of outputs
// - Followed by as L * (N * I) float64 numbers, each chunk of (N * I) will represent a hidden layer neurons.
// - Followed by as L * (N * I) float64 numbers, each chunk of (N * I) will represent a hidden layer weights.
// - Followed by as L * (N * I) float64 numbers, each chunk of (N * I) will represent a hidden layer biases.
func (n *Network) Save(file string) {
	f, err := os.Create(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// Write headers
	buf := []byte{66, 90, 1}
	_, err = f.Write(buf)
	if err != nil {
		panic(err)
	}

	// Write Topology
	buf = make([]byte, 8)
	binary.BigEndian.PutUint16(buf[0:], n.Topology.Inputs)
	binary.BigEndian.PutUint16(buf[2:], n.Topology.HiddenLayers)
	binary.BigEndian.PutUint16(buf[4:], n.Topology.HiddenNeurons)
	binary.BigEndian.PutUint16(buf[6:], n.Topology.Outputs)
	_, err = f.Write(buf)
	if err != nil {
		panic(err)
	}

	// Write hidden neurons
	for i := uint16(0); i < n.Topology.HiddenLayers; i++ {
		neurons := n.Neurons[i].RawMatrix().Data
		for j := 0; j < len(neurons); j++ {
			binary.BigEndian.PutUint64(buf, math.Float64bits(neurons[j]))
			_, err := f.Write(buf)
			if err != nil {
				panic(err)
			}
		}
	}

	for i := uint16(0); i < n.Topology.HiddenLayers; i++ {
		weights := n.Weights[i].RawMatrix().Data
		for j := 0; j < len(weights); j++ {
			binary.BigEndian.PutUint64(buf, math.Float64bits(weights[j]))
			_, err := f.Write(buf)
			if err != nil {
				panic(err)
			}
		}
	}

	for i := uint16(0); i < n.Topology.HiddenLayers; i++ {
		biases := n.Biases[i].RawMatrix().Data
		for j := 0; j < len(biases); j++ {
			binary.BigEndian.PutUint64(buf, math.Float64bits(biases[j]))
			_, err := f.Write(buf)
			if err != nil {
				panic(err)
			}
		}
	}
}

// load a neural network from file
func Load(path string) Network {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// Read headers
	buf := make([]byte, 3)
	_, err = io.ReadFull(f, buf)
	if err != nil {
		panic(err)
	}
	if buf[0] != 66 || buf[1] != 90 {
		panic("Magic word does not match expected")
	}
	if buf[2] != 1 {
		panic("Binary version is unsupported")
	}

	// Read Topology
	buf = make([]byte, 8)
	_, err = io.ReadFull(f, buf)
	if err != nil {
		panic(err)
	}
	inputs := binary.BigEndian.Uint16(buf[:2])
	layers := binary.BigEndian.Uint16(buf[2:4])
	neurons := binary.BigEndian.Uint16(buf[4:6])
	outputs := binary.BigEndian.Uint16(buf[6:])

	topology := NewTopology(inputs, outputs, layers, neurons)
	net := Network{
		Topology: topology,
	}

	net.Neurons = make([]*mat.Dense, topology.HiddenLayers)
	net.Weights = make([]*mat.Dense, topology.HiddenLayers)
	net.Biases = make([]*mat.Dense, topology.HiddenLayers)

	for i := uint16(0); i < topology.HiddenLayers; i++ {
		data := make([]float64, topology.Inputs*topology.HiddenNeurons)
		for j := 0; j < len(data); j++ {
			_, err := io.ReadFull(f, buf)
			if err != nil {
				panic(err)
			}
			data[j] = math.Float64frombits(binary.BigEndian.Uint64(buf))
		}
		net.Neurons[i] = mat.NewDense(int(topology.HiddenNeurons), int(topology.Inputs), data)
	}

	for i := uint16(0); i < topology.HiddenLayers; i++ {
		data := make([]float64, topology.Inputs*topology.HiddenNeurons)
		for j := 0; j < len(data); j++ {
			_, err := io.ReadFull(f, buf)
			if err != nil {
				panic(err)
			}
			data[j] = math.Float64frombits(binary.BigEndian.Uint64(buf))
		}
		net.Weights[i] = mat.NewDense(int(topology.HiddenNeurons), int(topology.Inputs), data)
	}

	for i := uint16(0); i < topology.HiddenLayers; i++ {
		data := make([]float64, topology.Inputs*topology.HiddenNeurons)
		for j := 0; j < len(data); j++ {
			_, err := io.ReadFull(f, buf)
			if err != nil {
				panic(err)
			}
			data[j] = math.Float64frombits(binary.BigEndian.Uint64(buf))
		}
		net.Biases[i] = mat.NewDense(int(topology.HiddenNeurons), int(topology.Inputs), data)
	}
	return net
}

// Helper functions
// randomly generate a float64 array
func randomArray(size uint16, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := uint16(0); i < size; i++ {
		// data[i] = rand.NormFloat64() * math.Pow(v, -0.5)
		data[i] = dist.Rand()
	}
	return
}
