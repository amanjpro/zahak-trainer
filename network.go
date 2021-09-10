package main

import (
	"encoding/binary"
	"io"
	"math"
	"os"

	"gonum.org/v1/gonum/stat/distuv"
)

// Network is a neural network with 3 layers
type (
	Topology struct {
		Inputs        uint16
		Outputs       uint16
		HiddenNeurons []uint16
	}

	Network struct {
		Topology Topology
		Neurons  [][]float32
		Weights  [][]float32
		Biases   [][]float32
	}
)

func NewTopology(inputs, outputs uint16, hiddenNeurons []uint16) Topology {
	return Topology{
		Inputs:        inputs,
		Outputs:       outputs,
		HiddenNeurons: hiddenNeurons,
	}
}

// CreateNetwork creates a neural network with random weights
func CreateNetwork(topology Topology) (net Network) {
	net = Network{
		Topology: topology,
	}

	net.Neurons = make([][]float32, len(topology.HiddenNeurons))
	net.Weights = make([][]float32, len(topology.HiddenNeurons))
	net.Biases = make([][]float32, len(topology.HiddenNeurons))

	cols := int(topology.Inputs)
	for i := 0; i < len(topology.HiddenNeurons); i++ {
		rows := int(topology.HiddenNeurons[i])
		net.Neurons[i] = randomArray(cols*rows, float32(topology.Inputs))
		net.Weights[i] = randomArray(cols*rows, float32(topology.Inputs))
		net.Biases[i] = randomArray(cols*rows, float32(topology.Inputs))
		cols = rows
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
// - All the matrices are written in column-major
// - The magic numbers:
//   - 66 (which is the ASCII code for B), uint8
//   - 90 (which is the ASCII code for Z), uint8
// - 1 The current version number, uint8
// - uint16 number to represent the number of inputs (let's call it I)
// - uint16 number to represent the number of hidden layers (let's call it L)
// - L uint16 to each the size of each hidden layer (let's call it NN)
// - uint16 number to represent the number of outputs
// - Followed float32 numbers, which represent all the hidden weights, stored one after the other
// - Followed float32 numbers, which represent all the hidden biases, stored one after the other
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
	buf = make([]byte, 6+2*len(n.Topology.HiddenNeurons))
	binary.BigEndian.PutUint16(buf[0:], n.Topology.Inputs)
	binary.BigEndian.PutUint16(buf[2:], uint16(len(n.Topology.HiddenNeurons)))
	i := 0
	for ; i < len(n.Topology.HiddenNeurons); i++ {
		binary.BigEndian.PutUint16(buf[4+2*i:], n.Topology.HiddenNeurons[i])
	}
	binary.BigEndian.PutUint16(buf[4+2*i:], n.Topology.Outputs)
	_, err = f.Write(buf)
	if err != nil {
		panic(err)
	}

	buf = make([]byte, 4)
	for i := 0; i < len(n.Topology.HiddenNeurons); i++ {
		weights := n.Weights[i]
		for j := 0; j < len(weights); j++ {
			binary.BigEndian.PutUint32(buf, math.Float32bits(weights[j]))
			_, err := f.Write(buf)
			if err != nil {
				panic(err)
			}
		}
	}

	for i := 0; i < len(n.Topology.HiddenNeurons); i++ {
		biases := n.Biases[i]
		for j := 0; j < len(biases); j++ {
			binary.BigEndian.PutUint32(buf, math.Float32bits(biases[j]))
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
	buf = make([]byte, 4)
	_, err = io.ReadFull(f, buf)
	if err != nil {
		panic(err)
	}
	inputs := binary.BigEndian.Uint16(buf[:2])
	layers := binary.BigEndian.Uint16(buf[2:4])

	buf = make([]byte, 2+2*layers)
	_, err = io.ReadFull(f, buf)
	if err != nil {
		panic(err)
	}
	neurons := make([]uint16, layers)
	i := uint16(0)
	for ; i < layers; i++ {
		neurons[i] = binary.BigEndian.Uint16(buf[i*2 : (i+1)*2])
	}
	outputs := binary.BigEndian.Uint16(buf[i*2:])

	topology := NewTopology(inputs, outputs, neurons)

	net := Network{
		Topology: topology,
	}

	net.Neurons = make([][]float32, len(topology.HiddenNeurons))
	net.Weights = make([][]float32, len(topology.HiddenNeurons))
	net.Biases = make([][]float32, len(topology.HiddenNeurons))

	buf = make([]byte, 4)
	cols := int(topology.Inputs)
	for i := 0; i < len(topology.HiddenNeurons); i++ {
		rows := int(neurons[i])
		data := make([]float32, rows*cols)
		for j := 0; j < len(data); j++ {
			_, err := io.ReadFull(f, buf)
			if err != nil {
				panic(err)
			}
			data[j] = math.Float32frombits(binary.BigEndian.Uint32(buf))
		}
		net.Neurons[i] = randomArray(rows*cols, float32(topology.Inputs))
		net.Weights[i] = data
		cols = rows
	}

	cols = int(topology.Inputs)
	for i := 0; i < len(topology.HiddenNeurons); i++ {
		rows := int(neurons[i])
		data := make([]float32, rows*cols)
		for j := 0; j < len(data); j++ {
			_, err := io.ReadFull(f, buf)
			if err != nil {
				panic(err)
			}
			data[j] = math.Float32frombits(binary.BigEndian.Uint32(buf))
		}
		net.Biases[i] = data
		cols = rows
	}
	return net
}

// Helper functions
// randomly generate a float64 array
func randomArray(size int, v float32) (data []float32) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(float64(v)),
		Max: 1 / math.Sqrt(float64(v)),
	}

	data = make([]float32, size)
	for i := 0; i < size; i++ {
		// data[i] = rand.NormFloat64() * math.Pow(v, -0.5)
		data[i] = float32(dist.Rand())
	}
	return
}
