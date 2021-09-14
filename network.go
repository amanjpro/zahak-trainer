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
		Inputs        uint32
		Outputs       uint32
		HiddenNeurons []uint32
	}

	Network struct {
		Id          uint32
		Topology    Topology
		Activations []Matrix
		Weights     []Matrix
		Biases      []Matrix
	}
)

func NewTopology(inputs, outputs uint32, hiddenNeurons []uint32) Topology {
	return Topology{
		Inputs:        inputs,
		Outputs:       outputs,
		HiddenNeurons: hiddenNeurons,
	}
}

// CreateNetwork creates a neural network with random weights
func CreateNetwork(topology Topology, id uint32) (net Network) {
	net = Network{
		Topology: topology,
		Id:       id,
	}

	net.Activations = make([]Matrix, len(topology.HiddenNeurons)+1)
	net.Weights = make([]Matrix, len(topology.HiddenNeurons)+1)
	net.Biases = make([]Matrix, len(topology.HiddenNeurons)+1)

	inputSize := topology.Inputs
	i := 0
	for ; i < len(net.Activations); i++ {
		var outputSize uint32
		if i == len(topology.HiddenNeurons) {
			outputSize = topology.Outputs
		} else {
			outputSize = topology.HiddenNeurons[i]
		}
		net.Weights[i] = NewMatrix(outputSize, inputSize, randomArray(inputSize*outputSize, float32(topology.Inputs)))
		net.Activations[i] = SingletonMatrix(outputSize, randomArray(outputSize, float32(topology.Inputs)))
		net.Biases[i] = SingletonMatrix(outputSize, randomArray(outputSize, float32(topology.Inputs)))
		inputSize = outputSize
	}
	return
}

// Binary specification for the NNUE file:
// - All the data is stored in big-endian layout
// - All the matrices are written in column-major
// - The magic number/version consists of 4 bytes (int32):
//   - 66 (which is the ASCII code for B), uint8
//   - 90 (which is the ASCII code for Z), uint8
//   - 1 The major part of the current version number, uint8
//   - 0 The minor part of the current version number, uint8
// - 4 bytes (int32) to denote the network ID
// - 4 bytes (int32) to denote input size
// - 4 bytes (int32) to denote output size
// - 4 bytes (int32) number to represent the number of inputs
// - 4 bytes (int32) for the size of each layer
// - All weights for a layer, followed by all the biases of the same layer
// - Other layers follow just like the above point
func (n *Network) Save(file string) {
	f, err := os.Create(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// Write headers
	buf := []byte{66, 90, 1, 0}
	_, err = f.Write(buf)
	if err != nil {
		panic(err)
	}

	// Write network Id
	binary.BigEndian.PutUint32(buf, n.Id)
	_, err = f.Write(buf)
	if err != nil {
		panic(err)
	}

	// Write Topology
	buf = make([]byte, 3*4+4*len(n.Topology.HiddenNeurons))
	binary.BigEndian.PutUint32(buf[0:], n.Topology.Inputs)
	binary.BigEndian.PutUint32(buf[4:], n.Topology.Outputs)
	binary.BigEndian.PutUint32(buf[8:], uint32(len(n.Topology.HiddenNeurons)))
	for i := 0; i < len(n.Topology.HiddenNeurons); i++ {
		binary.BigEndian.PutUint32(buf[12+4*i:], n.Topology.HiddenNeurons[i])
	}
	_, err = f.Write(buf)
	if err != nil {
		panic(err)
	}

	buf = make([]byte, 4)
	for i := 0; i < len(n.Activations); i++ {
		weights := n.Weights[i].Data
		for j := 0; j < len(weights); j++ {
			binary.BigEndian.PutUint32(buf, math.Float32bits(weights[j]))
			_, err := f.Write(buf)
			if err != nil {
				panic(err)
			}
		}

		biases := n.Biases[i].Data
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
	buf := make([]byte, 4)
	_, err = io.ReadFull(f, buf)
	if err != nil {
		panic(err)
	}
	if buf[0] != 66 || buf[1] != 90 || buf[2] != 1 || buf[3] != 0 {
		panic("Magic word does not match expected, exiting")
	}

	_, err = io.ReadFull(f, buf)
	if err != nil {
		panic(err)
	}
	id := binary.BigEndian.Uint32(buf)

	// Read Topology Header
	buf = make([]byte, 12)
	_, err = io.ReadFull(f, buf)
	if err != nil {
		panic(err)
	}
	inputs := binary.BigEndian.Uint32(buf[:4])
	outputs := binary.BigEndian.Uint32(buf[4:8])
	layers := binary.BigEndian.Uint32(buf[8:])

	buf = make([]byte, 4*layers)
	_, err = io.ReadFull(f, buf)
	if err != nil {
		panic(err)
	}
	neurons := make([]uint32, layers)
	for i := uint32(0); i < layers; i++ {
		neurons[i] = binary.BigEndian.Uint32(buf[i*4 : (i+1)*4])
	}

	topology := NewTopology(inputs, outputs, neurons)

	net := Network{
		Topology: topology,
		Id:       id,
	}

	net.Activations = make([]Matrix, len(topology.HiddenNeurons)+1)
	net.Weights = make([]Matrix, len(topology.HiddenNeurons)+1)
	net.Biases = make([]Matrix, len(topology.HiddenNeurons)+1)

	buf = make([]byte, 4)
	inputSize := topology.Inputs
	for i := 0; i < len(net.Activations); i++ {
		var outputSize uint32
		if i == len(neurons) {
			outputSize = outputs
		} else {
			outputSize = neurons[i]
		}
		data := make([]float32, outputSize*inputSize)
		for j := 0; j < len(data); j++ {
			_, err := io.ReadFull(f, buf)
			if err != nil {
				panic(err)
			}
			data[j] = math.Float32frombits(binary.BigEndian.Uint32(buf))
		}
		net.Weights[i] = NewMatrix(outputSize, inputSize, data)
		inputSize = outputSize

		data = make([]float32, outputSize)
		for j := 0; j < len(data); j++ {
			_, err := io.ReadFull(f, buf)
			if err != nil {
				panic(err)
			}
			data[j] = math.Float32frombits(binary.BigEndian.Uint32(buf))
		}
		net.Biases[i] = SingletonMatrix(outputSize, data)
		net.Activations[i] = SingletonMatrix(outputSize, randomArray(outputSize, float32(topology.Inputs)))
	}
	return net
}

func (n *Network) Predict(input Matrix) {

	activations := input
	activationFn := ReLu
	last := len(n.Activations) - 1

	for i := 0; i < len(n.Activations); i++ {
		if i != 0 {
			activations = n.Activations[i-1]
		}
		if i == last {
			activationFn = Sigmoid
		}

		n.Activations[i].Dot(&n.Weights[i], &activations)
		n.Activations[i].Add(n.Activations[i], n.Biases[i])
		n.Activations[i].Apply(n.Activations[i], activationFn)
	}
}

func (n *Network) Train(input Matrix, evalTarget, wdlTarget float32) float32 {

	activations := input
	activationFn := ReLu
	last := len(n.Activations) - 1
	for i := 0; i < len(n.Activations); i++ {
		if i != 0 {
			activations = n.Activations[i-1]
		}
		if i == last {
			activationFn = Sigmoid
		}

		n.Activations[i].Dot(&n.Weights[i], &activations)
		n.Activations[i].Add(n.Activations[i], n.Biases[i])
		n.Activations[i].Apply(n.Activations[i], activationFn)
	}

	errors := make([]Matrix, len(n.Activations))
	cost := CalculateCost(n.Activations[last], evalTarget, wdlTarget)
	res := cost.Data[0]
	for i := last; i >= 0; i-- {
		var err Matrix
		if i == last {
			err = cost
		} else {
			transposed := n.Weights[i+1].T()
			err = Dot(transposed, &errors[i+1])
			err.Apply(err, ReLuPrime)
		}
		errors[i] = err
		if i == 0 {
			activations = input
		} else {
			activations = n.Activations[i-1]
		}
		gradient := Multiply(n.Activations[i], err)
		gradient.Scale(gradient, LearningRate)
		transposedInput := activations.T()
		whoDelta := Dot(&gradient, transposedInput)

		n.Weights[i].Add(n.Weights[i], whoDelta)
		n.Biases[i].Add(n.Biases[i], gradient)
	}

	return res
}

// Helper functions
// randomly generate a float64 array
func randomArray(size uint32, v float32) (data []float32) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(float64(v)),
		Max: 1 / math.Sqrt(float64(v)),
	}

	data = make([]float32, size)
	for i := uint32(0); i < size; i++ {
		// data[i] = rand.NormFloat64() * math.Pow(v, -0.5)
		data[i] = float32(dist.Rand())
	}
	return
}
