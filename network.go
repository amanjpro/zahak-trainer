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
		Weights     []Matrix
		Biases      []Matrix
		Activations []Matrix
		Errors      []Matrix
		WGradients  []Gradients
		BGradients  []Gradients
	}
)

func NewTopology(inputs, outputs uint32, hiddenNeurons []uint32) Topology {
	return Topology{
		Inputs:        inputs,
		Outputs:       outputs,
		HiddenNeurons: hiddenNeurons,
	}
}

func (n *Network) Copy() *Network {
	net := Network{
		Id:       n.Id,
		Topology: n.Topology,
		Weights:  n.Weights,
		Biases:   n.Biases,
	}
	topology := n.Topology
	inputSize := topology.Inputs
	net.Activations = make([]Matrix, len(topology.HiddenNeurons)+1)
	net.Errors = make([]Matrix, len(topology.HiddenNeurons)+1)
	net.WGradients = make([]Gradients, len(topology.HiddenNeurons)+1)
	net.BGradients = make([]Gradients, len(topology.HiddenNeurons)+1)

	for i := 0; i < len(net.Activations); i++ {
		var outputSize uint32
		if i == len(topology.HiddenNeurons) {
			outputSize = topology.Outputs
		} else {
			outputSize = topology.HiddenNeurons[i]
		}
		net.Activations[i] = SingletonMatrix(outputSize, randomArray(outputSize, float32(topology.Inputs)))
		net.Errors[i] = SingletonMatrix(outputSize, randomArray(outputSize, float32(topology.Inputs)))
		net.WGradients[i] = NewGradients(outputSize, inputSize)
		net.BGradients[i] = NewGradients(outputSize, 1)
		inputSize = outputSize
	}

	return &net
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
	net.Errors = make([]Matrix, len(topology.HiddenNeurons)+1)
	net.WGradients = make([]Gradients, len(topology.HiddenNeurons)+1)
	net.BGradients = make([]Gradients, len(topology.HiddenNeurons)+1)

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
		net.Biases[i] = SingletonMatrix(outputSize, randomArray(outputSize, float32(topology.Inputs)))
		net.WGradients[i] = NewGradients(outputSize, inputSize)
		net.BGradients[i] = NewGradients(outputSize, 1)
		net.Activations[i] = SingletonMatrix(outputSize, randomArray(outputSize, float32(topology.Inputs)))
		net.Errors[i] = SingletonMatrix(outputSize, randomArray(outputSize, float32(topology.Inputs)))
		inputSize = outputSize
	}
	return
}

// Binary specification for the NNUE file:
// - All the data is stored in little-endian layout
// - All the matrices are written in column-major
// - The magic number/version consists of 4 bytes (int32):
//   - 66 (which is the ASCII code for B), uint8
//   - 90 (which is the ASCII code for Z), uint8
//   - 2 The major part of the current version number, uint8
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
	buf := []byte{66, 90, 2, 0}
	_, err = f.Write(buf)
	if err != nil {
		panic(err)
	}

	// Write network Id
	binary.LittleEndian.PutUint32(buf, n.Id)
	_, err = f.Write(buf)
	if err != nil {
		panic(err)
	}

	// Write Topology
	buf = make([]byte, 3*4+4*len(n.Topology.HiddenNeurons))
	binary.LittleEndian.PutUint32(buf[0:], n.Topology.Inputs)
	binary.LittleEndian.PutUint32(buf[4:], n.Topology.Outputs)
	binary.LittleEndian.PutUint32(buf[8:], uint32(len(n.Topology.HiddenNeurons)))
	for i := 0; i < len(n.Topology.HiddenNeurons); i++ {
		binary.LittleEndian.PutUint32(buf[12+4*i:], n.Topology.HiddenNeurons[i])
	}
	_, err = f.Write(buf)
	if err != nil {
		panic(err)
	}

	buf = make([]byte, 4)
	for i := 0; i < len(n.Activations); i++ {
		weights := n.Weights[i].Data
		for j := 0; j < len(weights); j++ {
			binary.LittleEndian.PutUint32(buf, math.Float32bits(weights[j]))
			_, err := f.Write(buf)
			if err != nil {
				panic(err)
			}
		}

		biases := n.Biases[i].Data
		for j := 0; j < len(biases); j++ {
			binary.LittleEndian.PutUint32(buf, math.Float32bits(biases[j]))
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
	if buf[0] != 66 || buf[1] != 90 {
		panic("Magic word does not match expected, exiting")
	}

	if buf[2] != 2 || buf[3] != 0 {
		panic("Network binary format is not supported")
	}

	_, err = io.ReadFull(f, buf)
	if err != nil {
		panic(err)
	}
	id := binary.LittleEndian.Uint32(buf)

	// Read Topology Header
	buf = make([]byte, 12)
	_, err = io.ReadFull(f, buf)
	if err != nil {
		panic(err)
	}
	inputs := binary.LittleEndian.Uint32(buf[:4])
	outputs := binary.LittleEndian.Uint32(buf[4:8])
	layers := binary.LittleEndian.Uint32(buf[8:])

	buf = make([]byte, 4*layers)
	_, err = io.ReadFull(f, buf)
	if err != nil {
		panic(err)
	}
	neurons := make([]uint32, layers)
	for i := uint32(0); i < layers; i++ {
		neurons[i] = binary.LittleEndian.Uint32(buf[i*4 : (i+1)*4])
	}

	topology := NewTopology(inputs, outputs, neurons)

	net := Network{
		Topology: topology,
		Id:       id,
	}

	net.Activations = make([]Matrix, len(topology.HiddenNeurons)+1)
	net.Weights = make([]Matrix, len(topology.HiddenNeurons)+1)
	net.Biases = make([]Matrix, len(topology.HiddenNeurons)+1)
	net.Errors = make([]Matrix, len(topology.HiddenNeurons)+1)
	net.WGradients = make([]Gradients, len(topology.HiddenNeurons)+1)
	net.BGradients = make([]Gradients, len(topology.HiddenNeurons)+1)

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
			data[j] = math.Float32frombits(binary.LittleEndian.Uint32(buf))
		}
		net.Weights[i] = NewMatrix(outputSize, inputSize, data)
		net.WGradients[i] = NewGradients(outputSize, inputSize)
		inputSize = outputSize

		data = make([]float32, outputSize)
		for j := 0; j < len(data); j++ {
			_, err := io.ReadFull(f, buf)
			if err != nil {
				panic(err)
			}
			data[j] = math.Float32frombits(binary.LittleEndian.Uint32(buf))
		}
		net.Biases[i] = SingletonMatrix(outputSize, data)
		net.Activations[i] = SingletonMatrix(outputSize, randomArray(outputSize, float32(topology.Inputs)))
		net.Errors[i] = SingletonMatrix(outputSize, randomArray(outputSize, float32(topology.Inputs)))
		net.BGradients[i] = NewGradients(outputSize, 1)
	}
	return net
}

func (n *Network) Predict(input []int16) float32 {

	// First layer needs special care
	activationFn := ReLu
	// apply input layer
	output := n.Activations[0]
	weight := n.Weights[0]
	bias := n.Biases[0]
	output.Reset()
	for _, i := range input {
		osize := output.Size()
		for j := uint32(0); j < osize; j++ {
			output.Data[j] += weight.Get(j, uint32(i))
		}
	}
	osize := output.Size()
	for j := uint32(0); j < osize; j++ {
		output.Data[j] = activationFn(output.Data[j] + bias.Data[j])
	}
	last := len(n.Activations) - 1

	for l := 1; l < len(n.Activations); l++ {
		input := n.Activations[l-1]
		output = n.Activations[l]
		weight := n.Weights[l]
		bias := n.Biases[l]
		if l == last {
			activationFn = Sigmoid
		}

		osize := output.Size()
		for i := uint32(0); i < osize; i++ {
			output.Data[i] = 0
			isize := input.Size()
			for j := uint32(0); j < isize; j++ {
				output.Data[i] += input.Data[j] * weight.Get(i, j)
			}

			output.Data[i] = activationFn(output.Data[i] + bias.Data[i])
		}
	}

	return output.Data[0] // This makes the assumption that the output layer is always of size 1
}

func (n *Network) FindErrors(outputGradient float32) {
	last := len(n.Activations) - 1
	n.Errors[last].Data[0] = outputGradient

	for l := last - 1; l >= 0; l-- {
		output := n.Activations[l]
		weight := n.Weights[l+1]
		outputError := n.Errors[l+1]
		inputError := n.Errors[l]

		isize := inputError.Size()
		for i := uint32(0); i < isize; i++ {
			inputError.Data[i] = 0
			osize := outputError.Size()
			for j := uint32(0); j < osize; j++ {
				inputError.Data[i] += outputError.Data[j] * weight.Get(j, i) * ReLuPrime(output.Data[i])
			}
		}
	}
}

func (n *Network) UpdateGradients(input []int16) {
	wGradients := n.WGradients[0]
	bGradients := n.BGradients[0]
	err := n.Errors[0]

	// First layer needs special care
	for _, i := range input {
		esize := err.Size()
		for j := uint32(0); j < esize; j++ {
			wGradients.Update(j, uint32(i), err.Data[j])
		}
	}

	esize := err.Size()
	for j := uint32(0); j < esize; j++ {
		bGradients.Data[j].Update(err.Data[j])
	}

	for l := 1; l < len(n.Activations); l++ {

		wGradients = n.WGradients[l]
		bGradients = n.BGradients[l]
		input := n.Activations[l-1]
		err = n.Errors[l]

		for i := uint32(0); i < wGradients.Rows; i++ {
			err := err.Data[i]
			bGradients.Update(i, 0, err)
			for j := uint32(0); j < wGradients.Cols; j++ {
				gradient := input.Data[j] * err
				wGradients.Update(i, j, gradient)
			}
		}
	}
}

func (n *Network) Train(input []int16, evalTarget, wdlTarget float32) float32 {

	// First use the net to predict the outcome of the input
	lastOutput := n.Predict(input)

	// Measure how well did we do
	outputGradient := CalculateCostGradient(lastOutput, evalTarget, wdlTarget) * SigmoidPrime(lastOutput)

	// Use the output gradients (errors really) to measure the inner errors
	n.FindErrors(outputGradient)

	// Now, find the necessary updates to the gradients
	n.UpdateGradients(input)

	return ValidationCost(lastOutput, evalTarget, wdlTarget)
}

func (n *Network) ApplyGradients() {
	for i := 0; i < len(n.Activations); i++ {
		n.BGradients[i].Apply(&n.Biases[i])
		n.WGradients[i].Apply(&n.Weights[i])
	}
}

// Helper functions
// randomly generate a float64 array
func randomArray(size uint32, v float32) (data []float32) {
	dist := distuv.Uniform{
		Min: 0.0 / math.Sqrt(float64(v)),
		Max: 2.0 / math.Sqrt(float64(v)),
	}

	data = make([]float32, size)
	for i := uint32(0); i < size; i++ {
		// data[i] = rand.NormFloat64() * math.Pow(v, -0.5)
		data[i] = float32(dist.Rand())
	}
	return
}
