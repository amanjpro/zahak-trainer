package main

import (
	"flag"
	"fmt"
)

var (
	DefaultNumberOfEpochs        = 100
	DefaultNumberOfInputs        = 100
	DefaultNumberOfHiddenLayers  = 100
	DefaultNumberOfHiddenNeurons = 100
	DefaultNumberOfOutputs       = 100
)

func main() {
	epochs := flag.Int("epochs", DefaultNumberOfEpochs, "Number of epochs")
	inputs := flag.Int("inputs", DefaultNumberOfInputs, "Number of inputs")
	layers := flag.Int("layers", DefaultNumberOfHiddenLayers, "Number of hidden layers")
	neurons := flag.Int("neurons", DefaultNumberOfHiddenNeurons, "Number of hidden neurons")
	outputs := flag.Int("outputs", DefaultNumberOfOutputs, "Number of outputs")

	checkpointPath := flag.String("from-checkpoint", "", "Start from a known checkpoint")
	saveCheckpointPath := flag.String("to-checkpoint", "", "Save checkpoints to")
	binPath := flag.String("nnue-path", "", "Final NNUE path")
	epdPath := flag.String("input-path", "", "Path to input dataset")

	flag.Parse()

	topology := NewTopology(uint16(*inputs), uint16(*outputs), uint16(*layers), uint16(*neurons))
	network := CreateNetwork(topology)
	network.Save("/tmp/net.nnue")
	network = Load("/tmp/net.nnue")

	if *checkpointPath != "" {
		network = Load(*checkpointPath)
	}

	fmt.Println("Options:", *epochs, *saveCheckpointPath, *binPath, *epdPath)
	if false {
		network.Save(*saveCheckpointPath)
	}
}
