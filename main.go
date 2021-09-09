package main

import (
	"flag"
	"fmt"
	"strconv"
	"strings"
)

var (
	DefaultNumberOfEpochs        = 100
	DefaultNumberOfInputs        = 768
	DefaultNumberOfHiddenNeurons = "128"
	DefaultNumberOfOutputs       = 1
)

func main() {
	epochs := flag.Int("epochs", DefaultNumberOfEpochs, "Number of epochs")
	inputs := flag.Int("inputs", DefaultNumberOfInputs, "Number of inputs")
	neurons := flag.String("hiddens", DefaultNumberOfHiddenNeurons, "Number of hidden neurons, for multi-layer you can send comma separated numbers")
	outputs := flag.Int("outputs", DefaultNumberOfOutputs, "Number of outputs")

	checkpointPath := flag.String("from-checkpoint", "", "Start from a known checkpoint")
	saveCheckpointPath := flag.String("to-checkpoint", "", "Save checkpoints to")
	binPath := flag.String("nnue-path", "", "Final NNUE path")
	epdPath := flag.String("input-path", "", "Path to input dataset")

	flag.Parse()

	words := strings.Split(*neurons, ",")
	if len(words) < 1 {
		panic("At least one layer of hidden neurons are required")
	}
	hiddenNeurons := make([]uint16, len(words))
	for i, w := range words {
		parsed, err := strconv.Atoi(w)
		if err != nil {
			panic(err)
		}
		hiddenNeurons[i] = uint16(parsed)
	}

	topology := NewTopology(uint16(*inputs), uint16(*outputs), hiddenNeurons)
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
