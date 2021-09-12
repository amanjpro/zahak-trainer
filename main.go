package main

import (
	"flag"
	"math/rand"
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
	learningRate := flag.Float64("lr", float64(LearningRate), "Learning Rate")
	sigmoidScale := flag.Float64("sigmoid-scale", float64(SigmoidScale), "Sigmoid scale")
	networkId := flag.Int("network-id", int(uint32(rand.Int())), "A unique id for the network")
	epdPath := flag.String("input-path", "", "Path to input dataset (FENs)")
	startNet := flag.String("from-net", "", "Path to a network, to be used as a starting point")
	binPath := flag.String("output-path", "", "Final NNUE path directory")

	flag.Parse()

	words := strings.Split(*neurons, ",")
	if len(words) < 1 {
		panic("At least one layer of hidden neurons are required")
	}
	hiddenNeurons := make([]uint32, len(words))
	for i, w := range words {
		parsed, err := strconv.Atoi(w)
		if err != nil {
			panic(err)
		}
		hiddenNeurons[i] = uint32(parsed)
	}

	var network Network
	if *startNet != "" {
		network = Load(*startNet)
	} else {
		topology := NewTopology(uint32(*inputs), uint32(*outputs), hiddenNeurons)
		network = CreateNetwork(topology, uint32(*networkId))
	}

	SigmoidScale = float32(*sigmoidScale)
	LearningRate = float32(*learningRate)

	trainer := NewTrainer(network, LoadDataset(*epdPath), *epochs)

	trainer.Train(*binPath)
}
