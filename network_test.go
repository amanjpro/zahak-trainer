package main

import (
	"fmt"
	"testing"
)

func createNetwork() *Network {
	top := NewTopology(8, 1, []uint32{4, 2})
	net := CreateNetwork(top, 30)

	// For predictability set all weights and biases to 1
	for i := 0; i < len(net.Activations); i++ {
		bb := net.Biases[i]
		for j := uint32(0); j < bb.Size(); j++ {
			bb.Data[j] = 1
		}

		ww := net.Weights[i]
		for j := uint32(0); j < ww.Size(); j++ {
			ww.Data[j] = 1
		}
	}

	return &net
}

func TestPredict(t *testing.T) {
	net := createNetwork()

	net.Predict([]int16{0, 1, 2, 3, 4, 5, 6, 7})

	activations := [][]float32{
		{9, 9, 9, 9},
		{37, 37},
		{Sigmoid(75)},
	}

	for i := 0; i < len(net.Activations); i++ {
		expected := activations[i]
		actual := net.Activations[i]
		if !sameArray(expected, actual.Data) {
			t.Errorf(fmt.Sprintf("Got %v, Expected %v", actual.Data, expected))
		}
	}
}

func TestFindErrors(t *testing.T) {
	net := createNetwork()

	net.Predict([]int16{0, 1, 2, 3})
	net.FindErrors(0.5)

	errors := [][]float32{
		{1, 1, 1, 1},
		{0.5, 0.5},
		{0.5},
	}

	for i := 0; i < len(net.Activations); i++ {
		expected := errors[i]
		actual := net.Errors[i]
		if !sameArray(expected, actual.Data) {
			t.Errorf(fmt.Sprintf("Got %v, Expected %v", actual.Data, expected))
		}
	}
}

func TestBinaryReaderWriter(t *testing.T) {
	top := NewTopology(10, 11, []uint32{12, 13, 14, 15, 16})
	net1 := CreateNetwork(top, 30)

	net1.Save("/tmp/net.nnue")
	net2 := Load("/tmp/net.nnue")

	if !sameTopology(net1.Topology, net2.Topology) {
		t.Errorf("Topology was read incorrectly")
	}

	if net1.Id != net2.Id {
		t.Errorf("Network Id was read incorrectly")
	}

	for i := 0; i < len(top.HiddenNeurons); i++ {
		data1 := net1.Weights[i].Data
		data2 := net2.Weights[i].Data
		if len(data1) != len(data2) {
			t.Errorf(fmt.Sprintf("Data length is mismatched: expected %d, got %d", len(data1), len(data2)))
		}
		for j := 0; j < len(data1); j++ {
			if data1[j] != data2[j] {
				t.Errorf(fmt.Sprintf("Hidden Weights was read incorrectly, %f instead of %f", data2[j], data1[j]))
			}
		}
	}

	for i := 0; i < len(top.HiddenNeurons); i++ {
		data1 := net1.Biases[i].Data
		data2 := net2.Biases[i].Data
		if len(data1) != len(data2) {
			t.Errorf(fmt.Sprintf("Data length is mismatched: expected %d, got %d", len(data1), len(data2)))
		}
		for j := 0; j < len(data1); j++ {
			if data1[j] != data2[j] {
				t.Errorf(fmt.Sprintf("Hidden Biases was read incorrectly, %f instead of %f", data2[j], data1[j]))
			}
		}
	}

}

func sameTopology(top1, top2 Topology) bool {
	if top1.Inputs != top2.Inputs {
		return false
	}

	if top1.Outputs != top2.Outputs {
		return false
	}

	if len(top1.HiddenNeurons) != len(top2.HiddenNeurons) {
		return false
	}

	for i := 0; i < len(top1.HiddenNeurons); i++ {
		if top1.HiddenNeurons[i] != top2.HiddenNeurons[i] {
			return false
		}
	}
	return true
}

func sameArray(expected, actual []float32) bool {
	if len(expected) != len(actual) {
		return false
	}
	for i := 0; i < len(expected); i++ {
		if expected[i] != actual[i] {
			return false
		}
	}
	return true
}
