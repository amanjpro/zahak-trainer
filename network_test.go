package main

import (
	"fmt"
	"math/rand"
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

func TestPredictPartialInput(t *testing.T) {
	net := createNetwork()

	net.Predict([]int16{0, 2, 3, 5, 6})

	activations := [][]float32{
		{6, 6, 6, 6},
		fill(2, 25),
		fill(1, Sigmoid(51)),
	}

	for i := 0; i < len(net.Activations); i++ {
		expected := activations[i]
		actual := net.Activations[i]
		if !sameArray(expected, actual.Data) {
			t.Errorf(fmt.Sprintf("Got %v, Expected %v", actual.Data, expected))
		}
	}
}

func TestPredict(t *testing.T) {
	net := createNetwork()

	net.Predict([]int16{0, 1, 2, 3, 4, 5, 6, 7})

	activations := [][]float32{
		fill(4, 9),
		fill(2, 37),
		fill(1, Sigmoid(75)),
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

	net.Predict([]int16{0, 1, 2, 3, 4, 5, 6, 7})
	net.FindErrors(0.5)

	errors := [][]float32{
		fill(4, 1),
		fill(2, 0.5),
		fill(1, 0.5),
	}

	for i := 0; i < len(net.Activations); i++ {
		expected := errors[i]
		actual := net.Errors[i]
		if !sameArray(expected, actual.Data) {
			t.Errorf(fmt.Sprintf("Got %v, Expected %v", actual.Data, expected))
		}
	}
}

func TestFindErrorsPartialInput(t *testing.T) {
	net := createNetwork()

	net.Predict([]int16{0, 2, 3, 5, 6})
	net.FindErrors(0.5)

	errors := [][]float32{
		fill(4, ReLuPrime(6)),
		fill(2, 0.5*ReLuPrime(25)),
		fill(1, 0.5),
	}

	for i := 0; i < len(net.Activations); i++ {
		expected := errors[i]
		actual := net.Errors[i]
		if !sameArray(expected, actual.Data) {
			t.Errorf(fmt.Sprintf("Got %v, Expected %v", actual.Data, expected))
		}
	}
}

func TestUpdateGradients(t *testing.T) {
	net := createNetwork()

	input := []int16{0, 1, 2, 3, 4, 5, 6, 7}
	net.Predict(input)
	net.FindErrors(0.5)
	net.UpdateGradients(input)

	wgrads := [][]float32{
		fill(32, 1),
		fill(8, 4.5),
		fill(2, 37*0.5),
	}

	for i := 0; i < len(net.Activations); i++ {
		expected := wgrads[i]
		actual := net.WGradients[i]
		if !sameArray(expected, actual.Values()) {
			t.Errorf(fmt.Sprintf("Got %v, Expected %v", actual.Values(), expected))
		}
	}

	bgrads := [][]float32{
		fill(4, 1),
		fill(2, 0.5),
		fill(1, 0.5),
	}
	for i := 0; i < len(net.Activations); i++ {
		expected := bgrads[i]
		actual := net.BGradients[i]
		if !sameArray(expected, actual.Values()) {
			t.Errorf(fmt.Sprintf("Got %v, Expected %v", actual.Values(), expected))
		}
	}
}

func TestGradientUpdatePartialInput(t *testing.T) {
	net := createNetwork()

	input := []int16{0, 2, 3, 5, 6}
	net.Predict(input)
	net.FindErrors(0.5)
	net.UpdateGradients(input)

	wgrads := [][]float32{
		{1, 1, 1, 1,
			0, 0, 0, 0,
			1, 1, 1, 1,
			1, 1, 1, 1,
			0, 0, 0, 0,
			1, 1, 1, 1,
			1, 1, 1, 1,
			0, 0, 0, 0},
		fill(8, 3),
		fill(2, 25*0.5),
	}

	for i := 0; i < len(net.Activations); i++ {
		expected := wgrads[i]
		actual := net.WGradients[i]
		if !sameArray(expected, actual.Values()) {
			t.Errorf(fmt.Sprintf("Got %v, Expected %v", actual.Values(), expected))
		}
	}

	bgrads := [][]float32{
		fill(4, ReLuPrime(6)),
		fill(2, 0.5*ReLuPrime(25)),
		fill(1, 0.5),
	}

	for i := 0; i < len(net.Activations); i++ {
		expected := bgrads[i]
		actual := net.BGradients[i]
		if !sameArray(expected, actual.Values()) {
			t.Errorf(fmt.Sprintf("Got %v, Expected %v", actual.Values(), expected))
		}
	}
}

func TestApplyGradients(t *testing.T) {
	net := createNetwork()

	input := []int16{0, 1, 2, 3, 4, 5, 6, 7}
	net.Predict(input)
	net.FindErrors(0.5)
	net.UpdateGradients(input)

	wgrads := [][]float32{
		fill(32, 1),
		fill(8, 4.5),
		fill(2, 37*0.5),
	}

	bgrads := [][]float32{
		fill(4, 1),
		fill(2, 0.5),
		fill(1, 0.5),
	}

	update := func(x float32, gv float32) float32 {
		g := Gradient{Value: gv}
		return x - g.Calculate()
	}

	applyAll := func(xs []float32, gs []float32) []float32 {
		ys := make([]float32, len(xs))
		for i := 0; i < len(xs); i++ {
			ys[i] = update(xs[i], gs[i])
		}
		return ys
	}

	bExpected := make([][]float32, len(bgrads))
	wExpected := make([][]float32, len(wgrads))

	for i := 0; i < len(net.Activations); i++ {
		wExpected[i] = applyAll(net.Weights[i].Data, wgrads[i])
		bExpected[i] = applyAll(net.Biases[i].Data, bgrads[i])
	}

	net.ApplyGradients()

	for i := 0; i < len(net.Activations); i++ {
		expected := wExpected[i]
		actual := net.Weights[i]
		if !sameArray(expected, actual.Data) {
			t.Errorf(fmt.Sprintf("Issues in weight update: Got %v, Expected %v", actual.Data, expected))
		}

		expected = bExpected[i]
		actual = net.Biases[i]
		if !sameArray(expected, actual.Data) {
			t.Errorf(fmt.Sprintf("Issues in bias update: Got %v, Expected %v", actual.Data, expected))
		}
	}

	for i := 0; i < len(net.Activations); i++ {
		bgrad := net.BGradients[i]
		for _, g := range bgrad.Data {
			if g.Value != 0 {
				t.Errorf("Gradients do not reset")
			}
		}

		wgrad := net.WGradients[i]
		for _, g := range wgrad.Data {
			if g.Value != 0 {
				t.Errorf("Gradients do not reset")
			}
		}
	}
}

func TestApplyGradientsPartialInput(t *testing.T) {
	net := createNetwork()

	input := []int16{0, 2, 3, 5, 6}
	net.Predict(input)
	net.FindErrors(0.5)
	net.UpdateGradients(input)

	wgrads := [][]float32{
		{1, 1, 1, 1,
			0, 0, 0, 0,
			1, 1, 1, 1,
			1, 1, 1, 1,
			0, 0, 0, 0,
			1, 1, 1, 1,
			1, 1, 1, 1,
			0, 0, 0, 0},
		fill(8, 3),
		fill(2, 25*0.5),
	}

	bgrads := [][]float32{
		fill(4, ReLuPrime(6)),
		fill(2, 0.5*ReLuPrime(25)),
		fill(1, 0.5),
	}

	update := func(x float32, gv float32) float32 {
		g := Gradient{Value: gv}
		return x - g.Calculate()
	}

	applyAll := func(xs []float32, gs []float32) []float32 {
		ys := make([]float32, len(xs))
		for i := 0; i < len(xs); i++ {
			ys[i] = update(xs[i], gs[i])
		}
		return ys
	}

	bExpected := make([][]float32, len(bgrads))
	wExpected := make([][]float32, len(wgrads))

	for i := 0; i < len(net.Activations); i++ {
		wExpected[i] = applyAll(net.Weights[i].Data, wgrads[i])
		bExpected[i] = applyAll(net.Biases[i].Data, bgrads[i])
	}

	net.ApplyGradients()

	for i := 0; i < len(net.Activations); i++ {
		expected := wExpected[i]
		actual := net.Weights[i]
		if !sameArray(expected, actual.Data) {
			t.Errorf(fmt.Sprintf("Issues in weight update: Got %v, Expected %v", actual.Data, expected))
		}

		expected = bExpected[i]
		actual = net.Biases[i]
		if !sameArray(expected, actual.Data) {
			t.Errorf(fmt.Sprintf("Issues in bias update: Got %v, Expected %v", actual.Data, expected))
		}
	}

	for i := 0; i < len(net.Activations); i++ {
		bgrad := net.BGradients[i]
		for _, g := range bgrad.Data {
			if g.Value != 0 {
				t.Errorf("Gradients do not reset")
			}
		}

		wgrad := net.WGradients[i]
		for _, g := range wgrad.Data {
			if g.Value != 0 {
				t.Errorf("Gradients do not reset")
			}
		}
	}
}

func TestGradientsDifferences(t *testing.T) {
	for i := 0; i < 1000; i++ {
		top := NewTopology(768, 1, []uint32{128, 64})
		net := CreateNetwork(top, 30)

		line := "5k2/ppp5/4P3/3R3p/6P1/1K2Nr2/PP3P2/8 b - - 1 32;score:-72;eval:50;qs:0;outcome:0.5"
		data := ParseLine(line)

		net.Train(data.Input, Sigmoid(float32(data.Score)), float32(data.Outcome)/2)

		lastOutput := net.Predict(data.Input)
		e1 := ValidationCost(lastOutput, Sigmoid(float32(data.Score)), float32(data.Outcome)/2)

		change := float32(0.001)
		changeIndex := rand.Intn(len(net.Weights[2].Data))

		// Tweak a weight
		net.Weights[2].Data[changeIndex] += change

		lastOutput = net.Predict(data.Input)
		e2 := ValidationCost(lastOutput, Sigmoid(float32(data.Score)), float32(data.Outcome)/2)

		grad := (e2 - e1) / change

		diff := net.WGradients[2].Data[changeIndex].Value - grad
		if diff < 0 {
			diff *= -1
		}

		avg := (grad + net.WGradients[2].Data[changeIndex].Value) / 2

		if (diff/avg)*100 >= 10 {
			t.Errorf(fmt.Sprintf("Computed gradient is off: Got %v, Expected %v. Change is %f", net.WGradients[2].Data[changeIndex].Value, grad, (diff/grad)*100))
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

func fill(size int, with float32) []float32 {
	a := make([]float32, size)
	for i := 0; i < len(a); i++ {
		a[i] = with
	}
	return a
}
