package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

type (
	Sample struct {
		Input      []float64
		EvalTarget float64
		WDLTarget  float64
	}

	Gradient struct {
		Value        float64
		FirstMoment  float64
		SecondMoment float64
	}

	Data struct {
		Position Position
		Score    float64
		Outcome  float64
	}
)

const (
	SigmoidScale   float64 = 2.5 / 1024
	CostEvalWeight float64 = 0.5
	CostWDLWeight  float64 = 1.0 - CostEvalWeight
)

func Sigmoid(x float64) float64 {

	return 1.0 / (1.0 + math.Exp(SigmoidScale*-x))
}

func SigmoidPrime(x float64) float64 {
	return x * (1.0 - x) * SigmoidScale
}

func Relu(x float64) float64 {
	return math.Max(x, 0.0)
}

func ReluPrime(x float64) float64 {
	if x > 0.0 {
		return 1.0
	}
	return 0.0
}

func (net *Network) CalculateCost(sample Sample) float64 {
	var output float64 = net.ForwardPropagate(sample.Input)

	return CostEvalWeight*math.Pow(output-sample.EvalTarget, 2.0) +
		CostWDLWeight*math.Pow(output-sample.WDLTarget, 2.0)
}

func (net *Network) CalculateCosts(samples []Sample) float64 {
	var cost float64 = 0.0

	for _, sample := range samples {
		cost += net.CalculateCost(sample)
	}

	return cost / float64(len(samples))
}

func CalculateCostGradient(sample Sample, output float64) float64 {
	return 2.0*CostEvalWeight*(output-sample.EvalTarget) +
		2.0*CostWDLWeight*(output-sample.WDLTarget)
}

func (grad *Gradient) UpdateGradient(delta float64) {
	grad.Value += delta
}

func (grad *Gradient) CalculateGradient() float64 {
	var beta1 float64 = 0.9
	var beta2 float64 = 0.999
	var lr float64 = 0.01

	if grad.Value == 0 {
		return 0
	}

	grad.FirstMoment = grad.FirstMoment*beta1 + grad.Value*(1.0-beta1)
	grad.SecondMoment = grad.SecondMoment*beta2 + (grad.Value*grad.Value)*(1.0-beta2)

	return lr * grad.FirstMoment / (math.Sqrt(grad.SecondMoment) + 1e-8)
}

func (grad *Gradient) ApplyGradient(parameter float64) float64 {
	parameter -= grad.CalculateGradient()
	grad.Value = 0
	return parameter
}

func ParseLine(line string) Data {
	parts := strings.Split(line, ";")
	if len(parts) != 4 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	pos := FromFen(parts[0])

	scorePart := strings.Split(parts[1], ":")
	if len(scorePart) != 2 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}
	score, err := strconv.ParseFloat(scorePart[1], 64)
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	outcomePart := strings.Split(parts[3], ":")
	if len(outcomePart) != 2 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}
	outcome, err := strconv.ParseFloat(outcomePart[1], 64)
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	return Data{
		Position: pos,
		Score:    score,
		Outcome:  outcome,
	}
}
