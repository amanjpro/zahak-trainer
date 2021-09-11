package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

type (
	Sample struct {
		Input      *Matrix
		EvalTarget *Matrix
		WDLTarget  *Matrix
	}

	Gradient struct {
		Value        float32
		FirstMoment  float32
		SecondMoment float32
	}

	Data struct {
		Position Position
		Score    float32
		Outcome  float32
	}
)

const (
	SigmoidScale   float32 = 2.5 / 1024
	CostEvalWeight float32 = 0.5
	CostWDLWeight  float32 = 1.0 - CostEvalWeight
	LearningRate   float32 = 0.01
)

func Sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(SigmoidScale*-x))))
}

func SigmoidPrime(x float32) float32 {
	return x * (1.0 - x) * SigmoidScale
}

func ReLu(x float32) float32 {
	return float32(math.Max(float64(x), 0.0))
}

func ReLuPrime(x float32) float32 {
	if x > 0.0 {
		return 1.0
	}
	return 0.0
}

func CalculateCost(output, wdlTarget, evalTarget *Matrix) *Matrix {
	costMatrix := FromTemplate(output)
	for i := uint32(0); i < output.Size(); i++ {
		lhs := CostEvalWeight * float32(math.Pow(float64(output.Data[i]-evalTarget.Data[i]), 2.0))
		rhs := CostWDLWeight * float32(math.Pow(float64(output.Data[i]-wdlTarget.Data[i]), 2.0))
		costMatrix.Data[i] = lhs + rhs
	}
	return costMatrix
}

// func (net *Network) CalculateCosts(samples []Sample) float32 {
// 	var cost float32 = 0.0
//
// 	for _, sample := range samples {
// 		cost += net.CalculateCost(sample)
// 	}
//
// 	return cost / float32(len(samples))
// }

// func CalculateCostGradient(sample Sample, output *Matrix) float32 {
// 	return 2.0*CostEvalWeight*(output.Mean()-sample.EvalTarget.Mean()) +
// 		2.0*CostWDLWeight*(output.Mean()-sample.WDLTarget.Mean())
// }

func (grad *Gradient) UpdateGradient(delta float32) {
	grad.Value += delta
}

func (grad *Gradient) CalculateGradient() float32 {
	var beta1 float32 = 0.9
	var beta2 float32 = 0.999
	var lr float32 = 0.01

	if grad.Value == 0 {
		return 0
	}

	grad.FirstMoment = grad.FirstMoment*beta1 + grad.Value*(1.0-beta1)
	grad.SecondMoment = grad.SecondMoment*beta2 + (grad.Value*grad.Value)*(1.0-beta2)

	return lr * grad.FirstMoment / float32(math.Sqrt(float64(grad.SecondMoment))+1e-8)
}

func (grad *Gradient) ApplyGradient(parameter float32) float32 {
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
	score, err := strconv.ParseFloat(scorePart[1], 32)
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	outcomePart := strings.Split(parts[3], ":")
	if len(outcomePart) != 2 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}
	outcome, err := strconv.ParseFloat(outcomePart[1], 32)
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	return Data{
		Position: pos,
		Score:    float32(score),
		Outcome:  float32(outcome),
	}
}
