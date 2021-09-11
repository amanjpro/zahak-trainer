package main

import (
	"fmt"
	"strconv"
	"strings"
)

type (
	Data struct {
		Input   *Matrix
		Score   *Matrix
		Outcome *Matrix
	}
)

func (net *Network) ParseLine(line string) Data {
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
		Input:   net.CreateInput(pos),
		Score:   NewMatrix(1, 1, []float32{Sigmoid(float32(score))}),
		Outcome: NewMatrix(1, 1, []float32{float32(outcome)}),
	}
}
