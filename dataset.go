package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type (
	Data struct {
		Input   Position
		Score   *Matrix
		Outcome *Matrix
	}
)

func LoadDataset(path string) []*Data {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data := make([]*Data, 0, 14_000_000)

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			break
		}
		data = append(data, ParseLine(line))
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}

	return data
}

func ParseLine(line string) *Data {
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

	return &Data{
		Input:   pos,
		Score:   NewMatrix(1, 1, []float32{Sigmoid(float32(score))}),
		Outcome: NewMatrix(1, 1, []float32{float32(outcome)}),
	}
}
