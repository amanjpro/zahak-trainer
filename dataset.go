package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

type (
	Data struct {
		Input   *Position
		Score   float32
		Outcome float32
	}
)

func LoadDataset(path string) *[]Data {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data := make([]Data, 0, 58_671_054)

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

	return &data
}

func ParseLine(line string) Data {
	parts := strings.Split(line, ";")
	if len(parts) != 5 {
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

	score = math.Max(math.Min(score*100, 2000), -2000)

	outcomePart := strings.Split(parts[4], ":")
	if len(outcomePart) != 2 {
		panic(fmt.Sprintf("Bad line %s\n", parts[4]))
	}
	outcome, err := strconv.ParseFloat(outcomePart[1], 32)
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	normalizedScore := Sigmoid(float32(score))

	return Data{
		Input:   pos,
		Score:   normalizedScore,
		Outcome: float32(outcome),
	}
}
