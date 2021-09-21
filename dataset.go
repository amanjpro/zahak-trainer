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
		Input   []int16
		Score   float32
		Outcome float32
	}
)

func LoadDataset(path string) []*Data {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data := make([]*Data, 0, 2_000_100) //53_907_348)

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			break
		}
		sample := ParseLine(line)
		data = append(data, sample)
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}

	return data
}

func ParseLine(line string) *Data {
	parts := strings.Split(line, ";")
	if len(parts) != 5 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	pos := FromFen(parts[0])

	scorePart := strings.Split(parts[1], ":")
	if len(scorePart) != 2 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}
	score, err := strconv.Atoi(scorePart[1])
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	score = clamp(-2000, score, 2000)

	outcomePart := strings.Split(parts[4], ":")
	outcome, err := strconv.ParseFloat(outcomePart[1], 32)
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}
	normalizedScore := Sigmoid(float32(score))

	return &Data{
		Input:   pos,
		Score:   normalizedScore,
		Outcome: float32(outcome),
	}
}

func clamp(lower, value, upper int) int {
	if value > upper {
		return upper
	}

	if value < lower {
		return lower
	}
	return value
}
