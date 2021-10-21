package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
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

func LoadDataset(paths string) []*Data {
	pathsArray := strings.Split(paths, ",")
	data := make([]*Data, 0, 57_220_422)
	for _, path := range pathsArray {
		file, err := os.Open(path)
		if err != nil {
			panic(err)
		}
		defer file.Close()

		reader := bufio.NewReader(file)
		skipNext := false
		for true {
			buf, pre, err := reader.ReadLine()
			if errors.Is(err, io.EOF) {
				break
			} else if err != nil {
				panic(err)
			}
			if pre {
				continue
			} else if skipNext {
				skipNext = false
				continue
			}
			skipNext = pre
			line := string(buf)
			if line == "" {
				break
			}
			sample := ParseLine(line)
			data = append(data, sample)
		}

	}

	runtime.GC()

	return data
}

func ParseLine(line string) *Data {
	startIndex := 0
	endIndex := strings.Index(line, ";")
	if endIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	pos := FromFen(line[:endIndex])

	startIndex = endIndex + 7
	endIndex = strings.Index(line, ";eval")
	if endIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	score, err := strconv.Atoi(line[startIndex:endIndex])
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}

	normalizedScore := Sigmoid(float32(score))

	startIndex = strings.Index(line, ";outcome:") + 9
	if startIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}
	outcome, err := strconv.ParseFloat(line[startIndex:], 32)
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}

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
