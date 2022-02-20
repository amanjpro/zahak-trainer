package main

import (
	"bufio"
	"encoding/binary"
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
		Score   int16
		Outcome int8
	}
)

func countSamples(paths []string) int64 {
	fmt.Printf("Paths to load %s\n", paths)
	totalCount := int64(0)
	countLines := func(f *os.File) int64 {
		count := int64(0)
		input := bufio.NewScanner(f)
		for input.Scan() {
			count++
			if err := input.Err(); err != nil {
				panic(err)
			}
		}

		return count
	}
	for _, path := range paths {
		file, err := os.Open(path)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		totalCount += countLines(file)
	}

	fmt.Printf("Loading %d samples\n", totalCount)

	return totalCount
}

func SaveDataset(paths string, file string) {
	pathsArray := strings.Split(paths, ",")

	f, err := os.Create(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	buf8 := make([]byte, 8)
	samples := countSamples(pathsArray)
	binary.LittleEndian.PutUint64(buf8, uint64(samples))
	_, err = f.Write(buf8)
	if err != nil {
		panic(err)
	}

	var buf4 = make([]byte, 4)
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
			skipNext = pre
			if pre {
				continue
			} else if skipNext {
				skipNext = false
				continue
			}
			line := string(buf)
			if line == "" {
				break
			}
			sample := ParseLine(line)
			binary.LittleEndian.PutUint16(buf4, uint16(sample.Outcome))
			_, err = f.Write(buf4)
			if err != nil {
				panic(err)
			}
			binary.LittleEndian.PutUint16(buf4, uint16(sample.Score))
			_, err = f.Write(buf4)
			if err != nil {
				panic(err)
			}
			binary.LittleEndian.PutUint16(buf4, uint16(len(sample.Input)))
			_, err = f.Write(buf4)
			if err != nil {
				panic(err)
			}
			for _, i := range sample.Input {
				binary.LittleEndian.PutUint16(buf4, uint16(i))
				_, err = f.Write(buf4)
				if err != nil {
					panic(err)
				}
			}
		}
	}
}

func LoadBinpack(path string) []Data {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	buf8 := make([]byte, 8)
	_, err = io.ReadFull(f, buf8)
	if err != nil {
		panic(err)
	}

	counter := int64(0)

	var buf4 = make([]byte, 4)
	data := make([]Data, binary.LittleEndian.Uint64(buf8))
	for i := 0; i < len(data); i++ {
		_, err = io.ReadFull(f, buf4)
		if err != nil {
			panic(err)
		}
		outcome := int8(binary.LittleEndian.Uint16(buf4))
		_, err = io.ReadFull(f, buf4)
		if err != nil {
			panic(err)
		}
		score := int16(binary.LittleEndian.Uint16(buf4))
		_, err = io.ReadFull(f, buf4)
		if err != nil {
			panic(err)
		}
		inputLength := binary.LittleEndian.Uint16(buf4)
		input := make([]int16, inputLength)
		for j := uint16(0); j < inputLength; j++ {
			_, err = io.ReadFull(f, buf4)
			if err != nil {
				panic(err)
			}
			input[j] = int16(binary.LittleEndian.Uint16(buf4))
		}

		data[i] = Data{
			Score:   score,
			Outcome: outcome,
			Input:   input,
		}

		if counter == 2000 {
			fmt.Printf("%d of %d is loaded\r", i, len(data))
			counter = 0
		}
		counter++
	}

	return data
}

func LoadDataset(paths string) []Data {
	pathsArray := strings.Split(paths, ",")
	data := make([]Data, countSamples(pathsArray))
	samples := 0
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
			data[samples] = sample
			samples += 1
		}
	}

	runtime.GC()

	return data
}

func ParseLine(line string) Data {
	startIndex := 0
	endIndex := strings.Index(line, ";")
	if endIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	pos := FromFen(line[:endIndex])
	// wm := pos[len(pos)-1] == 768

	startIndex = endIndex + 7
	endIndex = strings.Index(line, ";eval")
	if endIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	score, err := strconv.Atoi(line[startIndex:endIndex])
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}

	startIndex = strings.Index(line, ";outcome:") + 9
	if startIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}
	var outcome int8
	if line[startIndex:] == "0.0" {
		outcome = 0
	} else if line[startIndex:] == "1.0" {
		outcome = 2
	} else {
		outcome = 1
	}

	// if !wm {
	// 	score *= -1
	// 	if outcome == 2 {
	// 		outcome = 0
	// 	} else if outcome == 0 {
	// 		outcome = 2
	// 	}
	// }

	return Data{
		Input:   pos,
		Score:   int16(score),
		Outcome: outcome,
	}
}
