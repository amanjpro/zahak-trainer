package main

import (
	"fmt"
	"strconv"
	"unicode"
)

type (
	Color       uint8
	PieceType   uint8
	Piece       uint8
	Square      uint8
	PositionTag uint8
)

// const (
// 	WhiteToMove PositionTag = 1 << iota
// 	BlackToMove
// 	WhiteCanCastleKingSide
// 	WhiteCanCastleQueenSide
// 	BlackCanCastleKingSide
// 	BlackCanCastleQueenSide
// )

const (
	A1 Square = iota
	B1
	C1
	D1
	E1
	F1
	G1
	H1
	A2
	B2
	C2
	D2
	E2
	F2
	G2
	H2
	A3
	B3
	C3
	D3
	E3
	F3
	G3
	H3
	A4
	B4
	C4
	D4
	E4
	F4
	G4
	H4
	A5
	B5
	C5
	D5
	E5
	F5
	G5
	H5
	A6
	B6
	C6
	D6
	E6
	F6
	G6
	H6
	A7
	B7
	C7
	D7
	E7
	F7
	G7
	H7
	A8
	B8
	C8
	D8
	E8
	F8
	G8
	H8
)

const (
	White Color = iota
	Black
)

const (
	Pawn PieceType = iota
	Knight
	Bishop
	Rook
	Queen
	King
)

const (
	WhitePawn Piece = iota
	WhiteKnight
	WhiteBishop
	WhiteRook
	WhiteQueen
	WhiteKing
	BlackPawn
	BlackKnight
	BlackBishop
	BlackRook
	BlackQueen
	BlackKing
	NoPiece
)

func (p Piece) Flip() Piece {
	if p < BlackPawn {
		return p + BlackPawn
	} else {
		return p - BlackPawn
	}
}

func pieceFromName(name rune) Piece {
	switch name {
	case 'P':
		return WhitePawn
	case 'N':
		return WhiteKnight
	case 'B':
		return WhiteBishop
	case 'R':
		return WhiteRook
	case 'Q':
		return WhiteQueen
	case 'K':
		return WhiteKing
	case 'p':
		return BlackPawn
	case 'n':
		return BlackKnight
	case 'b':
		return BlackBishop
	case 'r':
		return BlackRook
	case 'q':
		return BlackQueen
	case 'k':
		return BlackKing
	}
	return NoPiece
}

var ranks = []Square{A8, A7, A6, A5, A4, A3, A2, A1}

func FromFen(fen string) ([]int16, bool) {

	length := 0
	whiteToMove := false
	stop := false
	seenSpace := false
	for i := 0; i < len(fen); i++ {
		if stop {
			break
		}

		ch := rune(fen[i])
		if ch == ' ' {
			seenSpace = true
			continue
		}

		if seenSpace {
			if ch == 'w' {
				whiteToMove = true
				stop = true
			} else if ch == 'b' {
				stop = true
			}
		} else if ('a' <= ch && ch <= 'z') || ('A' <= ch && ch <= 'Z') {
			length++
		}
	}

	input := make([]int16, length)

	rank := 0
	boardIndex := A8
	pieceCounts := 0
	for i := 0; i < len(fen); i++ {
		ch := rune(fen[i])
		if ch == ' ' || rank >= len(ranks) {
			break // end of the board
		} else if unicode.IsDigit(ch) {
			n, _ := strconv.Atoi(string(ch))
			boardIndex += Square(n)
		} else if ch == '/' && boardIndex%8 == 0 {
			rank++
			boardIndex = ranks[rank]
			continue
		} else if p := pieceFromName(ch); p != NoPiece {
			if whiteToMove {
				input[pieceCounts] = int16(p)*64 + int16(boardIndex)
			} else {
				input[pieceCounts] = int16(p.Flip())*64 + int16(boardIndex^56)
			}
			pieceCounts++
			boardIndex++
		} else {
			panic(fmt.Sprintf("Invalid FEN notation %s, boardIndex == %d, parsing %s\n",
				fen, boardIndex, string(ch)))
		}
	}

	return input, whiteToMove
}
