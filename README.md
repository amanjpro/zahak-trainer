# Zahak-Trainer

A trainer for Zahak's NNUE. The produced nets are compatible with both Zahak
and Bit-Genie engine. In theory one should be able to load networks that is
produced with this trainer to either engine, it might not always produce the
optimal play though.

Input format is expected to look like the following:

```
1n1qkb1r/r2b1p2/4pn1p/p2p4/2pP1BpP/2P1P3/1PN2PP1/RN1QKB1R b KQk - 1 11;score:-31;eval:25;qs:25;outcome:0.5
r5kr/1p3ppp/1p6/1P1p4/3P2P1/qP5P/3NPP2/2RQK2R w K - 0 16;score:459;eval:449;qs:449;outcome:1.0
8/1r2p1k1/2Nnn1pp/R7/8/5NPP/5PK1/8 b - - 6 38;score:32;eval:12;qs:12;outcome:0.5
r3k1r1/1p6/2p1pb1p/2P2p2/p2P1P2/4P1P1/PP1B1K1P/1R2R3 w q - 2 27;score:188;eval:164;qs:164;outcome:1.0
6R1/pp6/1k4P1/r7/5K2/2p4P/8/8 w - - 2 54;score:-94;eval:-88;qs:-88;outcome:0.0
8/5k2/8/4p2p/4P3/2r2PP1/2p2K2/2R5 w - - 0 53;score:-123;eval:-112;qs:-112;outcome:0.5
8/3k1n2/1K5p/7R/6P1/8/8/8 b - - 3 55;score:315;eval:306;qs:306;outcome:0.5
8/6R1/8/5K2/5p2/2r2k2/8/8 w - - 4 67;score:-193;eval:-154;qs:-154;outcome:0.0
r2qkbr1/ppp1pb2/2n2n2/3pP2p/P2P2p1/2PQ2P1/1P1N1PB1/R1B1K1NR b KQq - 0 9;score:87;eval:-1;qs:-1;outcome:0.5
5rk1/R4pp1/5n1p/1r6/1bqp1P2/4PK1P/3P2P1/1N2Q1NR w - - 0 23;score:-98;eval:-145;qs:-145;outcome:0.0
```

Basically, `<FEN>;score:<SCORE>;eval:<EVAL>;qs:<QUIESCENCE SEARCH SCORE>;outcome:[1.0|0.5|0.0]`

Self-play games data can be generated by using `cutechess-cli`, for example:

```
cutechess-cli -tournament gauntlet -concurrency 15 \
  -pgnout zahak_games/PGN_NAME.pgn "fi" \
  -engine conf=zahak_latest tc="inf" depth=9 \
  -engine conf=zahak_latest tc="inf" depth=9 \
  -ratinginterval 1 \
  -recover \
  -event SELF_PLAY_GAMES \
  -draw movenumber=40 movecount=10 score=20 \
  -resign movecount=5 score=1000 \
  -resultformat per-color \
  -openings order=random policy=round file=SOME_BOOK_HERE format="epd" \
  -each proto=uci option.Hash=32 option.Threads=1 \
  -rounds 100000000
```

And then [fengen](https://github.com/amanjpro/fengen) can be used to convert
the games into FENs, please consult the `fengen` repository for more
information.

The trainer exposes many command line arguments to configure the net and its
architecture:

```
$ ./zahak-trainer -help
Usage of ./zahak-trainer:
  -epochs int
    Number of epochs (default 100)
  -from-net string
    Path to a network, to be used as a starting point
  -hiddens string
    Number of hidden neurons, for multi-layer you can send comma separated numbers (default "256")
  -input-path string
    Path to input dataset (FENs), for multiple files send a comma separated set of files
  -inputs int
    Number of inputs (default 769)
  -lr float
    Learning Rate (default 0.009999999776482582)
  -network-id int
    A unique id for the network (default 1277010531)
  -output-path string
    Final NNUE path directory
  -outputs int
    Number of outputs (default 1)
  -profile
    Profile the trainer
  -sigmoid-scale float
    Sigmoid scale (default 0.0068359375)
```


# Acknowledgement

- Aryan Parekh the author of
  [Bit-Genie](https://github.com/Aryan1508/Bit-Genie) for helping me understand
  the basics of
  NN and NNUE, and sticking with me and patiently available for answering all my
  (stupid) qeustions until the very end.
