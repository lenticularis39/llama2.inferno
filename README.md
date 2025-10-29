## llama2.inferno

This is a port of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c)
to Inferno/Limbo, in progress.

## prerequisities

See llama2.c above.

## feel the ~~magic~~ hellfire

Build llama2 with:

```bash
% mk
m4 -U include -U index -U len llama2-m4.b > llama2.b
limbo -g llama2.b
```

Run llama2 like this:

```bash
% llama2
Usage: llama2 <checkpoint> [options]
Example: llama2 model.bin -n 256 -i 'Once upon a time'
Options: 
  -v          verbose mode, default disabled
  -t <float>  temperature in [0,inf], default 1.0
  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9
  -s <int>    random seed, default time(NULL)
  -n <int>    number of steps to run for, default 256. 0 = max_seq_len
  -i <string> input prompt
  -z <string> optional path to custom tokenizer
% llama2 stories15M.bin -n 10 -t 0.8 -p 0.95 -s 42 -i 'Hello world'
Hello world had been sent to the world. He
achieved tok/s: 1.182499
```

## performance

For better performance, use Inferno JIT, if available on your platform.
