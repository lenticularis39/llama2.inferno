# Inference for Llama-2 Transformer model in pure Limbo

implement Llama2;

include "sys.m";
include "draw.m";

sys: Sys;

Config : adt {
	dim: int; # transformer dimension
	hidden_dim: int; # for ffn layers
	n_layers: int; # number of layers
	n_heads: int; # number of heads
	n_kv_heads: int; # number of query heads
	vocab_size: int; # vocabulary size, usually 256 (byte-level)
	seq_len: int;

	read: fn(c: self ref Config, fd: ref Sys->FD);
};

TransformerWeights : adt {
	# token embedding table
	token_embedding_table: array of real; # (vocab_size, dim)
	# weights for rmsnorms
	rms_att_weight: array of real; # (layer, dim) rmsnorm weights
	rms_ffn_weight: array of real; # (layer, dim)
	# weights for matmuls, note dim == n_heads * head_size
	wq: array of real; # (layer, dim, n_heads * head_size)
	wk: array of real; # (layer, dim, n_kv_heads * head_size)
	wv: array of real; # (layer, dim, n_kv_heads * head_size)
	wo: array of real; # (layer, n_heads * head_size, dim)
	# weights for ffn
	w1: array of real; # (layer, hidden_dim, dim)
	w2: array of real; # (layer, dim, hidden_dim)
	w3: array of real; # (layer, hidden_dim, dim)
	# final rmsnorm
	rms_final_weight: array of real; # (dim,)
	# (optional) classifier weights for the logits, on the last layer
	wcls: array of real;
};

RunState : adt {
	x: array of real; # activation at current time stamp (dim,)
	xb: array of real; # same, but inside a residual branch (dim,)
	xb2: array of real; # an additional buffer just for convenience (dim,)
	hb: array of real; # buffer for hidden dimension in the ffn (hidden_dim,)
	hb2: array of real; # buffer for hidden dimension in the ffn (hidden_dim,)
	q: array of real; # query (dim,)
	k: array of real; # key (dim.)
	v: array of real; # value (dim.)
	att: array of real; # buffer for scores/attention values (n_heads, seq_len)
	logits: array of real; # output logits
	# kv cache
	key_cache: array of real; # (layer, seq_len, dim)
	value_cache: array of real; # (layer, seq_len, dim)
};

Transformer: adt {
	config: Config; # the hyperparameters of the architecture (the blueprint)
	weights: TransformerWeights; # the weights of the model
	state: RunState; # buffers for the "wave" of the activations in the forward pass
};

Llama2: module {
	buf: array of byte;
	init: fn(ctxt: ref Draw->Context, argv: list of string);
};

read_int_le(fd: ref Sys->FD) : int {
	if (sys->read(fd, buf, 4) < 4)
		raise "fail:eof";

	return (int buf[0] << 24) | (int buf[1] << 16)
		   | (int buf[2] << 8) | (int buf[3]);
}

Config.read(c: self ref Config, fd: ref Sys->FD) {
	c.dim = read_int_le(fd);
	c.hidden_dim = read_int_le(fd);
	c.n_layers = read_int_le(fd);
	c.n_heads = read_int_le(fd);
	c.n_kv_heads = read_int_le(fd);
	c.vocab_size = read_int_le(fd);
	c.seq_len = read_int_le(fd);
}

alloc_run_state(s: ref RunState, p: ref Config) {
	kv_dim := (p.dim * p.n_kv_heads); # p.n_heads;
	s.x = array[p.dim] of real;
	s.xb = array[p.dim] of real;
	s.xb2 = array[p.dim] of real;
	s.hb = array[p.hidden_dim] of real;
	s.hb2 = array[p.hidden_dim] of real;
	s.q = array[p.dim] of real;
	s.key_cache = array[p.n_layers * p.seq_len * kv_dim] of real;
	s.value_cache = array[p.n_layers * p.seq_len * kv_dim] of real;
	s.att = array[p.n_heads * p.seq_len] of real;
	s.logits = array[p.vocab_size] of real;
}

read_checkpoint(checkpoint: string, config: ref Config, weights: ref TransformerWeights) {
	buffer: array of int;

	fd := sys->open(checkpoint, sys->OREAD);
	if (fd == nil)
		raise "fail:open";

	# read in the config header
	config.read(fd);

	# negative vocab size is a hacky way of signaling unshared weights
	shared_weights := config.vocab_size > 0;
	config.vocab_size *= 2 * shared_weights - 1;
}

init(ctxt: ref Draw->Context, argv: list of string) {
	buf = array[4] of byte;
	sys = load Sys Sys->PATH;
	c := ref Config;
	w := ref TransformerWeights;

	argv = tl argv;
	if (argv == nil)
		raise "fail:noarg";

	sys->print("argv: %s\n", hd argv);

	read_checkpoint(hd argv, c, w);
	sys->print("dim: %d, hidden_dim: %d, n_layers: %d\n",
			  c.dim, c.hidden_dim, c.n_layers);
	sys->print("n_heads: %d, n_kv_heads: %d, vocab_size: %d\n",
			  c.n_heads, c.n_kv_heads, c.vocab_size);
	sys->print("seq_len: %d\n", c.seq_len);
}
