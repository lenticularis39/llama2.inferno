# Inference for Llama-2 Transformer model in pure Limbo

implement Llama2;

include "sys.m";
include "draw.m";
include "math.m";

sys: Sys;
math: Math;

read_int_buf: array of byte;
read_int_ibuf : array of int;

Config: adt {
	dim: int; # transformer dimension
	hidden_dim: int; # for ffn layers
	n_layers: int; # number of layers
	n_heads: int; # number of heads
	n_kv_heads: int; # number of query heads
	vocab_size: int; # vocabulary size, usually 256 (byte-level)
	seq_len: int;
	shared_weights: int;

	read: fn(c: self ref Config, fd: ref Sys->FD);
};

TransformerWeights: adt {
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

	read: fn(w: self ref TransformerWeights, c : ref Config, fd: ref Sys->FD);
};

RunState: adt {
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
	config: ref Config; # the hyperparameters of the architecture (the blueprint)
	weights: ref TransformerWeights; # the weights of the model
	state: ref RunState; # buffers for the "wave" of the activations in the forward pass
};

Llama2: module {
	init: fn(ctxt: ref Draw->Context, argv: list of string);
};

endian_swap(b : array of byte) {
	(b[0], b[1], b[2], b[3]) = (b[3], b[2], b[1], b[0]);
}

read_int(fd: ref Sys->FD): int {
	if (sys->read(fd, read_int_buf, 4) < 4)
		raise "fail:eof";

	endian_swap(read_int_buf);
	math->import_int(read_int_buf, read_int_ibuf);

	return read_int_ibuf[0];
}

read_weights(fd: ref Sys->FD, size: int): array of real {
	buf := array[size * 4] of byte;
	weights := array[size] of real;

	if (sys->read(fd, buf, len buf) != len buf)
		raise "fail:eof";

	for (i := 0; i < size; i++)
		endian_swap(buf[4 * i:]);
	math->import_real32(buf, weights);

	return weights;
}

Config.read(c: self ref Config, fd: ref Sys->FD) {
	c.dim = read_int(fd);
	c.hidden_dim = read_int(fd);
	c.n_layers = read_int(fd);
	c.n_heads = read_int(fd);
	c.n_kv_heads = read_int(fd);
	c.vocab_size = read_int(fd);
	c.seq_len = read_int(fd);
}

TransformerWeights.read(w: self ref TransformerWeights, c: ref Config, fd: ref Sys->FD) {
	buf : array of byte;
	size : int;

	head_size := c.dim / c.n_heads;

	w.token_embedding_table = read_weights(fd, c.vocab_size * c.dim);
	w.rms_att_weight = read_weights(fd, c.n_layers);
	w.wq = read_weights(fd, c.n_layers * c.dim * (c.n_heads * head_size));
	w.wk = read_weights(fd, c.n_layers * c.dim * (c.n_kv_heads * head_size));
	w.wv = read_weights(fd, c.n_layers * c.dim * (c.n_kv_heads * head_size));
	w.wo = read_weights(fd, c.n_layers * (c.n_heads * head_size) * c.dim);
	w.rms_ffn_weight = read_weights(fd, c.n_layers * c.dim);
	w.w1 = read_weights(fd, c.n_layers * c.dim * c.hidden_dim);
	w.w2 = read_weights(fd, c.n_layers * c.hidden_dim * c.dim);
	w.w3 = read_weights(fd, c.n_layers * c.dim * c.hidden_dim);
	w.rms_final_weight = read_weights(fd, c.dim);

	if (c.shared_weights)
		w.wcls = w.token_embedding_table;
	else
		w.wcls = read_weights(fd, c.seq_len * head_size);
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
	config.shared_weights = config.vocab_size > 0;
	config.vocab_size *= 2 * config.shared_weights - 1;

	# load weights
	weights.read(config, fd);
}

build_transformer(t: ref Transformer, checkpoint_path: string) {
	# allocate Transformer fields
	t.config = ref Config;
	t.weights = ref TransformerWeights;
	t.state = ref RunState;
	# read in the Config and the Weights from the checkpoint
	read_checkpoint(checkpoint_path, t.config, t.weights);
	# allocate the RunState buffers
	alloc_run_state(t.state, t.config);
}

init(ctxt: ref Draw->Context, argv: list of string) {
	sys = load Sys Sys->PATH;
	math = load Math Math->PATH;

	read_int_buf = array[4] of byte;
	read_int_ibuf = array[1] of int;

	t := ref Transformer;

	argv = tl argv;
	if (argv == nil)
		raise "fail:noarg";

	sys->print("argv: %s\n", hd argv);

	build_transformer(t, hd argv);
	c := t.config;

	sys->print("dim: %d, hidden_dim: %d, n_layers: %d\n",
			  c.dim, c.hidden_dim, c.n_layers);
	sys->print("n_heads: %d, n_kv_heads: %d, vocab_size: %d\n",
			  c.n_heads, c.n_kv_heads, c.vocab_size);
	sys->print("seq_len: %d\n", c.seq_len);
}
