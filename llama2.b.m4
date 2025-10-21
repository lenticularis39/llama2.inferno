# Inference for Llama-2 Transformer model in pure Limbo

implement Llama2;

include "sys.m";
include "draw.m";
include "math.m";
include "strinttab.m";

sys: Sys;
math: Math;
strinttab: StringIntTab;

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

read_real(fd: ref Sys->FD): real {
	if (sys->read(fd, read_int_buf, 4) < 4)
		raise "fail:eof";

	endian_swap(read_int_buf);
	ibuf := array[1] of real;
	math->import_real32(read_int_buf, ibuf);

	return ibuf[0];
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

read_string(fd: ref Sys->FD, size: int): string {
	buf := array[size] of byte;

	if (sys->read(fd, buf, len buf) != len buf)
		raise "fail:eof";

	return string buf;
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
	w.rms_att_weight = read_weights(fd, c.n_layers * c.dim);
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

# ----------------------------------------------------------
# neural net blocks; the dynamics of the Transformer

rmsnorm(o: array of real, x: array of real, weight: array of real) {
	# calculate sum of squares
	minlen := len o;
	if (len x < minlen)
		minlen = len x;
	if (len weight < minlen)
		minlen = len weight;

	ss := 0.0;
	for (j := 0; j < minlen; j++) {
		ss += x[j] * x[j];
	}
	ss /= real (len x);
	ss += 1e-5;
	ss = 1.0 / math->sqrt(ss);
	# normalize and scale
	for (j = 0; j < minlen; j++) {
		o[j] = weight[j] * (ss * x[j]);
	}
}

softmax(x: array of real) {
	# find max value (for numerical stability)
	max_val := x[0];
	for (i := 1; i < len x; i++) {
		if (x[i] > max_val) {
			max_val = x[i];
		}
	}
	# exp and sum
	sum := 0.0;
	for (i = 0; i < len x; i++) {
		x[i] = math->exp(x[i] - max_val);
		sum += x[i];
	}
	# normalize
	for (i = 0; i < len x; i++) {
		x[i] /= sum;
	}
}

matmul(xout: array of real, x: array of real, w: array of real, n: int, d: int) {
	# W (d,n) @ x (n,) -> xout (d,)
	# by fat the most amount of time is spent inside this little function
	for (i := 0; i < d; i++) {
		val := 0.0;
		for (j := 0; j < n; j++) {
			val += w[i * n + j] * x[j];
		}
		xout[i] = val;
	}
}

forward(transformer: ref Transformer, token: int, pos: int): array of real {
	# a few convenience variables
	c := transformer.config;
	w := transformer.weights;
	s := transformer.state;
	x := s.x;
	dim := c.dim;
	kv_dim := (c.dim * c.n_kv_heads) / c.n_heads;
	kv_mul := c.n_heads / c.n_kv_heads; # integer multiplier of the kv sharing in multiquery
	hidden_dim := c.hidden_dim;
	head_size := dim / c.n_heads;

	# copy the token embedding into x
	content_row := w.token_embedding_table[token * dim:];
	x[:] = content_row[:dim];

	# forward all the layers
	for (l := 0; l < c.n_layers; l++) {
		# attention rmsnorm
		rmsnorm(s.xb, x, w.rms_att_weight[l * dim:]);

		# key and value point to the kv cache
		loff := l * c.seq_len * kv_dim;
		s.k = s.key_cache[loff + pos * kv_dim:];
		s.v = s.value_cache[loff + pos * kv_dim:];

		# qkv matmuls for this position
		matmul(s.q, s.xb, w.wq[l * dim * dim:], dim, dim);
		matmul(s.k, s.xb, w.wk[l * dim * kv_dim:], dim, kv_dim);
		matmul(s.v, s.xb, w.wv[l * dim * kv_dim:], dim, kv_dim);

		# RoPE relative positional encoding: complex-valued rotate q and k in each head
		for (i := 0; i < dim; i += 2) {
			head_dim := i % head_size;
			freq := 1.0 / math->pow(10000.0, (real head_dim) / (real head_size));
			val := (real pos) * freq;
			fcr := math->cos(val);
			fci := math->sin(val);
			rotn: int;
			if (i < kv_dim)
				rotn = 2;
			else
				rotn = 1;
			for (v := 0; v < rotn; v++) {
				vec : array of real;
				if (v == 0)
					vec = s.q;
				else
					vec = s.k;
				v0 := vec[i];
				v1 := vec[i + 1];
				vec[i] = v0 * fcr - v1 * fci;
				vec[i + 1] = v0 * fci + v1 * fcr;
			}
		}

		# multihead attention, iterate over all heads
		for (h := 0; h < c.n_heads; h++) {
			# get the query vector for this head
			q := s.q[h * head_size:];
			# attention scores for this head
			att := s.att[h * c.seq_len:];
			# iterate over all timesteps, including the current one
			for (t := 0; t <= pos; t++) {
				# get the key vector for this head and at this timestep
				k := s.key_cache[loff + t * kv_dim + (h / kv_mul) * head_size:];
				# calculate the attention score as the dot product of q and k
				score := 0.0;
				for (i = 0; i < head_size; i++) {
					score += q[i] * k[i];
				}
				score /= math->sqrt(real head_size);
				# save the score to the attention buffer
				att[t] = score;
			}

			# softmax the scores to get attention weights, from 0..pos inclusively
			softmax(att[:pos + 1]);

			# weighted sum of the values, store back into xb
			xb := s.xb[h * head_size:];
			xb[:] = array[head_size] of {* => 0.0};
			for (t = 0; t <= pos; t++) {
				# get the value vector for this head and at this timestamp
				v := s.value_cache[loff + t * kv_dim + (h / kv_mul) * head_size:];
				# get the attention weight for this timestamp
				a := att[t];
				# accumulate the weighted value into xb
				for (i = 0; i < head_size; i++) {
					xb[i] += a * v[i];
				}
			}
		}

		# final matmul to get the output of the attention
		matmul(s.xb2, s.xb, w.wo[l * dim * dim:], dim, dim);

		# residual connection back into x
		for (i = 0; i < dim; i++) {
			x[i] += s.xb2[i];
		}

		# ffn rmsnorm
		rmsnorm(s.xb, x, w.rms_ffn_weight[l * dim:l * dim + dim]);

		# Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		# first calculate self.w1(w) and self.w3(x)
		matmul(s.hb, s.xb, w.w1[l * dim * hidden_dim:], dim, hidden_dim);
		matmul(s.hb2, s.xb, w.w3[l * dim * hidden_dim:], dim, hidden_dim);

		# SwiGLU non-linearity
		for (i = 0; i < hidden_dim; i++) {
			val := s.hb[i];
			# silu(x)=x*sigmoid(x)
			val *= (1.0 / (1.0 + math->exp(-val)));
			# elementwise multiply with w3(x)
			val *= s.hb2[i];
			s.hb[i] = val;
		}

		# final matmul to get the output of the ffn
		matmul(s.xb, s.hb, w.w2[l * dim * hidden_dim:], hidden_dim, dim);

		# residual connection
		for (i = 0; i < dim; i++) {
			x[i] += s.xb[i];
		}
	}

	# final rmsnorm
	rmsnorm(x, x, w.rms_final_weight[:dim]);

	# classifier into logits
	matmul(s.logits, x, w.wcls, c.dim, c.vocab_size);
	return s.logits;
}

# ----------------------------------------------------------
# The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

Tokenizer : adt {
	vocab: array of string;
	vocab_scores: array of real;
	sorted_vocab: array of StringIntTab->StringInt;
	vocab_size: int;
	max_token_length: int;
	byte_pieces: array of byte;
};

build_tokenizer(t: ref Tokenizer, tokenizer_path: string, vocab_size: int) {
	# i should have written the vocab_size into the tokenizer file... sigh
	t.vocab_size = vocab_size;
	# allocate space to hold the scores and the strings
	t.vocab = array[vocab_size] of string;
	t.vocab_scores = array[vocab_size] of real;
	t.sorted_vocab = nil; # initialized lazily
	t.byte_pieces = array[512] of byte;
	for (i := 0; i < 256; i++) {
		t.byte_pieces[i * 2] = byte i;
		t.byte_pieces[i * 2 + 1] = byte '\0';
	}
	# read in the file
	fd := sys->open(tokenizer_path, sys->OREAD);
	t.max_token_length = read_int(fd);
	size: int;
	for (i = 0; i < vocab_size; i++) {
		t.vocab_scores[i] = read_real(fd);
		size = read_int(fd);
		t.vocab[i] = read_string(fd, size);
	}
}

decode(t: ref Tokenizer, prev_token: int, token: int): string {
	piece := t.vocab[token];
	# following BOS (1) token, stencepiece decoder strips any leading whitespace
	if (prev_token == 1 && piece[0] == ' ')
		return t.vocab[token + 1];
	return piece;
}

define(DECLARE_SORT,
	sort_$2(a: array of $1) {
		mergesort_$2(a, array[len a] of $1);
	}

	mergesort_$2(a, b: array of $1) {
		r := len a;
		if (r > 1) {
			m := (r - 1) / 2 + 1;
			mergesort_$2(a[0:m], b[0:m]);
			mergesort_$2(a[m:], b[m:]);
			b[0:] = a;
			for ((i, j, k) := (0, m, 0); i < m && j < r; k++) {
				if(b[i].$3 $4 b[j].$3)
					a[k] = b[j++];
				else
					a[k] = b[i++];
			}
			if (i < m)
				a[k:] = b[i:m];
			else if (j < r)
				a[k:] = b[j:r];
		}
	}
)

DECLARE_SORT(strinttab->StringInt, strinttab, key, >)

encode(t: ref Tokenizer, text: string, bos: int, eos: int, tokens: array of int): int {
	if (t.sorted_vocab == nil) {
		# lazily allocate and sort the vocabulary
		t.sorted_vocab = array[t.vocab_size] of strinttab->StringInt;
		for (i := 0; i < t.vocab_size; i++) {
			t.sorted_vocab[i].key = t.vocab[i];
			t.sorted_vocab[i].val = i;
		}
		sort_strinttab(t.sorted_vocab);
	}

	# create a temporary buffer that will store merge candidates
	# TODO: remove, unneeded since UTF-8 is handled by Limbo
	b: string;

	# start at 0 tokens
	# TODO: remove and replace with "len tokens"
	n_tokens := 0;

	# add optional BOS (=1) token, if desired
	if (bos)
		tokens[n_tokens++] = 1;

	# add_dummy_prefix is true by default
	# so prepend a dummy prefix token to the input string, but only if text != ""
	if (len text != 0) {
		(found, dummy_prefix) := strinttab->lookup(t.sorted_vocab, " ");
		if (found)
			tokens[n_tokens++] = dummy_prefix;
	}

	for (i := 0; i < len text; i++) {
		c := text[i];

		# append the current character to the buffer
		b[len b] = c;

		# lookup 
		(found, id) := strinttab->lookup(t.sorted_vocab, b);

		if (found)
			# we found this codepoint in vocab, add it as a token
			# TODO: byte fallback encoding
			tokens[n_tokens++] = id;

		b = "";
	}

	# merge the best consecutive pair each iteration, according to the scores in vocab_scores
	while (1) {
		best_score := -1e10;
		best_id := -1;
		best_idx := -1;

		for (i = 0; i < (n_tokens - 1); i++) {
			# check if we can merge the pair (tokens[i], tokens[i + 1])
			(found, id) := strinttab->lookup(t.sorted_vocab,
									 t.vocab[tokens[i]] + t.vocab[tokens[i + 1]]);
			if (found && t.vocab_scores[id] > best_score) {
				# this merge pair exists in vocab! record its score and position
				best_score = t.vocab_scores[id];
				best_id = id;
				best_idx = i;
			}
		}

		if (best_idx == -1)
			# we couldn't find any more pairs to merge, so we're done
			break;

		# merge the consecutie pair (best_idx, best_idx + 1) into new token best_id
		tokens[best_idx] = best_id;
		# delete token at position best_idx + 1, shift the entire sequence back 1
		for (i = best_idx + 1; i < (n_tokens - 1); i++)
			tokens[i] = tokens[i + 1];
		n_tokens--; # token length decreased
	}

	if (eos)
		# add optional EOS (= 2) token, if desired
		tokens[n_tokens++] = 2;
	
	return n_tokens;
}

# ----------------------------------------------------------
# The Sampler, which takes logits and returns a sampled token
# sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
ProbIndex: adt {
	prob: real;
	index: int;
};

Sampler: adt {
	vocab_size: int;
	probindex: array of ProbIndex;
	temperature: real;
	topp: real;
	rng_state: big;
};

sample_argmax(probabilities: array of real): int {
	# return the index that has the highest probability
	max_i := 0;
	max_p := probabilities[0];
	for (i := 1; i < len probabilities; i++) {
		if (probabilities[i] > max_p) {
			max_i = i;
			max_p = probabilities[i];
		}
	}
	return max_i;
}

sample_mult(probabilities: array of real, coin: real): int {
	# sample index from probabilities (they must sum to 1!)
	# coin is a random number in [0, 1), usually from random_real()
	cdf := 0.0;
	for (i := 0; i < len probabilities; i++) {
		cdf += probabilities[i];
		if (coin < cdf)
			return i;
	}
	return len probabilities - 1; # in case of rounding errors
}

DECLARE_SORT(ProbIndex, probindex, prob, <)

sample_topp(probabilities: array of real, topp: real, probindex: array of ProbIndex, coin: real): int {
	# top-p sampling (or "nucleus sampling") samples from the smallest set of
	# tokens that exceed probability topp. This way we never sample tokens that
	# have very low probabilities and are less likely to go "off the rails".
	# coin is a random number in [0, 1), usually from random_real()

	n0 := 0;
	# indices are sorted in descending order of probabilities
	# values smaller than (1 - topp) / (n - 1) cannot be part of the result
	# so for efficiency we crop these out as candidates before sorting
	cutoff := (1.0 - topp) / real (len probabilities - 1);
	for (i := 0; i < len probabilities; i++) {
		if (probabilities[i] >= cutoff) {
			probindex[n0].index = i;
			probindex[n0].prob = probabilities[i];
			n0++;
		}
	}
	probindex = probindex[:n0];
	sort_probindex(probindex);

	# truncate the list where cumulative probability exceeds topp
	cumulative_prob := 0.0;
	last_idx := n0 - 1; # in case of rounding errors, consider all elements
	for (i = 0; i < n0; i++) {
		cumulative_prob += probindex[i].prob;
		if (cumulative_prob > topp) {
			last_idx = i;
			break; # we've exceeded topp by including last_idx
		}
	}

	# sample from the truncated list
	r := coin * cumulative_prob;
	cdf := 0.0;
	for (i = 0; i <= last_idx; i++) {
		cdf += probindex[i].prob;
		if (r < cdf) {
			return probindex[i].index;
		}
	}
	return probindex[last_idx].index; # in case of rounding errors
}

build_sampler(sampler: ref Sampler, vocab_size: int, temperature: real, topp: real, rng_seed: big) {
	sampler.vocab_size = vocab_size;
	sampler.temperature = temperature;
	sampler.topp = topp;
	sampler.rng_state = rng_seed;
	# buffer only used for nucleus sampling; may not need but it's ~small
	sampler.probindex = array[sampler.vocab_size] of ProbIndex;
}

random_int(sampler: ref Sampler): int {
	sampler.rng_state ^= sampler.rng_state >> 12;
	sampler.rng_state ^= sampler.rng_state << 25;
	sampler.rng_state ^= sampler.rng_state >> 27;
	return int ((sampler.rng_state * 16r2545F4914F6CDD1) >> 32);
}

random_real(sampler: ref Sampler): real {
	return (real (random_int(sampler) >> 8)) / 16777216.0;
}

sample(sampler: ref Sampler, logits: array of real): int {
	# sample the token given the logits and some hyperparameters
	next: int;
	if (sampler.temperature == 0.0) {
		# greedy argmax sampling: take the token with the highest probability
		next = sample_argmax(logits);
	} else {
		# apply the temperature to the logits
		for (q := 0; q < len logits; q++)
			logits[q] /= sampler.temperature;
		# apply softmax to the logits to get the probabilities for next token
		softmax(logits);
		# flip a (float) coin (this is our source of entropy for sampling)
		coin := random_real(sampler);
		# we sample from this distribution to get the next token
		if (sampler.topp <= 0.0 || sampler.topp >= 1.0)
			# simply sample from the predicted probability distribution
			next = sample_mult(logits, coin);
		else
			# top-p (nucleus) sampling, clamping the least likely tokens to zero
			next = sample_topp(logits, sampler.topp, sampler.probindex, coin);
	}
	return next;
}

init(ctxt: ref Draw->Context, argv: list of string) {
	# Initialize modules and variables
	sys = load Sys Sys->PATH;
	math = load Math Math->PATH;
	strinttab = load StringIntTab StringIntTab->PATH;

	read_int_buf = array[4] of byte;
	read_int_ibuf = array[1] of int;

	# Parse arguments
	argv = tl argv;
	if (argv == nil)
		raise "fail:noarg";
	tfile := hd argv;

	argv = tl argv;
	prompt := "The sky was blue and";
	if (argv != nil) {
		prompt = hd argv;
	}

	# Load transformer
	t := ref Transformer;
	sys->print("loading transformer...");
	build_transformer(t, tfile);
	sys->print(" loaded!\n");

	# Print model data
	c := t.config;
	sys->print("dim: %d, hidden_dim: %d, n_layers: %d\n",
			  c.dim, c.hidden_dim, c.n_layers);
	sys->print("n_heads: %d, n_kv_heads: %d, vocab_size: %d\n",
			  c.n_heads, c.n_kv_heads, c.vocab_size);
	sys->print("seq_len: %d\n", c.seq_len);

	# Load tokenizer
	tk := ref Tokenizer;
	sys->print("loading tokenizer...");
	build_tokenizer(tk, "tokenizer.bin", c.vocab_size);
	sys->print(" loaded!\n");

	# Test tokenizer
	sys->print("vocab[400]: %s, vocab[401]: %s\n",
			 tk.vocab[400], tk.vocab[401]);
	a := array[100] of int;
	n := encode(tk, prompt, 0, 0, a);
	sys->print("tokenized %s: %d tokens\n", prompt, n);
	for (i := 0; i < n; i++) {
		sys->print("%d %s;", a[i], decode(tk, a[i], a[i]));
	}
	sys->print("\n");

	# Load sampler
	sampler := ref Sampler;
	sys->print("building sampler...");
	build_sampler(sampler, c.vocab_size, 1.0, 0.9, big 42);
	sys->print(" built!\n");

	# Test inference
	prev_token := 0;
	token := 0;
	for (i = 0; i < 40; i++) {
		if (i < n) {
			token = a[i];
		}
		logits := forward(t, token, i);
		token = sample(sampler, logits);
		if (i < n)
			sys->print("%s", decode(tk, prev_token, a[i]));
		else
			sys->print("%s", decode(tk, prev_token, token));
		prev_token = token;
	}
	sys->print("\n");
}
