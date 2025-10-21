all:V: llama2.dis

clean:V:
	rm -f llama2.b llama2.dis

llama2.b: llama2.b.m4
	m4 -U include -U index -U len llama2.b.m4 > llama2.b

llama2.dis: llama2.b llama2.b.m4
	limbo -g llama2.b
