all:V: llama2.dis

clean:V:
	rm -f llama2.b llama2.dis

llama2.b: llama2-m4.b
	m4 -U include -U index -U len llama2-m4.b > llama2.b

llama2.dis: llama2.b llama2-m4.b
	limbo -g llama2.b
