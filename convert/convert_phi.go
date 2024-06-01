package convert

import "github.com/ollama/ollama/llm"

type phi struct {
	Parameters
}

func (p *phi) KV(v *Vocabulary, svs []*SpecialVocabulary) map[string]any {
	kv := p.Parameters.KV(v, svs)
	kv["general.architecture"] = "phi"
	kv["tokenizer.ggml.model"] = "llama"
	return kv
}

func (p *phi) Tensors(ts []Tensor) []llm.Tensor {
	var out []llm.Tensor
	return out
}

func (p *phi) tensorName(name string) (string, error) {
	return name, nil
}
