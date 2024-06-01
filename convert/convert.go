package convert

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/llm"
)

type Parameters struct {
	Architectures []string `json:"architectures"`
	VocabSize     uint32   `json:"vocab_size"`
}

func (Parameters) KV(v *Vocabulary, svs []*SpecialVocabulary) map[string]any {
	kv := map[string]any{
		"general.file_type":         uint32(1),
		"tokenizer.ggml.pre":        "default",
		"tokenizer.ggml.tokens":     v.Tokens,
		"tokenizer.ggml.scores":     v.Scores,
		"tokenizer.ggml.token_type": v.Types,
	}

	for _, sv := range svs {
		kv[fmt.Sprintf("tokenizer.ggml.%s_token_id", sv.Key())] = uint32(sv.ID)
		kv[fmt.Sprintf("tokenizer.ggml.add_%s_token", sv.Key())] = sv.AddToken
	}

	return kv
}

func (Parameters) SpecialTypes() []string {
	return []string{
		"bos", "eos", "unk", "sep", "pad", "cls", "mask",
	}
}

type Converter interface {
	KV(*Vocabulary, []*SpecialVocabulary) map[string]any
	Tensors([]Tensor) []llm.Tensor
	SpecialTypes() []string

	tensorName(string) (string, error)
}

func Convert(d string, ws io.WriteSeeker) error {
	f, err := os.Open(filepath.Join(d, "config.json"))
	if err != nil {
		return err
	}
	defer f.Close()

	var p Parameters
	if err := json.NewDecoder(f).Decode(&p); err != nil {
		return err
	}

	if len(p.Architectures) < 1 {
		return errors.New("unknown architecture")
	}

	var c Converter
	switch p.Architectures[0] {
	case "LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM":
		c = &llama{}
	case "GemmaForCausalLM":
		c = &gemma{}
	case "PhiForCausalLM", "Phi3ForCausalLM":
		c = &phi{}
	default:
		return errors.New("unsupported architecture")
	}

	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}

	if err := json.NewDecoder(f).Decode(&c); err != nil {
		return err
	}

	v, err := parseVocabulary(d)
	if err != nil {
		return err
	}

	sv, err := parseSpecialVocabulary(d, c.SpecialTypes())
	if err != nil {
		return err
	}

	if vocabSize := int(p.VocabSize); vocabSize > len(v.Tokens) {
		slog.Warn("vocabulary is smaller than expected, padding with dummy tokens", "expect", p.VocabSize, "actual", len(v.Tokens))
		for i := range vocabSize - len(v.Tokens) {
			v.Tokens = append(v.Tokens, fmt.Sprintf("<dummy%05d>", i))
			v.Scores = append(v.Scores, -1)
			v.Types = append(v.Types, tokenTypeUserDefined)
		}
	}

	ts, err := parseTensors(d)
	if err != nil {
		return err
	}

	return llm.WriteGGUF(ws, c.KV(v, sv), c.Tensors(ts))
}

func cutLast(s, sep string) (before, after string, ok bool) {
	i := strings.LastIndex(s, sep)
	if i >= 0 {
		return s[:i], s[i+len(sep):], true
	}
	return s, "", false
}
