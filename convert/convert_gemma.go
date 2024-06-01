package convert

import (
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/llm"
)

type gemma struct {
	Parameters
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`
	HeadDim               uint32  `json:"head_dim"`
}

func (p *gemma) KV(v *Vocabulary, svs []*SpecialVocabulary) map[string]any {
	kv := p.Parameters.KV(v, svs)
	kv["general.architecture"] = "gemma"
	kv["general.name"] = "gemma"
	kv["gemma.context_length"] = p.MaxPositionEmbeddings
	kv["gemma.embedding_length"] = p.HiddenSize
	kv["gemma.block_count"] = p.HiddenLayers
	kv["gemma.feed_forward_length"] = p.IntermediateSize
	kv["gemma.attention.head_count"] = p.NumAttentionHeads
	kv["gemma.attention.head_count_kv"] = p.NumKeyValueHeads
	kv["gemma.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	kv["gemma.attention.key_length"] = p.HeadDim
	kv["gemma.attention.value_length"] = p.HeadDim
	kv["tokenizer.ggml.model"] = "llama"
	kv["tokenizer.ggml.eot_token_id"] = uint32(107)
	kv["tokenizer.ggml.middle_token_id"] = uint32(68)
	kv["tokenizer.ggml.prefix_token_id"] = uint32(67)
	kv["tokenizer.ggml.suffix_token_id"] = uint32(69)
	return kv
}

func (p *gemma) Tensors(ts []Tensor) []llm.Tensor {
	var out []llm.Tensor
	for _, t := range ts {
		name, err := p.tensorName(t.Name())
		if err != nil {
			slog.Debug("skipping unknown tensor", "name", t.Name())
			continue
		}

		if strings.HasSuffix(name, "_norm.weight") {
			t.SetRepacker(p.addOne)
		}

		out = append(out, llm.Tensor{
			Name:     name,
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (p *gemma) tensorName(n string) (string, error) {
	n, suffix, ok := cutLast(n, ".")
	if !ok || suffix != "weight" {
		return "", fmt.Errorf("invalid tensor name: %q", n)
	}

	var parts []string
	prefix, n, ok := strings.Cut(n, ".")
	if !ok {
		return "", fmt.Errorf("invalid tensor name: %q", n)
	}

	switch prefix {
	case "model":
		switch n {
		case "embed_tokens":
			parts = append(parts, "token_embd")
		case "norm":
			parts = append(parts, "output_norm")
		default:
			prefix, n, ok := strings.Cut(n, ".")
			if !ok || prefix != "layers" {
				return "", fmt.Errorf("invalid tensor name: %q", n)
			}

			layer, n, ok := strings.Cut(n, ".")
			if !ok {
				return "", fmt.Errorf("invalid tensor name: %q", n)
			}

			if _, err := strconv.Atoi(layer); err != nil {
				return "", fmt.Errorf("invalid tensor name: %q", n)
			}

			parts = append(parts, "blk", layer)

			switch n {
			case "input_layernorm":
				parts = append(parts, "attn_norm")
			case "self_attn.q_proj":
				parts = append(parts, "attn_q")
			case "self_attn.k_proj":
				parts = append(parts, "attn_k")
			case "self_attn.v_proj":
				parts = append(parts, "attn_v")
			case "self_attn.o_proj":
				parts = append(parts, "attn_output")
			case "mlp.gate_proj":
				parts = append(parts, "ffn_gate")
			case "mlp.down_proj":
				parts = append(parts, "ffn_down")
			case "mlp.up_proj":
				parts = append(parts, "ffn_up")
			case "post_attention_layernorm":
				parts = append(parts, `ffn_norm`)
			default:
				return "", fmt.Errorf("invalid tensor name: %q", n)
			}
		}
	default:
		return "", fmt.Errorf("invalid tensor name: %q", n)
	}

	return strings.Join(append(parts, suffix), "."), nil
}

func (*gemma) addOne(_ string, data []float32, shape []uint64) ([]float32, error) {
	n := tensor.New(tensor.WithShape(int(shape[0])), tensor.WithBacking(data))
	ones := tensor.Ones(tensor.Float32, int(shape[0]))

	n, err := n.Add(ones)
	if err != nil {
		return nil, err
	}

	ts, err := native.SelectF32(n, 0)
	if err != nil {
		return nil, err
	}

	var f32s []float32
	for _, t := range ts {
		f32s = append(f32s, t...)
	}

	return f32s, nil
}
