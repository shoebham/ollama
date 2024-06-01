package convert

import (
	"cmp"
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"github.com/ollama/ollama/llm"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
)

type llama struct {
	Parameters
	NLayers               uint32  `json:"n_layers"`
	NumHiddenLayers       uint32  `json:"num_hidden_layers"`
	NLayer                uint32  `json:"n_layer"`
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	NCtx                  uint32  `json:"n_ctx"`
	HiddenSize            uint32  `json:"hidden_size"`
	NEmbd                 uint32  `json:"n_embd"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NInner                uint32  `json:"n_inner"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NHead                 uint32  `json:"n_head"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RopeTheta             float32 `json:"rope_theta"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`
	LayerNormEPS          float32 `json:"layer_norm_eps"`
	LayerNormEpsilon      float32 `json:"layer_norm_epsilon"`
	NormEpsilon           float32 `json:"norm_epsilon"`
	NumLocalExperts       uint32  `json:"num_local_experts"`
	NumExpertsPerToken    uint32  `json:"num_experts_per_tok"`
}

func (p *llama) KV(v *Vocabulary, svs []*SpecialVocabulary) map[string]any {
	kv := p.Parameters.KV(v, svs)
	kv["general.architecture"] = "llama"
	kv["general.name"] = "llama"

	kv["llama.block_count"] = cmp.Or(p.NLayers, p.NumHiddenLayers, p.NLayer)
	kv["llama.vocab_size"] = p.VocabSize

	if contextLength := cmp.Or(p.MaxPositionEmbeddings, p.NCtx); contextLength > 0 {
		kv["llama.context_length"] = contextLength
	}

	if embeddingLength := cmp.Or(p.HiddenSize, p.NEmbd); embeddingLength > 0 {
		kv["llama.embedding_length"] = cmp.Or(p.HiddenSize, p.NEmbd)
	}

	if feedForwardLength := cmp.Or(p.IntermediateSize, p.NInner); feedForwardLength > 0 {
		kv["llama.feed_forward_length"] = cmp.Or(p.IntermediateSize, p.NInner)
	}

	if headCount := cmp.Or(p.NumAttentionHeads, p.NHead); headCount > 0 {
		kv["llama.attention.head_count"] = cmp.Or(p.NumAttentionHeads, p.NHead)
		kv["llama.rope.dimension_count"] = p.HiddenSize / headCount
	}

	if p.NumKeyValueHeads > 0 {
		kv["llama.attention.head_count_kv"] = p.NumKeyValueHeads
	}

	if p.RopeTheta > 0 {
		kv["llama.attention.rope_freq_base"] = p.RopeTheta
	}

	if p.RMSNormEPS > 0 {
		kv["llama.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	}

	if layerNormEpsilon := cmp.Or(p.LayerNormEPS, p.LayerNormEpsilon, p.NormEpsilon); layerNormEpsilon > 0 {
		kv["llama.attention.layer_norm_epsilon"] = layerNormEpsilon
	}

	if p.NumLocalExperts > 0 {
		kv["llama.attention.expert_count"] = p.NumLocalExperts
	}

	if p.NumExpertsPerToken > 0 {
		kv["llama.attention.expert_used_count"] = p.NumExpertsPerToken
	}

	if len(v.Merges) > 0 {
		kv["tokenizer.ggml.merges"] = v.Merges
	}

	kv["tokenizer.ggml.model"] = "llama"
	return kv
}

func (p *llama) Tensors(ts []Tensor) []llm.Tensor {
	var out []llm.Tensor
	for _, t := range ts {
		name, err := p.tensorName(t.Name())
		if err != nil {
			slog.Debug("skipping unknown tensor", "name", t.Name())
			continue
		}

		if strings.HasSuffix(name, "attn_q.weight") ||
			strings.HasSuffix(name, "attn_k.weight") {

			t.SetRepacker(p.repack)

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

func (p *llama) tensorName(n string) (string, error) {
	n, suffix, ok := cutLast(n, ".")
	if !ok || suffix != "weight" {
		return "", fmt.Errorf("invalid tensor name: %q", n)
	}

	var parts []string
	if n == "lm_head" {
		parts = append(parts, "output")
		return strings.Join(append(parts, suffix), "."), nil
	}

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

func (p *llama) repack(name string, data []float32, shape []uint64) ([]float32, error) {
	var dims []int
	for _, dim := range shape {
		if dim != 0 {
			dims = append(dims, int(dim))
		}
	}

	var heads uint32
	if strings.HasSuffix(name, "q_proj.weight") {
		heads = p.NumAttentionHeads
	} else if strings.HasSuffix(name, "k_proj.weight") {
		heads = cmp.Or(p.NumKeyValueHeads, p.NumAttentionHeads)
	} else {
		return nil, fmt.Errorf("unknown tensor for repack: %s", name)
	}

	n := tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
	if err := n.Reshape(append([]int{int(heads), 2, dims[0] / int(heads) / 2}, dims[1:]...)...); err != nil {
		return nil, err
	}

	if err := n.T(0, 2, 1, 3); err != nil {
		return nil, err
	}

	if err := n.Reshape(dims...); err != nil {
		return nil, err
	}

	if err := n.Transpose(); err != nil {
		return nil, err
	}

	ts, err := native.SelectF32(n, 1)
	if err != nil {
		return nil, err
	}

	var f32s []float32
	for _, t := range ts {
		f32s = append(f32s, t...)
	}

	return f32s, nil
}
