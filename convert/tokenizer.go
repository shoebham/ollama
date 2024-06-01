package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
)

const (
	_ int32 = iota
	tokenTypeNormal
	tokenTypeUnknown
	tokenTypeControl
	tokenTypeUserDefined
	tokenTypeUnused
	tokenTypeByte
)

type tokenizer struct {
	Version     string  `json:"version"`
	AddedTokens []token `json:"added_tokens"`
	Model       struct {
		Type   string         `json:"type"`
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	} `json:"model"`

	PreTokenizer struct {
		PreTokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
}

type token struct {
	ID          int    `json:"id"`
	Content     string `json:"content"`
	Special     bool   `json:"special"`
	UserDefined bool
}

type Vocabulary struct {
	Tokens []string
	Scores []float32
	Types  []int32
	Merges []string
}

func parseVocabularyFromTokenizer(p string) (*Vocabulary, error) {
	f, err := os.Open(filepath.Join(p, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var bpe tokenizer
	if err := json.NewDecoder(f).Decode(&bpe); err != nil {
		return nil, err
	}

	var tokens []token
	for k, v := range bpe.Model.Vocab {
		tokens = append(tokens, token{
			ID:      v,
			Content: k,
		})
	}

	for _, t := range bpe.AddedTokens {
		t.UserDefined = true
		tokens = append(tokens, t)
	}

	slices.SortFunc(tokens, func(i, j token) int {
		return cmp.Compare(i.ID, j.ID)
	})

	var v Vocabulary
	for _, t := range tokens {
		v.Tokens = append(v.Tokens, t.Content)
		v.Scores = append(v.Scores, float32(t.ID))

		switch {
		case t.Special:
			v.Types = append(v.Types, tokenTypeControl)
		case t.UserDefined:
			v.Types = append(v.Types, tokenTypeUserDefined)
		default:
			v.Types = append(v.Types, tokenTypeNormal)
		}
	}

	v.Merges = bpe.Model.Merges
	return &v, nil
}

func parseVocabulary(d string) (*Vocabulary, error) {
	patterns := map[string]func(string) (*Vocabulary, error){
		"tokenizer.model": parseSentencePiece,
		"tokenizer.json":  parseVocabularyFromTokenizer,
	}

	for pattern, parseFn := range patterns {
		matches, err := filepath.Glob(filepath.Join(d, pattern))
		if err != nil {
			return nil, err
		}

		if len(matches) > 0 {
			return parseFn(d)
		}
	}

	return nil, errors.New("unknown tensor format")
}

type SpecialVocabulary struct {
	Type     string
	ID       int
	Content  string
	AddToken bool
}

func (sv SpecialVocabulary) Key() string {
	switch t := sv.Type; t {
	case "bos", "eos":
		return t
	case "pad":
		return "padding"
	case "unk":
		return "unknown"
	}

	panic("unknown special vocabulary type")
}

func parseSpecialVocabulary(d string, types []string) ([]*SpecialVocabulary, error) {
	return parseSpecialVocabularyFromTokenizer(d, types)
}

func parseSpecialVocabularyFromTokenizer(d string, types []string) ([]*SpecialVocabulary, error) {
	tokens := make(map[string]token)
	if f, err := os.Open(filepath.Join(d, "tokenizer.json")); errors.Is(err, os.ErrNotExist) {
	} else if err != nil {
		return nil, err
	} else {
		defer f.Close()

		var t tokenizer
		if err := json.NewDecoder(f).Decode(&t); err != nil {
			return nil, err
		}

		for _, t := range t.AddedTokens {
			tokens[t.Content] = t
		}
	}

	f, err := os.Open(filepath.Join(d, "tokenizer_config.json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, err
	} else if err != nil {
		return nil, err
	}
	defer f.Close()

	var p map[string]json.RawMessage
	if err := json.NewDecoder(f).Decode(&p); err != nil {
		return nil, err
	}

	var svs []*SpecialVocabulary
	for _, t := range types {
		sv := SpecialVocabulary{Type: t}
		if bts, ok := p[fmt.Sprintf("add_%s_token", t)]; ok {
			if err := json.Unmarshal(bts, &sv.AddToken); err != nil {
				return nil, err
			}
		}

		if bts, ok := p[fmt.Sprintf("%s_token", t)]; ok {
			var content string
			if err := json.Unmarshal(bts, &content); err != nil {
				var mm map[string]any
				if err := json.Unmarshal(bts, &mm); err != nil {
					continue
				}

				content, ok = mm["content"].(string)
				if !ok {
					continue
				}
			}

			sv.Content = content
		}

		if id, ok := tokens[sv.Content]; ok {
			sv.ID = id.ID
			svs = append(svs, &sv)
		}
	}

	return svs, nil
}
