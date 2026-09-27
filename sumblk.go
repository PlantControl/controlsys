package controlsys

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"

	"plantcontrol.org/v1/gonum/mat"
)

type tokenKind int

const (
	tokIdent tokenKind = iota
	tokNumber
	tokEquals
	tokPlus
	tokMinus
	tokStar
	tokSemicolon
	tokEOF
)

type token struct {
	kind tokenKind
	text string
	num  float64
}

type signalTerm struct {
	name  string
	coeff float64
}

type equation struct {
	output string
	terms  []signalTerm
}

func tokenize(expr string) ([]token, error) {
	var tokens []token
	runes := []rune(expr)
	i := 0
	for i < len(runes) {
		r := runes[i]
		if unicode.IsSpace(r) {
			i++
			continue
		}
		switch r {
		case '=':
			tokens = append(tokens, token{kind: tokEquals, text: "="})
			i++
		case '+':
			tokens = append(tokens, token{kind: tokPlus, text: "+"})
			i++
		case '-':
			tokens = append(tokens, token{kind: tokMinus, text: "-"})
			i++
		case '*':
			tokens = append(tokens, token{kind: tokStar, text: "*"})
			i++
		case ';':
			tokens = append(tokens, token{kind: tokSemicolon, text: ";"})
			i++
		default:
			if unicode.IsLetter(r) || r == '_' {
				start := i
				for i < len(runes) && (unicode.IsLetter(runes[i]) || unicode.IsDigit(runes[i]) || runes[i] == '_') {
					i++
				}
				tokens = append(tokens, token{kind: tokIdent, text: string(runes[start:i])})
			} else if unicode.IsDigit(r) || r == '.' {
				start := i
				for i < len(runes) && (unicode.IsDigit(runes[i]) || runes[i] == '.') {
					i++
				}
				s := string(runes[start:i])
				n, err := strconv.ParseFloat(s, 64)
				if err != nil {
					return nil, fmt.Errorf("sumblk: invalid number %q: %w", s, ErrInvalidExpression)
				}
				tokens = append(tokens, token{kind: tokNumber, text: s, num: n})
			} else {
				return nil, fmt.Errorf("sumblk: unexpected character %q: %w", string(r), ErrInvalidExpression)
			}
		}
	}
	tokens = append(tokens, token{kind: tokEOF})
	return tokens, nil
}

type sumblkParser struct {
	tokens []token
	pos    int
}

func (p *sumblkParser) peek() token {
	return p.tokens[p.pos]
}

func (p *sumblkParser) advance() token {
	t := p.tokens[p.pos]
	p.pos++
	return t
}

func (p *sumblkParser) expect(k tokenKind) (token, error) {
	t := p.peek()
	if t.kind != k {
		return t, fmt.Errorf("sumblk: expected %d, got %q: %w", k, t.text, ErrInvalidExpression)
	}
	p.advance()
	return t, nil
}

func (p *sumblkParser) parseTerm() (signalTerm, error) {
	coeff := 1.0
	sign := 1.0
	for p.peek().kind == tokPlus || p.peek().kind == tokMinus {
		if p.advance().kind == tokMinus {
			sign = -sign
		}
	}

	if p.peek().kind == tokNumber {
		t := p.advance()
		if _, err := p.expect(tokStar); err != nil {
			return signalTerm{}, fmt.Errorf("sumblk: expected '*' after number: %w", ErrInvalidExpression)
		}
		coeff = t.num
	}

	ident, err := p.expect(tokIdent)
	if err != nil {
		return signalTerm{}, fmt.Errorf("sumblk: expected signal name: %w", ErrInvalidExpression)
	}
	return signalTerm{name: ident.text, coeff: sign * coeff}, nil
}

func (p *sumblkParser) parseEquation() (equation, error) {
	out, err := p.expect(tokIdent)
	if err != nil {
		return equation{}, fmt.Errorf("sumblk: expected output name: %w", ErrInvalidExpression)
	}
	if _, err := p.expect(tokEquals); err != nil {
		return equation{}, err
	}

	first, err := p.parseTerm()
	if err != nil {
		return equation{}, err
	}
	terms := []signalTerm{first}

	for p.peek().kind == tokPlus || p.peek().kind == tokMinus {
		t, err := p.parseTerm()
		if err != nil {
			return equation{}, err
		}
		terms = append(terms, t)
	}
	return equation{output: out.text, terms: terms}, nil
}

func (p *sumblkParser) parse() ([]equation, error) {
	var eqs []equation
	eq, err := p.parseEquation()
	if err != nil {
		return nil, err
	}
	eqs = append(eqs, eq)

	for p.peek().kind == tokSemicolon {
		p.advance()
		if p.peek().kind == tokEOF {
			break
		}
		eq, err := p.parseEquation()
		if err != nil {
			return nil, err
		}
		eqs = append(eqs, eq)
	}

	if p.peek().kind != tokEOF {
		return nil, fmt.Errorf("sumblk: unexpected token %q: %w", p.peek().text, ErrInvalidExpression)
	}
	return eqs, nil
}

func parseEquations(tokens []token) ([]equation, error) {
	return (&sumblkParser{tokens: tokens}).parse()
}

type parsedSumBlock struct {
	equations []equation
	outputs   []string
	inputs    []string
	signals   []string
	widths    map[string]int
}

func parseSumBlock(expr string) (*parsedSumBlock, error) {
	trimmed := strings.TrimSpace(expr)
	if trimmed == "" {
		return nil, fmt.Errorf("sumblk: empty expression: %w", ErrInvalidExpression)
	}
	tokens, err := tokenize(trimmed)
	if err != nil {
		return nil, err
	}
	eqs, err := parseEquations(tokens)
	if err != nil {
		return nil, err
	}
	parsed := &parsedSumBlock{equations: eqs}
	if err := parsed.collectSignals(); err != nil {
		return nil, err
	}
	return parsed, nil
}

func (p *parsedSumBlock) collectSignals() error {
	outputSet := make(map[string]bool)
	for _, eq := range p.equations {
		if outputSet[eq.output] {
			return fmt.Errorf("sumblk: duplicate output %q: %w", eq.output, ErrInvalidExpression)
		}
		outputSet[eq.output] = true
		p.outputs = append(p.outputs, eq.output)
	}

	inputSet := make(map[string]bool)
	for _, eq := range p.equations {
		for _, t := range eq.terms {
			if !inputSet[t.name] {
				inputSet[t.name] = true
				p.inputs = append(p.inputs, t.name)
			}
		}
	}

	p.signals = make([]string, 0, len(p.outputs)+len(p.inputs))
	p.signals = append(p.signals, p.outputs...)
	for _, inp := range p.inputs {
		if !outputSet[inp] {
			p.signals = append(p.signals, inp)
		}
	}
	return nil
}

func (p *parsedSumBlock) applyWidths(widths []int) error {
	p.widths = make(map[string]int, len(p.signals))
	switch len(widths) {
	case 0:
		for _, s := range p.signals {
			p.widths[s] = 1
		}
	case 1:
		for _, s := range p.signals {
			p.widths[s] = widths[0]
		}
	default:
		if len(widths) != len(p.signals) {
			return fmt.Errorf("sumblk: got %d widths for %d signals: %w", len(widths), len(p.signals), ErrInvalidExpression)
		}
		for i, s := range p.signals {
			p.widths[s] = widths[i]
		}
	}
	return nil
}

func (p *parsedSumBlock) directMatrix() *mat.Dense {
	totalRows := 0
	for _, o := range p.outputs {
		totalRows += p.widths[o]
	}

	totalCols := 0
	inputColStart := make(map[string]int, len(p.inputs))
	for _, inp := range p.inputs {
		inputColStart[inp] = totalCols
		totalCols += p.widths[inp]
	}

	D := mat.NewDense(totalRows, totalCols, nil)
	rowOff := 0
	for _, eq := range p.equations {
		w := p.widths[eq.output]
		merged := make(map[string]float64)
		for _, t := range eq.terms {
			merged[t.name] += t.coeff
		}
		for name, coeff := range merged {
			col0 := inputColStart[name]
			wIn := p.widths[name]
			diag := min(wIn, w)
			for k := range diag {
				D.Set(rowOff+k, col0+k, coeff)
			}
		}
		rowOff += w
	}
	return D
}

func expandSignalNames(signals []string, widths map[string]int) []string {
	var expanded []string
	for _, signal := range signals {
		w := widths[signal]
		if w == 1 {
			expanded = append(expanded, signal)
			continue
		}
		for k := 1; k <= w; k++ {
			expanded = append(expanded, fmt.Sprintf("%s(%d)", signal, k))
		}
	}
	return expanded
}

func SumBlk(expr string, widths ...int) (*System, error) {
	parsed, err := parseSumBlock(expr)
	if err != nil {
		return nil, err
	}
	if err := parsed.applyWidths(widths); err != nil {
		return nil, err
	}

	sys, err := NewGain(parsed.directMatrix(), 0)
	if err != nil {
		return nil, err
	}
	sys.OutputName = expandSignalNames(parsed.outputs, parsed.widths)
	sys.InputName = expandSignalNames(parsed.inputs, parsed.widths)
	return sys, nil
}
