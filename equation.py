"""DigitChallenge logic: fill every gap with a distinct digit 1-9 so the equation holds.

Rules (stated by the test): each operand is 1-9, no two operands share a value.
Grammar (standard precedence, parentheses allowed):
    expr   := term (('+'|'-') term)*
    term   := factor (('*'|'/') factor)*
    factor := NUMBER | '(' expr ')' | '-' factor
A NUMBER is a run of adjacent digit/gap tokens, so "1?" is a two-digit number.
"""
from fractions import Fraction
from itertools import permutations, product

GAP = None
DIGITS = range(1, 10)
_NORMALISE = {
    "x": "*", "X": "*", "×": "*", "✕": "*", "·": "*", "∗": "*",
    "÷": "/", ":": "/",
    "＋": "+", "－": "-", "−": "-", "—": "-", "–": "-",
    "（": "(", "）": ")", "＝": "=",
    "?": GAP, "？": GAP, "_": GAP, "□": GAP,
}


def normalise(tokens):
    return [GAP if t is GAP else _NORMALISE.get(t, t) for t in tokens]


def tokenize(text: str):
    """'( ? × ? )－ ? ＋ ? =12' -> ['(', GAP, '*', GAP, ')', '-', GAP, '+', GAP, '=', '1', '2']"""
    return normalise([c for c in text if not c.isspace()])


def _bin(op, a, b):
    def f(g):
        x, y = a(g), b(g)
        if x is None or y is None:
            return None
        if op == "+":
            return x + y
        if op == "-":
            return x - y
        if op == "*":
            return x * y
        return None if y == 0 else x / y
    return f


class _Parser:
    """Compiles one side of the equation into f(gap_values) -> Fraction | None."""

    def __init__(self, tokens, first_gap_index):
        self.toks, self.i, self.gap = tokens, 0, first_gap_index

    def parse(self):
        fn = self._expr()
        if self.i != len(self.toks):
            raise ValueError(f"unexpected token {self.toks[self.i]!r}")
        return fn

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else "<end>"

    def _binary(self, ops, sub):
        fn = sub()
        while self._peek() in ops:
            op = self.toks[self.i]
            self.i += 1
            fn = _bin(op, fn, sub())
        return fn

    def _expr(self):
        return self._binary(("+", "-"), self._term)

    def _term(self):
        return self._binary(("*", "/"), self._factor)

    def _factor(self):
        t = self._peek()
        if t == "(":
            self.i += 1
            fn = self._expr()
            if self._peek() != ")":
                raise ValueError("missing ')'")
            self.i += 1
            return fn
        if t == "-":
            self.i += 1
            inner = self._factor()
            return lambda g: None if (v := inner(g)) is None else -v
        parts = []
        while self.i < len(self.toks) and (self.toks[self.i] is GAP or str(self.toks[self.i]).isdigit()):
            if self.toks[self.i] is GAP:
                parts.append(self.gap)
                self.gap += 1
            else:
                parts.append(str(self.toks[self.i]))
            self.i += 1
        if not parts:
            raise ValueError(f"expected a number at {t!r}")
        if all(isinstance(p, str) for p in parts):
            const = Fraction(int("".join(parts)))
            return lambda g: const
        return lambda g: Fraction(int("".join(str(g[p]) if isinstance(p, int) else p for p in parts)))


def compile_equation(tokens):
    """-> (n_gaps, holds(values) -> bool). Raises ValueError on malformed input."""
    tokens = normalise(tokens)
    if tokens.count("=") != 1:
        raise ValueError("need exactly one '='")
    i = tokens.index("=")
    left, right = tokens[:i], tokens[i + 1:]
    lf = _Parser(left, 0).parse()
    rf = _Parser(right, left.count(GAP)).parse()

    def holds(values):
        lv, rv = lf(values), rf(values)
        return lv is not None and rv is not None and lv == rv
    return left.count(GAP) + right.count(GAP), holds


def solve(tokens, pool=DIGITS, unique=True, limit=8):
    """Fillings (tuples, gaps left->right) that satisfy the equation, up to `limit`."""
    try:
        n, holds = compile_equation(tokens)
    except (ValueError, IndexError):
        return []
    if n == 0:
        return []
    digits = sorted(set(pool))
    combos = permutations(digits, n) if unique else product(digits, repeat=n)
    found = []
    for combo in combos:
        if holds(combo):
            found.append(combo)
            if len(found) >= limit:
                break
    return found
