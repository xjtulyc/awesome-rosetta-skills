---
name: formal-logic
description: >
  Use this Skill for formal logic and automated reasoning: Z3 SMT solver for
  propositional/first-order logic, modal logic S5, SAT problem encoding, and Lean4 proof verification.
tags:
  - philosophy
  - formal-logic
  - Z3
  - SMT-solver
  - theorem-proving
  - modal-logic
version: "1.0.0"
authors:
  - name: awesome-rosetta-skills contributors
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - z3-solver>=4.12
    - sympy>=1.12
    - numpy>=1.23
last_updated: "2026-03-18"
status: stable
---

# Formal Logic and Automated Reasoning with Z3

> **TL;DR** — Use the Z3 SMT solver for propositional and first-order logic, encode
> constraint satisfaction puzzles (Zebra/Einstein riddle, N-Queens) as SAT problems,
> reason about modal logic S5 validity using Kripke frame semantics, and verify logical
> equivalences and tautologies with sympy.

---

## When to Use

Use this Skill when you need to:

- Verify whether a logical formula is a tautology, contradiction, or contingency
- Solve constraint puzzles (logic grids, Sudoku, scheduling, graph coloring) with SAT/SMT
- Encode first-order logic (FOL) theories and check satisfiability
- Validate philosophical arguments by formalizing them in propositional or modal logic
- Check logical equivalences and argue about modal operators (necessity □, possibility ◇)

Do **not** use this Skill for:

- Probabilistic reasoning under uncertainty (use PyMC or pgmpy)
- Natural language argument parsing without formalization (see argument-mapping Skill)
- Full higher-order logic or dependent type systems (use Coq or Agda)

---

## Background

Z3 is Microsoft Research's high-performance SMT (Satisfiability Modulo Theories) solver.
It handles propositional logic, linear arithmetic (integers and reals), bit vectors,
arrays, and quantified first-order logic. Key functions:

| Z3 API | Description |
|---|---|
| `Bool(name)` | Declare a Boolean variable |
| `Int(name)`, `Real(name)` | Arithmetic variables |
| `And`, `Or`, `Not`, `Implies`, `Xor` | Logical connectives |
| `ForAll`, `Exists` | First-order quantifiers |
| `Function(name, sort, sort)` | Uninterpreted function declaration |
| `Solver.add(constraint)` | Assert a formula |
| `Solver.check()` | Returns `sat`, `unsat`, or `unknown` |
| `Solver.model()` | Extract satisfying assignment |

Modal logic S5 is the standard system for epistemic and alethic necessity.
A formula is **S5-valid** if it holds in every world of every Kripke frame where
the accessibility relation R is an equivalence relation (reflexive, symmetric, transitive).

sympy's `logic` module (`to_cnf`, `satisfiable`, `tautology`) provides symbolic
manipulation without external solvers, useful for smaller formulas and teaching purposes.

---

## Environment Setup

```bash
# Create Python environment
conda create -n formal-logic python=3.11 -y
conda activate formal-logic

# Install dependencies
pip install "z3-solver>=4.12" "sympy>=1.12" "numpy>=1.23"

# Verify Z3
python -c "import z3; print(z3.get_version_string())"

# Optional: Lean 4 for proof verification
# Install via elan (Lean version manager):
# curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
# lake new myproject && cd myproject
```

---

## Core Workflow

### Step 1 — Z3 Propositional Puzzles: Einstein/Zebra Riddle

```python
from z3 import (
    Bool, Int, Solver, And, Or, Not, Implies, Distinct, sat,
    Function, IntSort, BoolSort, ForAll, Exists, Real, If
)


def solve_einstein_riddle() -> dict:
    """
    Solve the Einstein/Zebra riddle using Z3 integer constraints.

    Five houses in a row; each has a unique nationality, colour, pet, drink, cigarette.
    Constraints encode all 15 clues. Return the satisfying assignment.

    Returns:
        Dict mapping house positions (1-5) to their attribute values.
    """
    s = Solver()

    # Attributes: each value 1-5 represents the house position
    Brit = Int("Brit"); Swede = Int("Swede"); Dane = Int("Dane")
    Norwegian = Int("Norwegian"); German = Int("German")

    Red = Int("Red"); Green = Int("Green"); White = Int("White")
    Yellow = Int("Yellow"); Blue = Int("Blue")

    Tea = Int("Tea"); Coffee = Int("Coffee"); Milk = Int("Milk")
    Beer = Int("Beer"); Water = Int("Water")

    PallMall = Int("PallMall"); Dunhill = Int("Dunhill")
    Blends = Int("Blends"); BlueMaster = Int("BlueMaster"); Prince = Int("Prince")

    Dog = Int("Dog"); Birds = Int("Birds"); Cats = Int("Cats")
    Horse = Int("Horse"); Fish = Int("Fish")

    all_vars = [
        Brit, Swede, Dane, Norwegian, German,
        Red, Green, White, Yellow, Blue,
        Tea, Coffee, Milk, Beer, Water,
        PallMall, Dunhill, Blends, BlueMaster, Prince,
        Dog, Birds, Cats, Horse, Fish,
    ]

    # Domain constraints: each attribute occupies a position 1-5
    for v in all_vars:
        s.add(v >= 1, v <= 5)

    # Uniqueness within each category
    s.add(Distinct(Brit, Swede, Dane, Norwegian, German))
    s.add(Distinct(Red, Green, White, Yellow, Blue))
    s.add(Distinct(Tea, Coffee, Milk, Beer, Water))
    s.add(Distinct(PallMall, Dunhill, Blends, BlueMaster, Prince))
    s.add(Distinct(Dog, Birds, Cats, Horse, Fish))

    # Clues
    s.add(Brit == Red)                           # 1. Brit lives in Red house
    s.add(Swede == Dog)                          # 2. Swede keeps Dogs
    s.add(Dane == Tea)                           # 3. Dane drinks Tea
    s.add(Green == White - 1)                    # 4. Green is left of White
    s.add(Green == Coffee)                       # 5. Green owner drinks Coffee
    s.add(PallMall == Birds)                     # 6. PallMall smoker keeps Birds
    s.add(Yellow == Dunhill)                     # 7. Yellow owner smokes Dunhill
    s.add(Milk == 3)                             # 8. Middle house drinks Milk
    s.add(Norwegian == 1)                        # 9. Norwegian lives in first house
    s.add(Or(Blends - Cats == 1, Cats - Blends == 1))   # 10. Blends next to Cats
    s.add(Or(Horse - Dunhill == 1, Dunhill - Horse == 1))  # 11. Horse next to Dunhill
    s.add(BlueMaster == Beer)                    # 12. BlueMaster smoker drinks Beer
    s.add(German == Prince)                      # 13. German smokes Prince
    s.add(Or(Norwegian - Blue == 1, Blue - Norwegian == 1))  # 14. Norwegian next to Blue
    s.add(Or(Blends - Water == 1, Water - Blends == 1))   # 15. Blends next to Water

    result = s.check()
    if result != sat:
        return {"error": "No solution found"}

    model = s.model()
    assignment = {str(v): model.eval(v).as_long() for v in all_vars}

    # Determine fish owner (the riddle question)
    fish_pos = assignment["Fish"]
    owner = next(
        name for name in ["Brit", "Swede", "Dane", "Norwegian", "German"]
        if assignment[name] == fish_pos
    )
    assignment["_fish_owner"] = owner
    return assignment


def check_tautology(formula_str: str) -> dict:
    """
    Check whether a propositional formula is a tautology using Z3.

    A formula F is a tautology iff ¬F is unsatisfiable.

    Args:
        formula_str: A description of the formula (handled via Z3 API in caller).

    Returns:
        Dict with is_tautology (bool), is_satisfiable (bool), is_contradiction (bool).
    """
    # This function demonstrates the tautology-check pattern.
    # Callers construct the Z3 formula and pass it directly.
    raise NotImplementedError(
        "Build your formula with Z3 API (Bool, And, Or, ...) then call check_formula_type()."
    )


def check_formula_type(formula) -> dict:
    """
    Determine whether a Z3 formula is a tautology, contradiction, or contingency.

    Args:
        formula: Any Z3 BoolRef expression.

    Returns:
        Dict with keys: is_tautology, is_contradiction, is_contingent,
        satisfying_model (dict or None).
    """
    # Check satisfiability of the formula
    s_sat = Solver()
    s_sat.add(formula)
    sat_result = s_sat.check()

    # Check satisfiability of its negation (for tautology)
    s_neg = Solver()
    s_neg.add(Not(formula))
    neg_result = s_neg.check()

    is_satisfiable = sat_result == sat
    neg_satisfiable = neg_result == sat

    model = None
    if is_satisfiable:
        raw_model = s_sat.model()
        model = {str(d): str(raw_model[d]) for d in raw_model.decls()}

    return {
        "is_tautology": is_satisfiable and not neg_satisfiable,
        "is_contradiction": not is_satisfiable,
        "is_contingent": is_satisfiable and neg_satisfiable,
        "is_satisfiable": is_satisfiable,
        "satisfying_model": model,
    }


def check_logical_equivalence(formula_a, formula_b) -> bool:
    """
    Check whether two Z3 formulas are logically equivalent.

    A ≡ B iff (A ↔ B) is a tautology, i.e., ¬(A ↔ B) is unsatisfiable.

    Args:
        formula_a: First Z3 BoolRef.
        formula_b: Second Z3 BoolRef.

    Returns:
        True if the formulas are equivalent, False otherwise.
    """
    from z3 import Xor
    biconditional = Not(Xor(formula_a, formula_b))
    result = check_formula_type(biconditional)
    return result["is_tautology"]
```

### Step 2 — First-Order Logic and Function Declaration

```python
def fol_example_socrates() -> dict:
    """
    Encode and verify the classic syllogism in Z3 first-order logic.

    Premises:
      1. All men are mortal: ForAll x. Man(x) → Mortal(x)
      2. Socrates is a man: Man(Socrates)
    Conclusion:
      Socrates is mortal: Mortal(Socrates)

    Returns:
        Dict with is_valid (bool) and explanation.
    """
    from z3 import DeclareSort, Const, Function, BoolSort, ForAll, Implies, And, Not, Solver, sat

    # Declare domain sort
    Person = DeclareSort("Person")

    # Declare predicates
    Man = Function("Man", Person, BoolSort())
    Mortal = Function("Mortal", Person, BoolSort())

    # Declare constants
    Socrates = Const("Socrates", Person)
    x = Const("x", Person)

    # Premises
    all_men_mortal = ForAll(x, Implies(Man(x), Mortal(x)))
    socrates_is_man = Man(Socrates)

    # Conclusion (negate to check validity)
    conclusion = Mortal(Socrates)

    # Valid iff premises ∧ ¬conclusion is unsat
    s = Solver()
    s.add(all_men_mortal)
    s.add(socrates_is_man)
    s.add(Not(conclusion))

    result = s.check()
    is_valid = result != sat  # UNSAT means the argument is valid

    return {
        "is_valid": is_valid,
        "explanation": "The syllogism is valid: premises entail the conclusion."
        if is_valid else "Unexpected: premises do not entail conclusion.",
    }


def solve_nqueens(n: int = 8) -> list[int]:
    """
    Solve the N-Queens problem using Z3 constraint satisfaction.

    Each queen is placed at column q[i] in row i.
    Constraints: all columns distinct, no diagonal attacks.

    Args:
        n: Board size (n×n).

    Returns:
        List of column positions [q_0, q_1, ..., q_{n-1}], or empty list if unsat.
    """
    queens = [Int(f"q_{i}") for i in range(n)]
    s = Solver()

    # Domain
    for q in queens:
        s.add(q >= 0, q < n)

    # All columns distinct
    s.add(Distinct(queens))

    # No diagonal attacks
    for i in range(n):
        for j in range(i + 1, n):
            s.add(queens[i] - queens[j] != i - j)
            s.add(queens[i] - queens[j] != j - i)

    if s.check() == sat:
        model = s.model()
        return [model.eval(queens[i]).as_long() for i in range(n)]
    return []
```

### Step 3 — Modal Logic S5 Validity Check

```python
def check_s5_validity(
    formula_fn,
    n_worlds: int = 4,
    verbose: bool = False,
) -> dict:
    """
    Check whether a modal formula is valid in S5 using a brute-force Kripke model.

    In S5, the accessibility relation R is an equivalence relation (reflexive,
    symmetric, transitive), which means every world accesses every other world.
    A formula is S5-valid iff it holds in ALL worlds of ALL possible valuations.

    This function tests validity by checking whether the negation is satisfiable
    in any Kripke model with `n_worlds` worlds (complete graph accessibility for S5).

    Args:
        formula_fn: A function (worlds, R, V, w) → bool, where worlds is a list
                    of world indices, R is the accessibility relation set of pairs,
                    V is a dict {prop: set of worlds where prop is true}, and w is
                    the current evaluation world.
        n_worlds:   Number of worlds in the Kripke frame.
        verbose:    If True, print counterexample valuations when found.

    Returns:
        Dict with is_valid (bool), counterexample (dict or None).
    """
    import itertools

    worlds = list(range(n_worlds))
    # S5: universal accessibility (every world sees every world)
    R = {(w1, w2) for w1 in worlds for w2 in worlds}

    # Generate all possible valuations for one propositional variable p
    # (2^n_worlds valuations; extend for multiple variables as needed)
    for truth_combo in itertools.product([True, False], repeat=n_worlds):
        V = {"p": {w for w, val in enumerate(truth_combo) if val}}

        # Check formula holds in all worlds
        for w in worlds:
            holds = formula_fn(worlds, R, V, w)
            if not holds:
                if verbose:
                    print(f"Counterexample found: V(p)={V['p']}, world={w}")
                return {
                    "is_valid": False,
                    "counterexample": {
                        "V_p": list(V["p"]),
                        "falsified_at_world": w,
                    },
                }

    return {"is_valid": True, "counterexample": None}


def modal_necessity(worlds, R, V, w, prop="p"):
    """□p: p holds in all accessible worlds (R[w])."""
    accessible = {v for (u, v) in R if u == w}
    return all(v in V.get(prop, set()) for v in accessible)


def modal_possibility(worlds, R, V, w, prop="p"):
    """◇p: p holds in some accessible world."""
    accessible = {v for (u, v) in R if u == w}
    return any(v in V.get(prop, set()) for v in accessible)
```

---

## Advanced Usage

### sympy Propositional Logic

```python
from sympy.logic.boolalg import to_cnf, Equivalent, Implies as Imp
from sympy.logic.inference import satisfiable
from sympy import symbols


def sympy_tautology_check(formula) -> bool:
    """
    Check if a sympy Boolean formula is a tautology.

    A formula is a tautology iff its negation is unsatisfiable.

    Args:
        formula: sympy Boolean expression.

    Returns:
        True if tautology, False otherwise.
    """
    from sympy import Not
    neg = Not(formula)
    result = satisfiable(neg)
    return result is False


def demonstrate_de_morgan() -> None:
    """Show that De Morgan's laws are tautologies using sympy."""
    A, B = symbols("A B")
    # ¬(A ∧ B) ↔ (¬A ∨ ¬B)
    law1 = Equivalent(~(A & B), (~A | ~B))
    # ¬(A ∨ B) ↔ (¬A ∧ ¬B)
    law2 = Equivalent(~(A | B), (~A & ~B))

    print(f"De Morgan law 1 is tautology: {sympy_tautology_check(law1)}")
    print(f"De Morgan law 2 is tautology: {sympy_tautology_check(law2)}")
    print(f"CNF of ¬(A → B): {to_cnf(~Imp(A, B))}")
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `z3.z3types.Z3Exception: unknown` | Quantified formula with non-linear arithmetic | Use linear arithmetic only; avoid multiplying Z3 variables |
| Z3 `check()` returns `unknown` on timeout | Undecidable theory fragment | Set `s.set("timeout", 5000)` (ms); simplify constraints |
| `Distinct()` with one variable | Z3 requires ≥2 arguments | Wrap in a list check before calling `Distinct()` |
| sympy `satisfiable()` returns `{}` not False | Tautology returns empty model | Empty dict `{}` means tautology (no variable assignment falsifies it) |
| N-Queens unsat for n=2 or n=3 | No valid placement exists | n=2 and n=3 are mathematically unsolvable; try n≥4 |
| FOL quantifier over large domain | Z3 may time out | Reduce domain size or use bounded model checking |

---

## External Resources

- Z3 Python API documentation: <https://z3prover.github.io/api/html/namespacez3py.html>
- Z3 tutorial (rise4fun): <https://microsoft.github.io/z3guide/>
- sympy Logic module: <https://docs.sympy.org/latest/modules/logic.html>
- Lean 4 theorem prover: <https://leanprover.github.io/>
- SEP article on Modal Logic: <https://plato.stanford.edu/entries/logic-modal/>
- SEP article on Automated Reasoning: <https://plato.stanford.edu/entries/reasoning-automated/>

---

## Examples

### Example 1 — Einstein Riddle and Tautology Checks

```python
# Solve the Einstein/Zebra riddle
solution = solve_einstein_riddle()
print(f"Fish is owned by: {solution['_fish_owner']}")
print(f"Fish house position: {solution['Fish']}")

# Check a tautology: A → (B → A) (Axiom K of propositional logic)
A = Bool("A")
B = Bool("B")
formula_k = Implies(A, Implies(B, A))
result = check_formula_type(formula_k)
print(f"\nFormula A → (B → A) is tautology: {result['is_tautology']}")

# Check equivalence: ¬(A → B) ↔ (A ∧ ¬B)
lhs = Not(Implies(A, B))
rhs = And(A, Not(B))
equiv = check_logical_equivalence(lhs, rhs)
print(f"¬(A → B) ≡ (A ∧ ¬B): {equiv}")

# Solve 8-Queens
queens_solution = solve_nqueens(8)
print(f"\n8-Queens solution: {queens_solution}")
```

### Example 2 — FOL Syllogism and Modal S5 Validity

```python
# Verify the Socrates syllogism in FOL
fol_result = fol_example_socrates()
print(f"Socrates syllogism valid: {fol_result['is_valid']}")
print(fol_result["explanation"])

# Check S5 axiom: □p → p  (T axiom: what is necessary is actual)
def t_axiom(worlds, R, V, w):
    """□p → p: if p is necessary at w, then p holds at w."""
    box_p = modal_necessity(worlds, R, V, w, prop="p")
    p_at_w = w in V.get("p", set())
    return (not box_p) or p_at_w  # material implication

t_result = check_s5_validity(t_axiom, n_worlds=4, verbose=True)
print(f"\nS5 T-axiom (□p → p) is valid: {t_result['is_valid']}")

# Check S5 axiom: ◇p → □◇p  (5 axiom: what is possible is necessarily possible)
def axiom_5(worlds, R, V, w):
    """◇p → □◇p"""
    poss_p = modal_possibility(worlds, R, V, w, prop="p")
    accessible = {v for (u, v) in R if u == w}
    box_poss_p = all(
        modal_possibility(worlds, R, V, v, prop="p") for v in accessible
    )
    return (not poss_p) or box_poss_p

a5_result = check_s5_validity(axiom_5, n_worlds=4)
print(f"S5 Axiom 5 (◇p → □◇p) is valid: {a5_result['is_valid']}")

# Demonstrate De Morgan's laws
demonstrate_de_morgan()
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — Z3 propositional/FOL, Einstein riddle, N-Queens, S5 modal logic, sympy tautology checks |
