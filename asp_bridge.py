# Description: Interfaces with the ASP program

import os
import re
import clingo
import math
import typing
import argparse
from pathlib import Path

# Benchmarked sweet spot on 16-CPU machine; cap at cpu_count for portability.
_CLINGO_THREADS = min(6, os.cpu_count() or 1)
from predicates.interface import PredictionInterface, PredictionEntry
from predicates.compiler import compile_asp_constraint

_ASP_DIR = Path(__file__).resolve().parent / "src" / "asp"

# Elements parametrized in smiles.lp (C/H/N/O/P/S only).
_ASP_ELEMENTS = {"C", "H", "N", "O", "P", "S"}

# ---------------------------------------------------------------------------
# Degree of Unsaturation (DoU) filter
#
# DoU = (2C + 2 + N - H) / 2  (for C/H/N-only; O and S don't affect DoU)
#
# Certain structural predicates are only satisfiable when DoU meets a minimum:
#   DoU >= 1: any ring or double bond (ring, C=O, C=C, C=N, N=O, S=O, ...)
#   DoU >= 2: triple bond (C≡C, C≡N)
#
# When a constraint forces one of these predicates to be true (:- not <pred>.)
# but the formula has insufficient DoU, the ASP solve will always return 0
# candidates. We detect this cheaply and skip the solve entirely.
# ---------------------------------------------------------------------------

_DoU_GE1_PREDICATES = frozenset({
    "has_ring",
    "has_carbonyl",
    "has_c_c_double_bond",
    "has_c_n_double_bond",
    "has_n_o_double_bond",
    "has_s_o_double_bond",
    "has_s_double_bond",
    "has_n_c_o",
    "has_o_c_o",
    "has_n_a_n",
    "has_o_aa_o",
    "has_n_aa_n",
    "has_n_aa_o",
})

_DoU_GE2_PREDICATES = frozenset({
    "has_c_c_triple_bond",
    "has_c_n_triple_bond",
})


def degrees_of_unsaturation(atom_counts: dict) -> float:
    """Return DoU for a formula. Non-integer or negative means invalid formula."""
    numerator = (
        2 * atom_counts.get("num_carbons", 0)
        + 2
        + atom_counts.get("num_nitrogens", 0)
        - atom_counts.get("num_hydrogens", 0)
    )
    return numerator / 2


def _min_required_dou(asp_code: list[str]) -> int:
    """Scan compiled ASP constraints for predicates that imply a minimum DoU."""
    for line in asp_code:
        for pred in _DoU_GE2_PREDICATES:
            if f"not {pred}" in line:
                return 2
    for line in asp_code:
        for pred in _DoU_GE1_PREDICATES:
            if f"not {pred}" in line:
                return 1
    return 0


def is_asp_supported(smiles: str) -> tuple[bool, str]:
    """Check whether a molecule is representable by the ASP solver.

    Returns (True, "") if supported, or (False, reason) otherwise.
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise RuntimeError("rdkit required for is_asp_supported()")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "invalid SMILES"

    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        # smiles.lp only parametrizes c/h/n/o/p/s — no valence rules for anything else
        if sym not in _ASP_ELEMENTS:
            return False, f"unsupported element: {sym}"
        # smiles.lp hardcodes formal_charge(0); EI-MS analytes are neutral anyway
        if atom.GetFormalCharge() != 0:
            return False, f"formal charge {atom.GetFormalCharge():+d} on {sym}"
        # isotope labels are RDKit metadata only; ASP matches by symbol+valence,
        # so [2H] and H are identical to it — ground truth would never match
        if atom.GetIsotope() != 0:
            return False, f"isotope label on {sym}"
        # ASP fills spare valence slots with implicit H; a radical leaves one
        # unpaired, so the solver would generate a different (hydrogenated) molecule
        if atom.GetNumRadicalElectrons() > 0:
            return False, f"radical electrons on {sym}"

    # ASP builds one graph via main chain + branches + rings; no mechanism
    # exists to generate two disconnected fragments
    if len(Chem.GetMolFrags(mol)) > 1:
        return False, f"disconnected molecule ({len(Chem.GetMolFrags(mol))} components)"

    return True, ""

# I have extracted this function from smiles.lp because my clingo doesn't have python support,
# so the ASP program can't use #script(python) to call it directly.
def min_main_chain_len(n):
    return min(
        2 * math.ceil(math.log((n - 1)/2 + 1, 3)) + 1,
        2 * math.ceil(math.log(n + 1, 3))
    )


def _model_to_smiles(model_str: str) -> str:
    """Convert a raw clingo model string to a SMILES string.

    Ported from smiles-vis.py to avoid subprocess overhead per solve call.
    """
    model: dict = {}
    for symbol in model_str.split():
        group = re.match(r"^([^(]+)\(([^)]*)\)$", symbol)
        if not group:
            continue
        pred = group[1]
        args = [
            arg[1:-1] if arg.startswith('"') else int(arg) if arg[0].isnumeric() else arg
            for arg in group[2].split(",")
        ]
        model.setdefault(pred, []).append(args)

    def _m(pred):
        return model.get(pred, [])

    def to_smiles(atom, reverse=False):
        branching = next((x[1] for x in _m("branching") if x[0] == atom), 0)
        element_symbol = next((x[1] for x in model["smiles_symbol"] if x[0] == atom), '_')
        bond_multiplicity = next(('=#'[x[1]-2] for x in _m("multi_bond") if x[0] == atom), '')
        cycle_start_markers = sorted([x[1] for x in _m("cycle_start") if x[0] == atom])
        cycle_end_markers = sorted([x[1] for x in _m("cycle_end") if x[0] == atom])

        def detect_multi_cycle(cycles, other):
            cycle_multiplicity = {}
            multi_cycle = []
            for c in cycles:
                if c not in multi_cycle:
                    cycle_multiplicity[c] = 1
                    c_atom = next((x[0] for x in other if x[1] == c), -1)
                    c_other_markers = [x[1] for x in other if x[0] == c_atom]
                    for cc in c_other_markers:
                        if cc in cycles and cc != c:
                            multi_cycle.append(cc)
                            cycle_multiplicity[c] += 1
            for cc in multi_cycle:
                cycles.remove(cc)
            return cycle_multiplicity

        detect_multi_cycle(cycle_start_markers, _m("cycle_end"))
        cycle_end_multiplicity = detect_multi_cycle(cycle_end_markers, _m("cycle_start"))
        cycle_markers = [
            f"{c[1]}{c[0]}" for c in sorted(
                [(cc, '') for cc in cycle_start_markers] +
                [(cc, '=#'[cycle_end_multiplicity[cc]-2]) if cycle_end_multiplicity[cc] > 1 else (cc, '')
                 for cc in cycle_end_markers]
            )
        ]

        next_atom = atom + 1
        smiles = []
        for i in range(1, branching + 1):
            (next_atom, smi) = to_smiles(next_atom, next_atom == 2 or (reverse and next_atom == atom + 1))
            smiles.append(smi)

        cm = "".join(str(c) for c in cycle_markers)
        if atom == 1:
            return (
                f'{smiles[0] if branching > 0 else ""}{element_symbol}{cm}'
                f'{"".join(f"({smiles[i]})" for i in range(2, branching))}{smiles[1] if branching > 1 else ""}'
            )
        if reverse:
            return (next_atom,
                f'{smiles[0] if branching > 0 else ""}{element_symbol}{cm}'
                f'{"".join(f"({smiles[i]})" for i in range(1, branching))}{bond_multiplicity}'
            )
        return (next_atom,
            f'{bond_multiplicity}{element_symbol}{cm}'
            f'{"".join(f"({smiles[i]})" for i in range(1, branching))}{smiles[0] if branching > 0 else ""}'
        )

    return to_smiles(1)


def solve(asp_code: list[str], atom_counts: int, output_list_of_smiles: bool = False, debug=False, timeout: int = 0, canonicalize: bool = False) -> typing.Tuple[str, int]:
    """
    Runs the ASP program with the given input and returns the results.

    Args:
        asp_code (list[str]): A list of strings representing additional ASP code
        atom_counts (dict): Atom count constraints for the formula
        output_list_of_smiles (bool): If True, return list of SMILES strings
        debug (bool): If True, print debug information
        timeout (int): Wall-clock time limit in seconds for the solve phase (0 = no limit)

    Returns:
        list: A list of candidate molecules returned by the ASP program
    """
    dou = degrees_of_unsaturation(atom_counts)
    if dou < 0 or dou != int(dou) or dou < _min_required_dou(asp_code):
        return ([] if output_list_of_smiles else "", 0)

    flags = [
        '0',
        f'-c c={atom_counts["num_carbons"]}',
        f'-c h={atom_counts["num_hydrogens"]}',
        f'-c n={atom_counts["num_nitrogens"]}',
        f'-c o={atom_counts["num_oxygens"]}',
        f'-c s={atom_counts["num_sulfurs"]}',
        f'-c p={atom_counts["num_phosphorus"]}',
        f'-t {_CLINGO_THREADS}'
    ]
    ctl = clingo.Control(flags)

    k = min_main_chain_len(sum(atom_counts.values()) - atom_counts['num_hydrogens'])
    program = f"min_main_chain_len({k}).\n" + "\n".join(asp_code)

    if debug:
        debug_path = str(_ASP_DIR / "debug" / "debug_program.lp")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(program)

    ctl.add("base", [], program)
    ctl.load(str(_ASP_DIR / "smiles.lp"))
    ctl.ground([("base", [])])

    models_raw = []
    def on_model(model):
        models_raw.append(" ".join(map(str, model.symbols(shown=True))))

    with ctl.solve(async_=True, on_model=on_model) as handle:
        finished = handle.wait(timeout if timeout > 0 else None)
        if not finished:
            import sys as _sys
            print(f"Warning: ASP solve timed out after {timeout}s; results may be incomplete.", file=_sys.stderr)
            handle.cancel()

    model_count = len(models_raw)
    visualisations = [_model_to_smiles(m) for m in models_raw]

    if output_list_of_smiles:
        if canonicalize:
            from rdkit import Chem
            canonical = []
            for smi in visualisations:
                mol = Chem.MolFromSmiles(smi)
                canonical.append(Chem.MolToSmiles(mol) if mol else smi)
            return canonical, model_count
        return visualisations, model_count

    final_output = ""
    for i, (vis, asp) in enumerate(zip(visualisations, models_raw)):
        final_output += f"CANDIDATE {i}:\n{vis}\n{asp}\n\n"

    return final_output, model_count

def main():
    # Run the ASP program and print the results
    parser = argparse.ArgumentParser(
        description='ASP bridge for molecular prediction',
        epilog='Usage examples:\n'
               '  python asp_bridge.py --carbons 3 --hydrogens 6 --oxygens 1\n'
               '  python asp_bridge.py --pred-ids 154 156 --pred-sets 1 2 --pred-sets 3 4\n'
               '  python asp_bridge.py --carbons 5 --nitrogens 1 --pred-ids 154',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-c', '--carbons', type=int, default=0, help='Number of carbons')
    parser.add_argument('-y', '--hydrogens', type=int, default=0, help='Number of hydrogens')
    parser.add_argument('-o', '--oxygens', type=int, default=0, help='Number of oxygens')
    parser.add_argument('-n', '--nitrogens', type=int, default=0, help='Number of nitrogens')
    parser.add_argument('-s', '--sulfurs', type=int, default=0, help='Number of sulfurs')
    parser.add_argument('-p', '--phosphorus', type=int, default=0, help='Number of phosphorus atoms')
    parser.add_argument('--pred-ids', type=int, nargs='+', default=[], help='Prediction IDs')
    parser.add_argument('--pred-sets', type=int, nargs='+', action='append', help='Prediction sets for each ID (can specify multiple sets per ID)')
    parser.add_argument('--show-constraints', action='store_true', help="Whether or not to print the additional ASP code generated")
    parser.add_argument('--canonicalize', action='store_true', help='Canonicalize output SMILES via RDKit')
    parser.add_argument('--return-list', action='store_true', help='Return a raw list of candidate molecules')
    args = parser.parse_args()

    # If all atom counts are zero, use a default formula
    if all(count == 0 for count in [args.carbons, args.hydrogens, args.oxygens, args.nitrogens, args.sulfurs, args.phosphorus]):
        print("No atom counts specified, using default formula C3H6O1")
        args.carbons = 3
        args.hydrogens = 6
        args.oxygens = 1

    predictions = {}
    
    # Handle prediction IDs and their associated sets
    if args.pred_sets:
        for i, pred_id in enumerate(args.pred_ids):
            if i < len(args.pred_sets):
                pred_set = args.pred_sets[i]
                predictions[pred_id] = PredictionEntry(pred_set=pred_set if isinstance(pred_set, list) else [pred_set])
            else:
                predictions[pred_id] = PredictionEntry(pred_set=[1])
    else:
        # Default behavior: use default pred_set for all IDs
        for pred_id in args.pred_ids:
            predictions[pred_id] = PredictionEntry(pred_set=[1])
    
    predictions['num_carbons'] = args.carbons
    predictions['num_hydrogens'] = args.hydrogens
    predictions['num_oxygens'] = args.oxygens
    predictions['num_nitrogens'] = args.nitrogens
    predictions['num_sulfurs'] = args.sulfurs
    predictions['num_phosphorus'] = args.phosphorus

    # Example predictions
    # predictions = {
        # 154: PredictionEntry(pred_set=[1], alpha=0.05, proba=[0.98]),
        # 'num_carbons': args.carbons,
        # 'num_hydrogens': args.hydrogens,
        # 'num_nitrogens': args.nitrogens,
        # 'num_oxygens': args.oxygens,
    # }
    
    interface = PredictionInterface.model_validate(predictions)
    asp_constraints, atom_counts = compile_asp_constraint(interface)
    if args.show_constraints:
        print(f"ASP constraints:\n{chr(10).join(asp_constraints)}\n")
        print(f"Atom counts: {atom_counts}\n")

    output, num_candidates = solve(asp_constraints, atom_counts, output_list_of_smiles=args.return_list, debug=True, canonicalize=args.canonicalize)

    print(f"Found {num_candidates} candidates.\n")
    print(output)

if __name__ == "__main__":
    main()
