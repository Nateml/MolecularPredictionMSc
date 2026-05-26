# Description: Interfaces with the ASP program

import subprocess
import clingo
import math
import typing
import argparse
from pathlib import Path
from predicates.interface import PredictionInterface, PredictionEntry
from predicates.compiler import compile_asp_constraint

_ASP_DIR = Path(__file__).resolve().parent / "src" / "asp"

# Elements parametrized in smiles.lp (C/H/N/O/P/S only).
_ASP_ELEMENTS = {"C", "H", "N", "O", "P", "S"}


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


def solve(asp_code: list[str], atom_counts: int, output_list_of_smiles: bool = False, debug=False, timeout: int = 0) -> typing.Tuple[str, int]:
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

    flags = [
        '0',
        f'-c c={atom_counts["num_carbons"]}',
        f'-c h={atom_counts["num_hydrogens"]}',
        f'-c n={atom_counts["num_nitrogens"]}',
        f'-c o={atom_counts["num_oxygens"]}',
        f'-c s={atom_counts["num_sulfurs"]}',
        f'-c p={atom_counts["num_phosphorus"]}',
    ]
    if timeout > 0:
        flags.append(f'-t {timeout}')   # wall-clock time limit (clingo 5.x)
    ctl = clingo.Control(flags)


    k = min_main_chain_len(sum(atom_counts.values()) - atom_counts['num_hydrogens'])
    program = f"min_main_chain_len({k}).\n" + "\n".join(asp_code)

    # Save program to a file for debugging
    debug_path = str(_ASP_DIR / "debug" / "debug_program.lp")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(program)

    #ctl.add("base", [], program)
    ctl.load(debug_path)

    # Load the ASP program
    ctl.load(str(_ASP_DIR / "smiles.lp"))

    # Save the full ASP program to a file for debugging
    with open(str(_ASP_DIR / "debug" / "full_program.lp"), "w", encoding="utf-8") as f:
        f.write(program)

    # Ground
    ctl.ground([("base", [])])

    model_count = 0
    output = ""
    timed_out = False
    with ctl.solve(yield_=True) as handle:
        for i, model in enumerate(handle, start=1):
            model_count += 1
            output += f"Answer: {i}\n"
            atoms = model.symbols(shown=True)
            output += " ".join(map(str, atoms)) + "\n"
        timed_out = not handle.get().exhausted

    if timed_out and timeout > 0:
        import sys as _sys
        print(f"Warning: ASP solve timed out after {timeout}s; results may be incomplete.", file=_sys.stderr)

    output += "SATISFIABLE\n"

    # Visualise with smiles-vis.py
    result = subprocess.run(
        ['python', str(_ASP_DIR / 'smiles-vis.py')],
        input=output,
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(f"smiles-vis.py failed with error: {result.stderr}")

    visualisations = result.stdout.split("\n")
    asp_output = output.split("\n")


    # Remove unwanted lines
    visualisations = visualisations[:-2]
    visualisations = [line.split(':')[1].strip() for line in visualisations]
    asp_output = asp_output[1:-2:2]

    if output_list_of_smiles:
        # return "\n".join(visualisations), model_count
        # Return a list of SMILES strings
        return visualisations, model_count

    final_output = ""
    i = 0
    for vis, asp in zip(visualisations, asp_output):
        final_output += f"CANDIDATE {i}:\n{vis}\n{asp}\n\n"
        i += 1

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
    print(f"ASP constraints:\n{chr(10).join(asp_constraints)}\n")
    print(f"Atom counts: {atom_counts}\n")

    output, num_candidates = solve(asp_constraints, atom_counts, output_list_of_smiles=False, debug=True)  # Set to True to get a copy/pastable list of SMILES strings (useful for pasting into a visualisation tool)
    print(f"Found {num_candidates} candidates.\n")
    print(output)

if __name__ == "__main__":
    main()