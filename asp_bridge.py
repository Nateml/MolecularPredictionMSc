# Description: Interfaces with the ASP program

import subprocess
import sys
import clingo
import timeit
import math
import textwrap


# I have extracted this function from smiles.lp because my clingo doesn't have python support,
# so the ASP program can't use #script(python) to call it directly.
def min_main_chain_len(n):
    return min(
        2 * math.ceil(math.log((n - 1)/2 + 1, 3)) + 1,
        2 * math.ceil(math.log(n + 1, 3))
    )

def solve(c, h, n, o):
    """
    Runs the ASP program with the given input and returns the results.

    Args:
        c (int): The number of carbon atoms
        h (int): The number of hydrogen atoms
        n (int): The number of nitrogen atoms
        o (int): The number of oxygen atoms

    Returns:
        list: A list of candidate molecules returned by the ASP program
    """

    # Set the constants
    # Add the function for calculating the minimum main chain length
    num_atoms = n + c + o  # exclude hydrogens
    k = min_main_chain_len(num_atoms)

    ctl = clingo.Control([
    "0",                 # enumerate all models
    f"-c c={c}",
    f"-c h={h}",
    f"-c n={n}",
    f"-c o={o}",
    ])

    program = textwrap.dedent(f"""
    min_main_chain_len({k}).
    """)
    ctl.add("base", [], program)

    # Load the ASP program
    ctl.load("asp/smiles.lp")

    # Ground
    ctl.ground([("base", [])])

    model_count = 0
    output = ""
    with ctl.solve(yield_=True) as handle:
        for i, model in enumerate(handle, start=1):
            model_count += 1

            output += f"Answer: {i}\n"

            atoms = model.symbols(shown=True)
            output += " ".join(map(str, atoms)) + "\n"

    output += "SATISFIABLE\n"

    # Visualise with smiles-vis.py
    result = subprocess.run(
        ['python', 'asp/smiles-vis.py', '-c'],
        input=output,
        capture_output=True,
        text=True,
        check=True
    )

    return result.stdout, model_count

def main():
    # Run the ASP program and print the results

    example = {"c": 7, "h": 16, "n": 0, "o": 0} 
    output, num_candidates = solve(**example)
    print(f"Found {num_candidates} candidates for C{example['c']}H{example['h']}N{example['n']}O{example['o']}\n")
    print(output)

if __name__ == "__main__":
    main()