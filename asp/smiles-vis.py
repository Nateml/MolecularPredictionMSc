import sys
import re
import pprint
import networkx as nx

check_arg = lambda short, long: len(sys.argv) > 1 and ('-' + short in sys.argv[1:] or '--' + long in sys.argv[1:])

if check_arg("h", "help"):
    exit("Usage: clingo 0 smiles.lp <params> | python smiles-vis.py [-h] [-d] [-c]\n"
       + "       clingo 0 smiles.lp smiles-to-edge.lp <params> | python smiles-vis.py [-h] [-d] [-c]\n\n"
       + "Convert answer-sets of logic program into SMILES representation\n"
       + "When edge relation is present, check for isomorphic graphs (using networkx)\n\n"
       + "Options:\n"
       + "-h, --help   : Print this help message and exit\n"
       + "-d, --debug  : Print the model alongside its SMILES string\n"
       + "-c, --color  : Colorful output with atom indices in subscript\n"
       + "               Show cycle facts, in case they are in the input\n\n"
       + "<params>     : Either supply sum formula using c, h, o, n via Clingo's --const option,\n"
       + "             : or use Genmol's to-factbase command to generate input\n\n"
       + "Examples:\n"
       + "clingo 0 smiles.lp smiles-to-edge.lp --const c=7 --const h=14 | python smiles-vis.py -c\n"
       + "clingo 0 smiles.lp <(genmol to-factbase -f C7H14) | python smiles-vis.py -c")

DEBUG = check_arg("d", "debug")
COLOR = check_arg("c", "color")

previous_graphs = []
num_isomorphic = 0

try:
    i = 0
    for line in sys.stdin:
        if "symbol" in line:
            i += 1

            model = dict()
            for symbol in line.split():
                group = re.match("^([^(]+)\\(([^)]*)\\)$", symbol)
                pred = group[1]
                args = group[2].split(",")
                if pred not in model.keys():
                    model[pred] = []
                model[pred].append([arg[1:-1] if arg.startswith('"') else int(arg) if arg[0].isnumeric() else arg for arg in args])

            def _model(pred, default = []):
                return model[pred] if pred in model else default

            if DEBUG:
                print()
                pprint.pp(model)

            def to_smiles(atom, reverse=False):
                branching = next((x[1] for x in _model("branching") if x[0] == atom), 0)
                element_symbol = next((x[1] for x in model["symbol"] if x[0] == atom), '_')
                bond_multiplicity = next(('=#'[x[1]-2] for x in _model("multi_bond") if x[0] == atom), '')
                cycle_start_markers = sorted([x[1] for x in _model("cycle_start") if x[0] == atom])
                cycle_end_markers = sorted([x[1] for x in _model("cycle_end") if x[0] == atom])

                def detect_multi_cycle(cycles, other):
                    cycle_multiplicity = dict()
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

                detect_multi_cycle(cycle_start_markers, _model("cycle_end"))
                cycle_end_multiplicity = detect_multi_cycle(cycle_end_markers, _model("cycle_start"))
                cycle_markers = [f"{c[1]}{c[0]}" for c in sorted([(cc, '') for cc in cycle_start_markers] + [(cc, '=#'[cycle_end_multiplicity[cc]-2]) if cycle_end_multiplicity[cc] > 1 else (cc, '') for cc in cycle_end_markers])]

                next_atom = atom+1
                smiles = []
                for i in range(1, branching+1):
                    (next_atom, smi) = to_smiles(next_atom, next_atom == 2 or (reverse and next_atom == atom+1))
                    smiles.append(smi)

                if COLOR:
                    element_symbol += "\033[38;5;241m" + str(atom).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")) + "\033[0m"
                    cycle_markers = [f"\033[34m{cycle_marker}\033[0m" for cycle_marker in cycle_markers]

                if atom == 1:
                    return f'{smiles[0] if branching > 0 else ""}{element_symbol}{"".join((str(c) for c in cycle_markers))}{"".join([f"({smiles[i]})" for i in range(2, branching)])}{smiles[1] if branching > 1 else ""}'
                if reverse:
                    return (next_atom, f'{smiles[0] if branching > 0 else ""}{element_symbol}{"".join((str(c) for c in cycle_markers))}{"".join([f"({smiles[i]})" for i in range(1, branching)])}{bond_multiplicity}')    
                return (next_atom, f'{bond_multiplicity}{element_symbol}{"".join((str(c) for c in cycle_markers))}{"".join([f"({smiles[i]})" for i in range(1, branching)])}{smiles[0] if branching > 0 else ""}')

            smi = to_smiles(1)

            cyc = [f"cycle({cycle[0]},I={cycle[1]},I'={cycle[2]},L={cycle[2]-cycle[1]},L1={cycle[3]},L2={cycle[4]})" for cycle in _model("cycle")] \
                + [f"cross_cycle({cycle[0]},I={cycle[1]},I'={cycle[2]},L1={cycle[3]},L2={cycle[4]})" for cycle in _model("cross_cycle")]
            smi_len = len(re.sub('\033[^m]*m', '', smi))
            num_atoms = max((x[0] for x in model["symbol"]))
            print(f"{i: >3}: {smi}{' '*(num_atoms*3+5-smi_len)}{', '.join(cyc)}")

            if "edge" in model:
                G = nx.MultiGraph()
                G.add_edges_from(sum([[(edge[0], edge[1])] * edge[2] for edge in model["edge"]], []))
                if DEBUG:
                    print(f"{G}: {G.edges()}")
                is_isomorphic = False
                for k, H in enumerate(previous_graphs):
                    if nx.is_isomorphic(G, H):
                        if not is_isomorphic:
                            num_isomorphic += 1
                            is_isomorphic = True
                        print(f"--> isomorphic to {k+1}")
                previous_graphs.append(G)

        elif not line.startswith("Answer"):
            print(line, end="")

    if len(previous_graphs) > 0:
        print(f"\nisomorphic {num_isomorphic}")

except KeyboardInterrupt:
    print()
