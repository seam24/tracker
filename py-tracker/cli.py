import os
import argparse
import json
from glob import iglob

from . import mutate

def parse_arguments():
    parser = argparse.ArgumentParser(prog="cmutate")
    parser.add_argument("-t", "--target", required=True, type=str)
    parser.add_argument("-s", "--scope", default=None, type=str)
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="Write mutants as JSON lines to this file instead of only printing.",
    )
    return parser.parse_args()


def guess_package_roots():
    this_dir = os.getcwd().split(os.sep)[-1]
    output = []

    if os.path.isdir("./lib"):
        output.append("./lib")
    if os.path.isdir("./src"):
        output.append("./src")
    if os.path.isdir(this_dir):
        output.append(this_dir)
    if os.path.isdir(this_dir.replace("-", "_")):
        output.append(this_dir.replace("-", "_"))
    if os.path.isdir(this_dir.replace("_", "-")):
        output.append(this_dir.replace("_", "-"))
    if os.path.isfile(this_dir + ".py"):
        output.append(this_dir + ".py")
    
    if len(output) == 0: raise FileNotFoundError(
        "Could not find package roots in the current directory"
    )
    return output


def walk_all_paths(start_prefix = None):

    if start_prefix is None:
        package_roots = guess_package_roots()
    else:
        package_roots = [start_prefix]

    for root in package_roots:
        if os.path.isfile(root):
            yield root
            continue

        for python_file in iglob(os.path.join(root, "**", "*.py"), recursive = True):
            yield python_file


def walk_all_modules(start_prefix = None):
    for file_path in walk_all_paths(start_prefix = start_prefix):
        module = file_path
        if os.path.basename(module) == "__init__.py":
            module = os.path.dirname(module)
        module = module.replace(os.sep, ".").replace(".py", "")
        yield module, file_path


# Module matching ---------------------

def _match_module_name(pattern, module_name):
    pattern = pattern.split(".") or []
    module  = module_name.split(".") or []

    positions = [(0, 0)]
    while len(positions) > 0:
        pi, mi = positions.pop(0)

        if pi >= len(pattern) and mi >= len(module):
            return True
        
        if pi >= len(pattern) or mi >= len(module): continue

        if pattern[pi] == module[mi] or pattern[pi] == "*":
            positions.append((pi + 1, mi + 1))
        elif pattern[pi] == "**":
            positions.append((pi + 1, mi + 1))
            positions.append((pi, mi + 1))
    
    return False


# --------------------------------------

def main():
    config = parse_arguments()

    if os.path.exists(config.target):
        modules = walk_all_modules(start_prefix=config.target)
        config.target = "*"
    else:
        modules = walk_all_modules()

    if config.target != "*":
        modules = filter(lambda module_file: _match_module_name(config.target, module_file[0]),
                         modules)

    all_mutants = []
    num_mutants = 0 
    for module, file_path in modules:
        with open(file_path, "r") as f:
            content = f.read()

        lines = content.splitlines(True)  # keep newline characters
        line_offsets = [0]  # line_offsets[i] = offset at start of line i
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))

        # for mutant in mutate(content, lang = "python"):
        #     precise_module = module
        #     if mutant.scope:
        #         precise_module = ".".join([precise_module, mutant.scope])
        #
        #     if config.scope:
        #         if not _match_module_name(config.scope, precise_module):
        #             continue
        #
        #     print(f"[#{num_mutants}] Mutation")
        #     print(f"- [{mutant.op_type}] {precise_module}")
        #     print("-"*80)
        #     print(mutant.unified_diff(content,
        #                               fromfile=file_path,
        #                               tofile=file_path))
        #     print("-"*80 + "\n\n")
        #     num_mutants += 1

        for mutant in mutate(content, lang="python"):
            precise_module = module
            if mutant.scope:
                precise_module = ".".join([precise_module, mutant.scope])

            if config.scope:
                if not _match_module_name(config.scope, precise_module):
                    continue

            start_line, start_col, end_line, end_col = mutant.source_pos

            # Convert (line, col) → absolute character offsets
            start_pos = line_offsets[start_line] + start_col
            end_pos = line_offsets[end_line] + end_col

            record = {
                "mutant_id": num_mutants,
                "file": file_path,
                "module": precise_module,
                "op_type": mutant.op_type,
                "scope": mutant.scope,
                "mutation_start_pos": start_pos,
                "mutation_end_pos": end_pos,
                "mutant_value": mutant.target_text,
            }

            all_mutants.append(record)
            num_mutants += 1


    # --------------------------------------
    # WRITE ONLY IF -o / --output SPECIFIED
    # --------------------------------------
    if config.output:
        with open(config.output, "w") as out_f:
            json.dump(all_mutants, out_f, indent=2)
        print(f"Saved {len(all_mutants)} mutants to {config.output}")
    else:
        print(f"Generated {len(all_mutants)} mutants (no output file requested).")



if __name__ == "__main__":
    main()