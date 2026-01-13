# gumtree_mutant_mapper.py
import subprocess
import tempfile
import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Union, Iterable, Tuple, Any
import uuid


# region Utils & Helpers
class InvalidOutputError(Exception):
    """Raise invalid output error"""
    pass


class MutTrackFailureError(Exception):
    """Raise invalid output error"""
    pass


def char_index_to_line(content: str) -> Dict[int, int]:
    """
    Build a mapping from each character index in a string to its line number. Works with Unix (`\\n`), Windows
    (`\\r\\n`), and old Mac (`\\r`) line endings.  The full source text. Character indices start at 0.
    Line numbers start at 1.
    Reference: https://github.com/seam24/tracker/blob/main/utils/parseUtils.py
    """
    chr_to_line: Dict[int, int] = {}
    line_num = 1
    idx = 0
    while idx < len(content):
        chr_to_line[idx] = line_num
        if content[idx] == '\r':
            # Handle Windows CRLF line breaks
            if idx + 1 < len(content) and content[idx + 1] == '\n':  # '\r\n'
                chr_to_line[idx + 1] = line_num  # map '\n' to same line
                idx += 1  # skip the '\n'
            line_num += 1
        elif content[idx] == '\n':
            # Handle Unix LF line breaks
            line_num += 1
        idx += 1
    return chr_to_line


def gumdiff(src1: str, src2: str, lang: str = ".py", commands: list[str] = None) -> str:
    """
    Run GumTree AST differencing inside the official gumtreediff Docker container.

    Parameters
    ----------
    src1 : str
        The source code of version A (old code).
    src2 : str
        The source code of version B (new code).
    lang : str, optional
        File extension to indicate the programming language. Defaults to ".py".
        Examples: ".py" (Python), ".java" (Java), ".c" (C).
    commands : list[str], optional
        Extra command arguments to pass to GumTree.
        - To get a JSON diff: ["textdiff", "-f", "JSON"]
        - To get an XML diff: ["axmldiff"]

    Returns
    -------
    str
        The raw output of GumTree (JSON, XML, or text depending on `commands`).

    Raises
    ------
    RuntimeError
        If the GumTree process fails.

    Notes
    -----
    - This function uses `docker run --rm gumtreediff/gumtree ...` which has a non-trivial
      startup overhead. When mapping many mutants across a commit range, you should avoid
      calling this repeatedly for the same file/commit pair. Instead compute the diff once
      and reuse it (see `map_mutant_with_diff()`).
    """
    if commands is None:
        commands = ["textdiff"]  # or whatever GumTree subcommand you use

    with tempfile.TemporaryDirectory() as tmp:
        # Generate random filenames
        a_name = f"{uuid.uuid4().hex}{lang}"
        b_name = f"{uuid.uuid4().hex}{lang}"

        a_path = os.path.join(tmp, a_name)
        b_path = os.path.join(tmp, b_name)

        with open(a_path, "w", encoding="utf-8") as f:
            f.write(src1)
        with open(b_path, "w", encoding="utf-8") as f:
            f.write(src2)

        # Run GumTree Docker image
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{tmp}:/diff",
            "gumtreediff/gumtree"
        ] + commands + [a_name, b_name]  # use the random names

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"GumTree failed: {result.stderr.strip()}")

        return result.stdout


def analyse_node_changes(
    diff_outputs: str,
    content_src: str,
    content_dst: str,
    types: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Union[int, str, None]]]:
    """
    Parse XML diff output to extract node change information and map
    character index ranges to line numbers in both source and destination contents.

    Args:
        diff_outputs (str): XML string output from GumTree `axmldiff`.
        content_src (str): Source file content (newer / "from" side).
        content_dst (str): Destination file content (older / "to" side).
        types (Optional[Iterable[str]]): Optional AST node-type filter. If provided,
            only nodes whose `type` attribute is in this set are recorded.

    Returns:
        Dict[str, Dict[str, Union[int, str, None]]]:
            Keyed by *source* character range string:
                "src_start_pos-src_end_pos"
            where src_end_pos is INCLUSIVE.

            Each value contains (among other fields):
                - dst_start_pos, dst_end_pos (inclusive), or None if node deleted

    Notes
    -----
    - This mapping is intended to be re-used: for a fixed file and fixed commit pair,
      the AST correspondence does not depend on which mutant we are mapping.
    - End positions are stored as INCLUSIVE indices to match the original logic in
      our `map_mutant()` / `mutant_pos_key` computation:
          mutant_pos_key = f"{start}-{end-1}"
    """
    root = ET.fromstring(diff_outputs)

    chr_to_line_src = char_index_to_line(content_src)
    chr_to_line_dst = char_index_to_line(content_dst)

    type_filter = set(types) if types is not None else None

    out: Dict[str, Dict[str, Union[int, str, None]]] = {}
    for element in root.iter():
        node_type = element.get('type')
        if (node_type is None) or (type_filter is not None and node_type not in type_filter):
            continue

        try:
            # must have
            src_pos = element.get("pos")
            src_length = element.get("length")
            if src_pos is None or src_length is None:
                raise InvalidOutputError("[Invalid Output] pos and length must exist")

            src_pos = int(src_pos)
            src_length = int(src_length)

            # inclusive end
            src_end_pos = src_pos + src_length - 1
            src_start_lno = chr_to_line_src[src_pos]
            src_end_lno = chr_to_line_src[src_end_pos]

            # optional destination range (deleted nodes may lack other_pos/other_length)
            dst_pos = element.get('other_pos')
            dst_length = element.get('other_length')

            if dst_pos is None:
                # Deleted node: represent by None destination coords
                dst_pos, dst_end_pos = None, None
                dst_start_lno, dst_end_lno = None, None
            else:
                if dst_length is None:
                    raise InvalidOutputError(
                        f"[Invalid Output] 'other_length' missing for node-{src_pos}-{src_end_pos}"
                    )

                dst_pos = int(dst_pos)
                dst_length = int(dst_length)

                # inclusive end
                dst_end_pos = dst_pos + dst_length - 1
                dst_start_lno = chr_to_line_dst[dst_pos]
                dst_end_lno = chr_to_line_dst[dst_end_pos]

        except InvalidOutputError as e:
            raise e
        except KeyError as e:
            raise type(e)("[PosNotExist] position does not exist in chr_to_line_src/dst") from e
        except ValueError as e:
            raise type(e)("[Invalid value] malformed attributes") from e

        label = element.get('label')
        key = f"{src_pos}-{src_end_pos}"
        out[key] = {
            "type": node_type,
            "label": label,
            "src_start_pos": src_pos,
            "src_end_pos": src_end_pos,
            "dst_start_pos": dst_pos,
            "dst_end_pos": dst_end_pos,
            "src_start_lno": src_start_lno,
            "src_end_lno": src_end_lno,
            "dst_start_lno": dst_start_lno,
            "dst_end_lno": dst_end_lno,
        }
    return out


def diff(a: str, b: str, lang: str = ".py", target_only_mths: bool = True):
    """
    Compute an AST-based diff between two source code versions using GumTree.

    This function runs GumTree (inside Docker) to compare two code strings and
    returns a structured mapping of changed AST nodes (positions, line numbers,
    and metadata). It’s typically used as a building block for tracking mutants
    or other fine-grained code changes between file revisions.

    Parameters
    ----------
    a : str
        The first version of the source code (typically the "from"/newer version).
    b : str
        The second version of the source code (typically the "to"/older version).
    lang : str, optional
        Programming language extension to inform GumTree’s parser (default ".py").
        Examples: ".py", ".java", ".c".
    target_only_mths : bool, optional
        If True (default), the function is intended to focus only on method-level
        changes but currently raises `NotImplementedError` since that analysis
        isn’t implemented yet.
        If False, the function analyzes and returns all changed AST nodes.

    Returns
    -------
    dict
        A mapping of changed nodes when `target_only_mths` is False.
        Keys are `"start-end"` character position ranges from `a` (inclusive end).
        Values are dictionaries with details such as destination ranges.

    Raises
    ------
    NotImplementedError
        If `target_only_mths` is True (method-level diffing not implemented).
    RuntimeError
        If the underlying GumTree process fails.

    Notes
    -----
    - Internally calls `gumdiff()` to run GumTree’s `axmldiff` command.
    - IMPORTANT for performance: `diff(a, b)` is expensive. If we need to map
      multiple mutants for the same (a, b, file) pair, compute this once and reuse it
      via `map_mutant_with_diff()`.
    """
    diff_out = gumdiff(a, b, lang, commands=["axmldiff"])
    if target_only_mths:
        raise NotImplementedError("analyse_mth_changes method has not been implemented!")
    return analyse_node_changes(diff_out, content_src=a, content_dst=b)


# endregion


def map_mutant(
    src_from: str,
    src_to: str,
    mu_info_from: Dict,
    lang: str = ".py",
    verbose: bool = True,
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Convenience wrapper that maps exactly one mutant by computing GumTree diff internally.

    Returns
    -------
    (mapped_mutant, new_start_pos, new_end_pos)

    - mapped_mutant: mutated version of `src_to` with the mutant applied, or None if
      the node was deleted / cannot be tracked.
    - new_start_pos: inclusive start index of the mutant in mapped_mutant
    - new_end_pos:  end index of the mutant in mapped_mutant (code_mutate convention)

    Notes
    -----
    - This function is SIMPLE but EXPENSIVE: it runs GumTree diff every call.
      When mapping many mutants across the same file and commit pair, it is usually better to do the following instead:
        1) compute `updated_node_range = diff(src_from, src_to, ...)` once
        2) call `map_mutant_with_diff(..., updated_node_range=updated_node_range)` for each mutant
    """
    updated_node_range = diff(src_from, src_to, lang, target_only_mths=False)
    return map_mutant_with_diff(
        src_to=src_to,
        mu_info_from=mu_info_from,
        updated_node_range=updated_node_range,
        verbose=verbose,
    )


def map_mutant_with_diff(
    src_to: str,
    mu_info_from: Dict[str, Any],
    updated_node_range: Dict[str, Dict[str, Any]],
    verbose: bool = True,
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Fast-path mutant mapping using a precomputed GumTree AST diff.

    This function assumes that the caller has already computed the AST mapping for the
    corresponding (src_from, src_to) file pair.

    Unlike `map_mutant()`, this function:
      - does NOT invoke GumTree
      - performs only dictionary lookups + string slicing

    Intended use case:
      - multiple mutants in the same file
      - mapping across the same commit transition
      - compute diff once, map many mutants cheaply
    """
    mutant_pos_key = f"{mu_info_from['mutation_start_pos']}-{mu_info_from['mutation_end_pos']-1}"

    tnode = updated_node_range.get(mutant_pos_key)
    if tnode is None:
        if verbose:
            print(f"[Mutant-Track] failed to find the mutant {mu_info_from['mutant_id']} at {mutant_pos_key}")
        return None, None, None

    dst_start_pos, dst_end_pos = tnode["dst_start_pos"], tnode["dst_end_pos"]

    if dst_start_pos is None:  # deleted
        if verbose:
            print(
                f"[Log] mutant {mu_info_from['mutant_id']} can't be tracked anymore "
                f"as the target node is deleted"
            )
        return None, None, None

    mutant_value = mu_info_from["mutant_value"]

    # Build mutated target source
    mapped_mutant = src_to[:dst_start_pos] + mutant_value + src_to[dst_end_pos + 1:]

    # positions in the *mutated* code, using code_mutate convention
    new_start_pos = dst_start_pos
    new_end_pos = dst_end_pos + 1

    return mapped_mutant, new_start_pos, new_end_pos


def test_case1():
    src_from = """
    import os
    import numpy

    def f():
     x=2
     if x>2:
       print(x*x)
     else:
       x+=3
       print(x+x)
    """

    # converting if x>2 to x==2
    mu_info_from = {"mutant_id": "1", "mutation_start_pos": 64, "mutation_end_pos": 65, "mutant_value": "=="}

    src_to = """
    import os
    import numpy

    def func():
      y=2
      if y>2:
        print(y*y)
      else:
        x=2
        y+=x
        print(y+y-1)
    """

    mapped_mutants, new_start_pos, new_end_pos = map_mutant(src_from, src_to, mu_info_from, lang=".py", verbose=False)

    expected_mutant = """
    import os
    import numpy

    def func():
      y=2
      if y==2:
        print(y*y)
      else:
        x=2
        y+=x
        print(y+y-1)
    """

    assert mapped_mutants == expected_mutant


def test_case2():
    src_from = """
    import os
    import numpy

    def f():
     x=2
     if x>2:
       print(x*x)
     else:
       x+=3
       print(x+x)
    """

    # converting if x>2 to x==2
    mu_info_from = {"mutant_id": "1", "mutation_start_pos": 64, "mutation_end_pos": 65, "mutant_value": "=="}

    src_to = """
    import os
    import numpy

    def func():
      y=2
      if y>2 and y<y+3:
        print(y*y)
      else:
        x=2
        y+=x
        print(y+y-1)
    """

    mapped_mutants, new_start_pos, new_end_pos = map_mutant(src_from, src_to, mu_info_from, lang=".py", verbose=False)

    expected_mutant = """
    import os
    import numpy

    def func():
      y=2
      if y==2 and y<y+3:
        print(y*y)
      else:
        x=2
        y+=x
        print(y+y-1)
    """

    assert mapped_mutants == expected_mutant
