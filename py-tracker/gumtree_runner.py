import subprocess
import tempfile
import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Union, Iterable, Tuple
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
            if idx + 1 < len(content) and content[idx + 1] == '\n': # '\r\n'
                chr_to_line[idx + 1] = line_num   # map '\n' to same line
                idx += 1 # skip the '\n'
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
    """
    if commands is None:
        commands = ["textdiff"]  # or whatever GumTree subcommand you use

    with tempfile.TemporaryDirectory() as tmp:
        # Generate random filenames
        a_name = f"{uuid.uuid4().hex}{lang}"
        b_name = f"{uuid.uuid4().hex}{lang}"

        a_path = os.path.join(tmp, a_name)
        b_path = os.path.join(tmp, b_name)

        with open(a_path, "w") as f:
            f.write(src1)
        with open(b_path, "w") as f:
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
    Parse XML diff output to extract method declaration change information and map
    character index ranges to line numbers in both source and destination contents.

    Args:
        diff_outputs (str): XML string output from a tree differencing tool (e.g., GumTree).
        content_src (str): Source file content (before the change).
        content_dst (str): Destination file content (after the change).

    Returns:
        Dict[str, Dict[str, Union[int, str, None]]]: A dictionary keyed by source line range
            (e.g., "12-20"), each containing:
                - start_pos, end_pos (in destination content)
                - start_lno, end_lno (in destination)
                - label (method name or ID [optional (None)])
    """
    root = ET.fromstring(diff_outputs)

    chr_to_line_src = char_index_to_line(content_src)
    chr_to_line_dst = char_index_to_line(content_dst)

    type_filter = set(types) if types is not None else None

    out: Dict[str, Dict[str, Union[int, str, None]]] = {}
    for element in root.iter():
        src_pos, src_end_pos = None, None  # for default key
        node_type = element.get('type')
        if (node_type is None) or (
                type_filter is not None and node_type not in type_filter
        ):
            continue

        try:
            # must have
            src_pos = element.get("pos")
            src_length = element.get("length")
            if src_pos is None or src_length is None:
                src_pos, src_end_pos = None, None
                raise InvalidOutputError("[Invalid Output] pos and length must exist")
            # ---- ORIGINAL ---------------------------------------
            # src_pos = int(src_pos)
            # src_length = int(src_length)
            # src_end_pos = src_pos + src_length
            # src_start_lno = chr_to_line_src[src_pos]
            # src_end_lno = chr_to_line_src[src_pos + src_length]
            # ---- UPDATED ---------------------------------------
            src_pos = int(src_pos)
            src_length = int(src_length)
            src_end_pos = src_pos + src_length - 1
            src_start_lno = chr_to_line_src[src_pos]
            src_end_lno = chr_to_line_src[src_end_pos]

            # optional destination range - e.g., deletion
            dst_pos = element.get('other_pos')
            # ---- ORIGINAL ---------------------------------------
            # dst_length = element.get('other_pos')
            # ---- UPDATED ---------------------------------------
            dst_length = element.get('other_length')
            if dst_pos is None:
                if dst_length is None:
                    raise TypeError(f"Node {src_pos}-{src_end_pos} has been deleted")
                else:
                    dst_length = None
                    raise InvalidOutputError(
                        f"[Invalid Output] 'other_length' must be also None: {dst_length}for the node-{src_pos}-{src_end_pos}")

            # ---- ORIGINAL ---------------------------------------
            # dst_pos = int(dst_pos)
            # dst_length = int(dst_length)
            # dst_end_pos = dst_pos + dst_length
            # dst_start_lno = chr_to_line_dst[dst_pos]
            # dst_end_lno = chr_to_line_dst[dst_pos + dst_length]
            # ---- UPDATED ---------------------------------------
            dst_pos = int(dst_pos)
            dst_length = int(dst_length)
            dst_end_pos = dst_pos + dst_length - 1
            dst_start_lno = chr_to_line_dst[dst_pos]
            dst_end_lno = chr_to_line_dst[dst_end_pos]

        except InvalidOutputError as e:
            raise e
        except KeyError as e:
            raise type(e)("[PosNotExist] position does not exist in chr_to_line_src/dst") from e
        except TypeError as e:  # None
            #print(e)
            dst_pos, dst_end_pos = None, None
            dst_start_lno, dst_end_lno = None, None
        except ValueError as e:  # malformed attributes
            raise type(e)("[Invalid value] malformed attributes") from e

        # get label
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


def diff(a:str,b:str, lang:str=".py",target_only_mths:bool = True):
    """
    Compute an AST-based diff between two source code versions using GumTree.

    This function runs GumTree (inside Docker) to compare two code strings and
    returns a structured mapping of changed AST nodes (positions, line numbers,
    and metadata). It’s typically used as a building block for tracking mutants
    or other fine-grained code changes between file revisions.

    Parameters
    ----------
    a : str
        The first version of the source code (e.g., "old" version).
    b : str
        The second version of the source code (e.g., "new" version).
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
        Keys are `"start-end"` character position ranges from the source version `a`.
        Values are dictionaries with details such as:
            - type (str): AST node type.
            - label (str or None): Identifier or label of the node.
            - src_start_pos, src_end_pos (int): Character positions in `a`.
            - dst_start_pos, dst_end_pos (int or None): Character positions in `b`.
            - src_start_lno, src_end_lno (int): Line numbers in `a`.
            - dst_start_lno, dst_end_lno (int or None): Line numbers in `b`.

    Raises
    ------
    NotImplementedError
        If `target_only_mths` is True (method-level diffing not implemented).
    RuntimeError
        If the underlying GumTree process fails.

    Notes
    -----
    - Internally calls `gumdiff()` to run GumTree’s `axmldiff` command.
    - Designed to be a helper for higher-level mutant tracking (e.g., `mu_map()`).
    """
    diff_out=gumdiff(a,b,lang,commands=["axmldiff"])
    if target_only_mths:
        raise NotImplementedError("analyse_mth_changes method has not been implemented!")
    else:
        ret = analyse_node_changes(diff_out, content_src=a, content_dst=b)
    return ret

# endregion


def map_mutant(
    src_from: str,
    src_to: str,
    mu_info_from: Dict,
    lang: str = ".py",
    verbose: bool = True,
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Returns
    -------
    (mapped_mutant, new_start_pos, new_end_pos)

    - mapped_mutant: mutated version of `src_to` with the mutant applied, or None if
      the node was deleted / cannot be tracked.
    - new_start_pos: inclusive start index of the mutant in mapped_mutant (using code_mutate convention), or None.
    - new_end_pos:  end index of the mutant in mapped_mutant (using code_mutate convention), or None.
    """
    updated_node_range = diff(src_from, src_to, lang, target_only_mths=False)
    mutant_pos_key = f"{mu_info_from['mutation_start_pos']}-{mu_info_from['mutation_end_pos']-1}"

    tnode = updated_node_range.get(mutant_pos_key)
    if tnode is None:  # does not exist
        # raise MutTrackFailureError(
        #     f"[Mutant-Track] failed to find the mutant {mu_info_from['mutant_id']} at {mutant_pos_key}"
        # )
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
        # consistent 3-tuple
        return None, None, None

    mutant_value = mu_info_from["mutant_value"]

    # Build mutated target source
    mapped_mutant = src_to[:dst_start_pos] + mutant_value + src_to[dst_end_pos + 1 :]

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

    mapped_mutants, new_start_pos, new_end_pos=map_mutant(src_from, src_to, mu_info_from, lang=".py", verbose=False)

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

    assert mapped_mutants==expected_mutant

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

    mapped_mutants, new_start_pos, new_end_pos= map_mutant(src_from, src_to, mu_info_from, lang=".py", verbose=False)

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

    assert mapped_mutants==expected_mutant
