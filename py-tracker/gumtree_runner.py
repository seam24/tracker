import subprocess
import tempfile
import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Union, Iterable

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
        commands = ["axmldiff"]

    # Use a temporary directory to write code files for GumTree to compare
    with tempfile.TemporaryDirectory() as tmp:
        a_path = os.path.join(tmp, f"a{lang}")
        b_path = os.path.join(tmp, f"b{lang}")

        with open(a_path, "w") as f:
            f.write(src1)
        with open(b_path, "w") as f:
            f.write(src2)

        # Run GumTree Docker image
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{tmp}:/diff",
            "gumtreediff/gumtree"
        ] + commands + [f"a{lang}", f"b{lang}"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"GumTree failed: {result.stderr.strip()}")

        return result.stdout


def analyse_mth_changes(
        # target_mthinfos:Dict[str, Dict[str,Union[List[int], str]]],
        diff_outputs: str,
        content_src: str,
        content_dst: str
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
                - label (method name or ID)
    """

    root = ET.fromstring(diff_outputs)
    # from tree, extract change information
    # consider that pydriller can't match the mdofiied line to a new line -> meaning we need to extract such informaton here
    # as we need to update the method's range, which will be keep chaning.
    ## later for the update, the key would be .. if some methods have been changed,
    chr_to_line_src = char_index_to_line(content_src)
    chr_to_line_dst = char_index_to_line(content_dst)

    updated_mth_range = {}
    mthod_declars =[]
    for element in root.iter():
        node_type = element.get('type')
        if node_type =="funcdef":
            mthod_declars.append(element)

    for mth_declar in mthod_declars:
        # --- source node ---
        pos = int(mth_declar.get('pos'))
        length = int(mth_declar.get('length'))
        end_pos = pos + length - 1  # fix off-by-one
        start_lno = chr_to_line_src[pos]
        end_lno = chr_to_line_src[end_pos]

        # --- destination node (if mapped) ---
        other_pos = mth_declar.get('other_pos')
        if other_pos is not None:
            other_pos = int(other_pos)
            other_length = int(mth_declar.get('other_length'))
            other_end_pos = other_pos + other_length - 1
            other_start_lno = chr_to_line_dst[other_pos]
            other_end_lno = chr_to_line_dst[other_end_pos]
        else:
            other_pos = None
            other_end_pos = None
            other_start_lno = None
            other_end_lno = None

        updated_mth_range[f"{start_lno}-{end_lno}"] = {
            "start_pos": other_pos,
            "end_pos": other_end_pos,
            "start_lno": other_start_lno,
            "end_lno": other_end_lno
        }

    return updated_mth_range

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
            print(e)
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
    diff_out=gumdiff(a,b,lang,commands=["axmldiff"])
    #print(diff_out)
    if target_only_mths:
        ret = analyse_mth_changes(diff_out, content_src=a, content_dst=b)
    else:
        ret = analyse_node_changes(diff_out, content_src=a, content_dst=b)
    return ret

if __name__ == "__main__":
    with open("tests/a.py",'r') as f:
        a = f.read()
    with open("tests/b.py",'r') as f:
        b = f.read()
    updated_node_range=diff(a, b,".py",target_only_mths=False)
    with open("tests/mu_a.py",'r') as f:mu_a = f.read()
    mid=1
    m_start_pos= 43
    m_end_pos = 43
    mutant_value="=="
    m_pos_key = f"{m_start_pos}-{m_end_pos}"
    tnode = updated_node_range.get(m_pos_key)
    print(tnode)
    if tnode is None:  # does not exist
        raise MutTrackFailureError(f"[Mutant-Track] failed to find the mutant {mid} at {m_pos_key}")

    # replace
    dst_start_pos, dst_end_pos = tnode['dst_start_pos'], tnode['dst_end_pos']

    if dst_start_pos is None:  # deleted
        #TODO: UPDATE KEEP TRACK
        print(f"[Log] mutant {mid} can't be tracked anymore as the target node is deleted")
        #continue

    print(b)
    print(mutant_value)
    b = b[:dst_start_pos] + mutant_value + b[dst_end_pos + 1:]

    print(b)