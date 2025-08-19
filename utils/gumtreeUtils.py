""" 
Gumtree operations 
"""
import os, sys
import xml.etree.ElementTree as ET
import utils.dockerUtils as dockerUtils
from typing import List, Dict, Tuple, Optional, Union, Iterable, Any
import json 

VALID_CMDS = ['webdiff', 'cluster', 'dotdiff', 'htmldiff', 'list', 'parse', 'swingdiff', 'textdiff', 'axmldiff'] # likely for our case, mainly focus on axmldiff and parse

class InvalidOutputError(Exception):
    """Raise invalid output error"""
    pass
    
def run_docker(
    image:str, 
    cmd:str,
    platform:Optional[str] = "linux/amd64", 
    host_path:Optional[str] = None, container_path:Optional[str] = None, 
    return_stdout:bool = False
) -> Union[bool, Tuple[str, bool]]:
    import shlex
    # docker run -v $PWD/NumberUtilsA.java:/diff/NumberUtilsA.java -v $PWD/NumberUtils.java:/diff/NumberUtils.java -p 4567:4567 gumtreediff/gumtree webdiff /diff/NumberUtilsA.java /diff/NumberUtils.java
    cmd = shlex.split(cmd)
    if cmd[0] not in VALID_CMDS:
        print (f"Invalid Gumtree command: {cmd[0]}\n\tVALID CMDS: {",".join(VALID_CMDS)}") 
        return False
    
    volumes = {host_path: container_path} if host_path and container_path else None 
        
    (exit_code, stdout, stderr) = dockerUtils.run_docker_image(
        image=image, 
        command=cmd, 
        platform=platform, 
        volumes=volumes, 
        detach=False
    )
    #print (f"Exit code: {exit_code}", f"\nstdout:{stdout}", f"\nstderr: {stderr}")
    #if exit_code == 0:
    if stderr:
        print (f"[Docker Error] Exit code {exit_code}: {stderr or 'Unknown error'}") 
        return False
    else: 
        if return_stdout:
            return (stdout, True)
        else:
            return True

def convert_hostfpath_to_container_fpath(fpath:str, host_path:str, container_path:str) -> str:
    """_summary_

    Args:
        fpath (str): _description_
        host_path (str): _description_
        container_path (str): _description_

    Raises:
        Exception: _description_

    Returns:
        str: _description_
    """
    # expect all to be absolute path 
    fpath = os.path.abspath(fpath) # to further ensure 
    fpath_dirpath = os.path.dirname(fpath)
    # fpath_dirpath should be udner host_path
    _common_path = os.path.commonpath([fpath_dirpath, host_path])
    #print (_common_path, type(_common_path))
    #print (host_path, type(host_path))
    #print (_common_path == host_path)
    if _common_path != host_path: 
        raise Exception(f"[ERROR] {fpath} should be under {host_path}")
    sub_fpath = os.path.relpath(fpath, host_path)
    fpath_container = os.path.join(container_path, sub_fpath)
    return fpath_container

def analyse_mth_changes(
    #target_mthinfos:Dict[str, Dict[str,Union[List[int], str]]], 
    diff_outputs:str, 
    content_src:str, 
    content_dst:str
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
    # will parse given raw changes, and return the mapped information   
    from utils.parseUtils import char_index_to_line
     
    root = ET.fromstring(diff_outputs)
    # from tree, extract change information 
    # consider that pydriller can't match the mdofiied line to a new line -> meaning we need to extract such informaton here
    # as we need to update the method's range, which will be keep chaning. 
    ## later for the update, the key would be .. if some methods have been changed,     
    chr_to_line_src = char_index_to_line(content_src)
    chr_to_line_dst = char_index_to_line(content_dst)
    
    updated_mth_range = {}
    mthod_declars = root.findall(".//*[@type='MethodDeclaration']")
    for mth_declar in mthod_declars:
        pos = int(mth_declar.get('pos'))
        length = int(mth_declar.get('length')) 
        
        # optional destination range - e.g., deletion
        other_pos = mth_declar.get('other_pos') 
        if other_pos is not None: 
            other_pos = int(other_pos)
            other_length = int(mth_declar.get('other_length'))
            other_start_lno = chr_to_line_dst[other_pos]
            other_end_lno = chr_to_line_dst[other_pos + other_length]
        else:
            other_length = None
            other_start_lno, other_end_lno = None, None
        
        # map chr to line 
        start_lno = chr_to_line_src[pos]
        end_lno = chr_to_line_src[pos + length]

        # get label 
        #label = mth_declar.get('label')
        
        updated_mth_range[f"{start_lno}-{end_lno}"] = {
            'start_pos':other_pos, 
            'end_pos':other_pos + other_length if other_pos is not None else None, 
            'start_lno':other_start_lno, 
            'end_lno':other_end_lno}
            #'label':label}
    return updated_mth_range


def analyse_node_changes(
    diff_outputs:str, 
    content_src:str, 
    content_dst:str, 
    types:Optional[Iterable[str]] = None, 
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
    # will parse given raw changes, and return the mapped information   
    from utils.parseUtils import char_index_to_line
     
    root = ET.fromstring(diff_outputs)

    chr_to_line_src = char_index_to_line(content_src)
    chr_to_line_dst = char_index_to_line(content_dst)
    
    type_filter = set(types) if types is not None else None
    
    out:Dict[str, Dict[str, Union[int, str, None]]] = {}
    for element in root.iter():
        src_pos, src_end_pos = None, None # for default key
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
        
            src_pos = int(src_pos)
            src_length = int(src_length)
            src_end_pos = src_pos + src_length
            src_start_lno = chr_to_line_src[src_pos]
            src_end_lno = chr_to_line_src[src_pos + src_length]
            
            # optional destination range - e.g., deletion
            dst_pos = element.get('other_pos')
            dst_length = element.get('other_pos')
            if dst_pos is None:
                if dst_length is None:
                    raise TypeError(f"Node {src_pos}-{src_end_pos} has been deleted")
                else:
                    dst_length = None
                    raise InvalidOutputError(f"[Invalid Output] 'other_length' must be also None: {dst_length}for the node-{src_pos}-{src_end_pos}") 
            
            dst_pos = int(dst_pos)
            dst_length = int(dst_length)
            dst_end_pos = dst_pos + dst_length
            dst_start_lno = chr_to_line_dst[dst_pos]
            dst_end_lno = chr_to_line_dst[dst_pos + dst_length]
           
        except InvalidOutputError as e:
            raise e
        except KeyError as e:
            raise type(e)("[PosNotExist] position does not exist in chr_to_line_src/dst") from e
        except TypeError as e: # None
            print (e)
            dst_pos, dst_end_pos = None, None
            dst_start_lno, dst_end_lno = None, None
        except ValueError as e: # malformed attributes
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

def diff(
    image:str, 
    srcfpath:str, 
    dstfpath:str, 
    platform:Optional[str] = "linux/amd64", 
    host_path:Optional[str] = None, 
    container_path:Optional[str] = "/data", 
    target_only_mths:bool = True
) -> Dict[str, Dict[str,int]]:
    """_summary_

    Args:
        image (str): _description_
        srcfpath (str): _description_ 
        dstfpath (str): _description_
        platform (Optional[str], optional): _description_. Defaults to "linux/amd64".
        host_path (Optional[str], optional): _description_. Defaults to None.
        container_path (Optional[str], optional): _description_. Defaults to "/data".

    Raises:
        Exception: _description_

    Returns:
        Dict[str, Tuple[int,int]]: str (key) is a string of {start_lno}-{end_lno}
    """
    # perform the diff between two files -- prev and dst version of the modified file
    # the goal is to keep track of evolving methods, along with how they have been changed (diff -> this part (the semantic) part can be obtained by pydriller, so not the scope of gumtree-operation. The goal here is to keepinig "track" and thereby matching what has been changed to what
    # so far, we only target methods: 
    # get content_src and content_afr 
    
    from utils.fileUtils import get_content
    src_fcontent = get_content(srcfpath, as_is = True)
    dst_fcontent = get_content(dstfpath, as_is = True)
    
    # srcfpath and dstfpath are file paths in the host-machine
    cont_srcfpath = convert_hostfpath_to_container_fpath(srcfpath, host_path, container_path)
    cont_dstfpath = convert_hostfpath_to_container_fpath(dstfpath, host_path, container_path)
    
    # run diff 
    cmd = f"axmldiff {cont_srcfpath} {cont_dstfpath}"
    # for successu
    out = run_docker(
        image, 
        cmd, 
        platform = platform, 
        host_path = host_path, container_path = container_path, 
        return_stdout = True
    )
    if not out: 
        raise Exception(f"[Error] Failed while executing: \n\t{cmd}") # shouldn't come into here
    else: 
        diff_outputs, _ = out
        # parse file
        # currenlty target only the me
        if target_only_mths:
            ret = analyse_mth_changes(diff_outputs, content_src=src_fcontent, content_dst=dst_fcontent)
        else:
            ret = analyse_node_changes(diff_outputs, content_src=src_fcontent, content_dst=dst_fcontent)
        return ret  


def get_target_nodes(
    node:Union[Dict, List], 
    target_types:Optional[List[str]] = None, #['MethodDeclaration']
) -> List[Dict]:
    """_summary_

    Args:
        node (Union[Dict, List]): _description_
        target_types (List[str], optional): _description_. Defaults to ['MethodDeclaration'].

    Returns:
        List[Dict]: a list of method declaration nodes in a dictionary form
    """
    if target_types is None: 
        target_types = ['MethodDeclaration']
    # By default only return methods 
    # the fucntion willbe called in "def parse"
    # parse the result of gumtree parse 
    collected = []
    type_to_exclude = "CompilationUnit"
    if isinstance(node, dict):
        node_type = node.get("type")
        if node_type != type_to_exclude:
            if node_type in target_types or len(target_types) == 0:
                collected.append(node)
        for child in node.get("children", []):
            collected.extend(get_target_nodes(child, target_types))
    elif isinstance(node, list):
        for item in node:
            collected.extend(get_target_nodes(item, target_types))
    return collected 


def parse(
    image:str, 
    targetfpath:str, 
    platform:Optional[str] = "linux/amd64", 
    host_path:Optional[str] = None, 
    container_path:Optional[str] = "/data", 
    target_types:Optional[List[str]] = None, 
) -> Optional[List[Dict]]:
    # By default, we assume json output format
    # run parsing and return all methods 
    output_basename = "parse_output.json"
    cont_targetfpath = convert_hostfpath_to_container_fpath(targetfpath, host_path, container_path) 
    cmd = f"parse -f JSON -o {os.path.join(container_path, output_basename)} {cont_targetfpath}"
    status = run_docker(
        image, 
        cmd, 
        platform = platform, 
        host_path = host_path, container_path = container_path
    )
    
    if status: # succeed
        # parse file 
        output_file = os.path.join(host_path, output_basename)
        with open(output_file) as f:
            import json 
            raw_parsed = json.load(f)
        root_node = raw_parsed.get('root')
        if root_node: 
            mth_declrs = get_target_nodes(root_node, target_types)
            return mth_declrs
        else: # None 
            print (f"[Error] Invalid format. Please check {output_file}") 
            return None 
    else:
        return None

#################################################################################################
#                                   Method Labeling Parsing                                     #
#################################################################################################
CLASS_LIKE = {"TypeDeclaration", "EnumDeclaration", "AnnotationTypeDeclaration"}

def _child(n: Dict[str, Any], t: str) -> Optional[Dict[str, Any]]:
    return next((c for c in n.get("children", []) if c.get("type") == t), None)

def _children(n: Dict[str, Any], t: str) -> List[Dict[str, Any]]:
    return [c for c in n.get("children", []) if c.get("type") == t]

def _type_to_str_minimal(type_node: Optional[Dict[str, Any]]) -> str: # deprecated
    if not type_node:
        return "?"
    # Prefer explicit text if present
    lbl = type_node.get("label")
    if lbl:
        return lbl
    # Otherwise try nested SimpleName.label
    sn = _child(type_node, "SimpleName")
    if sn and sn.get("label"):
        return sn["label"]
    # Fallback: collect any labels in subtree
    parts: List[str] = []
    stack = [type_node]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            if n.get("label"):
                parts.append(n["label"])
            stack.extend(n.get("children", []))
    return "".join(parts) or "?"

def _base_name_from_simple(n: Dict[str, Any]) -> str:
    # Try label first, then nested SimpleName.label
    if n.get("label"):
        return n["label"]
    sn = _child(n, "SimpleName")
    return sn["label"] if sn and sn.get("label") else "?"

def _type_to_str(tn: Optional[Dict[str, Any]]) -> str:
    """Reconstruct a Java type string from a GumTree/JavaParser type subtree."""
    if not tn:
        return "?"
    t = tn.get("type")

    # Simple/primitive/void types
    if t in ("SimpleType", "PrimitiveType", "VoidType"):
        return _base_name_from_simple(tn)

    # QualifiedType: prefix.SimpleName
    if t == "QualifiedType":
        prefix = _child(tn, "SimpleType") or _child(tn, "QualifiedType")
        name = _child(tn, "SimpleName")
        left = _type_to_str(prefix) if prefix else "?"
        right = name["label"] if name and name.get("label") else "?"
        return f"{left}.{right}"

    # ArrayType: elementType[]
    if t == "ArrayType":
        # Element can itself be ArrayType / ParameterizedType / SimpleType / PrimitiveType
        elem = (
            _child(tn, "ArrayType")
            or _child(tn, "ParameterizedType")
            or _child(tn, "SimpleType")
            or _child(tn, "PrimitiveType")
            or _child(tn, "QualifiedType")
        )
        # Count nested ArrayType levels
        levels = 0
        cur = tn
        while cur and cur.get("type") == "ArrayType":
            levels += 1
            cur = _child(cur, "ArrayType")  # dive if nested
        return _type_to_str(elem) + "[]" * levels

    # ParameterizedType: MainType<Arg1,Arg2,...>
    if t == "ParameterizedType":
        main = (
            _child(tn, "SimpleType")
            or _child(tn, "QualifiedType")
            or _child(tn, "PrimitiveType")
        )
        # Type arguments may be SimpleType / ParameterizedType / ArrayType / QualifiedType / WildcardType
        args_nodes: List[Dict[str, Any]] = []
        for k in ("SimpleType", "ParameterizedType", "ArrayType", "QualifiedType", "WildcardType"):
            args_nodes.extend(_children(tn, k))
        args = ",".join(_type_to_str(a) for a in args_nodes)
        return f"{_type_to_str(main)}<{args}>"

    # WildcardType: ?, ? extends T, ? super T (best effort)
    if t == "WildcardType":
        bound = (
            _child(tn, "SimpleType")
            or _child(tn, "QualifiedType")
            or _child(tn, "ParameterizedType")
            or _child(tn, "ArrayType")
        )
        # Bound keywords aren’t always explicit in the JSON; show best-effort
        return f"? { _type_to_str(bound) }" if bound else "?"

    # Fallback: collect labels in subtree
    lbl = tn.get("label")
    if lbl:
        return lbl
    parts: List[str] = []
    stack = [tn]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            if n.get("label"):
                parts.append(n["label"])
            stack.extend(n.get("children", []))
    return "".join(parts) or "?"

def _param_sig(method_node: Dict[str, Any]) -> str:
    # Params are SingleVariableDeclaration with a *Type child
    params = _children(method_node, "SingleVariableDeclaration")
    types: List[str] = []
    for p in params:
        t = (
            _child(p, "ArrayType")
            or _child(p, "ParameterizedType")
            or _child(p, "QualifiedType")
            or _child(p, "SimpleType")
            or _child(p, "PrimitiveType")
        )
        types.append(_type_to_str(t))
    return ",".join(types)

    
def method_local_signature(
    method_node: Dict[str, Any],
    enclosing_classes: Optional[List[str]] = None,
) -> str:
    """
    Build a local method signature for a single MethodDeclaration node:
      Outer.Inner#method(Type1,Type2[])
    - Constructors are already represented as MethodDeclaration with class name as SimpleName (per your data).
    - If enclosing_classes is omitted, the class qualifier is omitted.
    """
    name_node = _child(method_node, "SimpleName") # method name lives in child SimpleName
    method_name = (name_node or method_node).get("label") or "unnamed"
    params = _param_sig(method_node)

    if enclosing_classes:
        cls = ".".join(enclosing_classes)
        return f"{cls}#{method_name}({params})"
    else:
        return f"{method_name}({params})"

