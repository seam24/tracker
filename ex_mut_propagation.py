import os, sys
from typing import List, Dict, Union
from tqdm import tqdm
from pathlib import Path
from track.gitHelper import GitRepoHelper
from utils import gumtreeUtils 
from utils.fileUtils import save_content, get_content
import traceback 
import pickle 
from run import update_files_to_track

GUMTREE_IMAGE = None
GUMTREE_PLATFORM = None
OUTPUT_DIR = None

class MutTrackFailureError(Exception):
    """Raise invalid output error"""
    pass

def compile_and_test():
    print (f"[Log] Compile and Test")
    pass

def run(
    repo_path:str, 
    end_commit_hash:str, 
    mutants_info:Dict[str, Dict[str, List[Dict[str, Union[str, int, None]]]]]
):
    """_summary_

    Args:
        repo_path (str): _description_
        end_commit_hash (str): the hash of the last commit to look at (time-window based)
        mutants_info (Dict[str, Dict[str, List[Dict[str, Union[str, int, None]]]]]): _description_
            fpath (str): initial file path (when injected)
            value: Dict
                mid (str): mutant key (e.g., '1', '2', etc.)
                Value: List (can be just a dictionary instead of a list of dictionaries -- depend on whether we update nodes that compose a mutant individually or just replace the top node entirely)
                    item1: Dict
                        'start_pos': int
                        'end_pos': int (pos + 1)
                        'mutant_value': Optional[str] (mutant informaion. None if it is a deletion)
                            - src_code[start_pos:end_pos] == mutant_value
                        'keep_track': True (target node still exist) and False (target node deleted)
                    item2: Dict (the same)
                    ...
                    itemn: Dict (the same)  
    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    # assume the mutants are inserted at the current commit of the repo_path
    
    host_path = Path(repo_path).parent.absolute()
    container_path = "/data"
    
    rpHelper = GitRepoHelper(repo_path, order="reverse", only_no_merge=True)
    # the intermediate directoy to save temporary source files (saved for gumtree operation)
    intermediate_dir = os.path.join(host_path, "temp_srcdir")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    commits = list(rpHelper.pydriller_repo.traverse_commits())
    print (f"[Start] Traversal starts from {commits[0].hash}")
    
    files_to_track = set(mutants_info) # may change due to some files becoming to be able to influence the mutatns
    initial_to_latest: Dict[str,str] = {fpath:fpath for fpath in files_to_track}
    latest_to_initial: Dict[str,str] = {fpath:fpath for fpath in files_to_track}
    for i, commit in enumerate(tqdm(commits)):
        if commit.hash == end_commit_hash:
            print (f"[End-Condition] reach the end of the search: {end_commit_hash}")
            break
        
        print (f"At commit: {commit.hash}")
        prev_commit = commits[i+1] if i < len(commits) - 1 else None
        fpath_to_update:Dict[str, str] = {} # afpath(src) : bfpath
        for modified_file in commit.modified_files: 
            # old_path: not in the current commit (the previosu commit) 
            # new_path: in the current commit (and thereby, Method.start_lno and Method.end_lno are from the file of the curretn commit,and thereby form "new")
            src_path = modified_file.new_path
            if src_path not in files_to_track:
                continue
            
            dst_path = modified_file.old_path  
            # check whether we need to further process this file's changes
            if dst_path is None: 
                print (f"[Newly Introduced] A {src_path} is introduced at {commit.hash}")
                print (f"                   End the tracking for {src_path}")
                continue
            
            # prepare the update for later 
            fpath_to_update[src_path] = dst_path
                
            # We need to get the file as "as-is" instead of decoded string to match with the outputs provided by gumtree
            out = GitRepoHelper.get_file_content(rpHelper.get_git_commit(commit.hash), src_path)
            if out is None:
                raise ValueError(f"Failed to retrive {src_path}")
            srcode_src, _ = out 
            out = GitRepoHelper.get_file_content(rpHelper.get_git_commit(prev_commit.hash), dst_path)
            if out is None:
                raise ValueError(f"Failed to retrive {dst_path}") 
            srcode_dst, _ = out

            # write
            local_src_path = os.path.join(intermediate_dir, "src_" + os.path.basename(src_path))
            local_dst_path = os.path.join(intermediate_dir, "dst_" + os.path.basename(dst_path))
            save_content(srcode_src, local_src_path, as_is = True, encoding='utf-8')# b/c later need to be given as input to Gumtree
            save_content(srcode_dst, local_dst_path, as_is = True, encoding='utf-8')# b/c later need to be given as input to Gumtree
            # run gumtree and obtain diff information 
            # key: {start_lno}-{end_lno}
            updated_node_range = gumtreeUtils.diff(
                image=GUMTREE_IMAGE, 
                srcfpath=local_src_path, 
                dstfpath=local_dst_path, 
                platform=GUMTREE_PLATFORM, 
                host_path=str(host_path),
                container_path=container_path, 
                target_only_mths=False
            ) 
            fpath_key = latest_to_initial[src_path]
            mutants_to_lookat = mutants_info[fpath_key]
            
            for mid, minfos in mutants_to_lookat.items(): 
                # mid: str 
                # minfo: List[Dict[str, Union[str, int, None]]] -> mutant position information (new and prev -- start and end pos), mutant value: replacement value
                for minfo in minfos: # assume list
                    if not minfo['keep_track']: continue
                    # update
                    m_start_pos = minfo['start_pos']
                    m_end_pos = minfo['end_pos']
                    mutant_value = minfo['mutant_value']
                    
                    # insert mutant value to dst_start_pos:dst_end_pos
                    m_pos_key = f"{m_start_pos}-{m_end_pos}"
                    tnode = updated_node_range.get(m_pos_key)
                    if tnode is None: # does not exist 
                        raise MutTrackFailureError(f"[Mutant-Track] failed to find the mutant {mid} at {m_pos_key}")
                    
                    # replace
                    dst_start_pos, dst_end_pos = tnode['dst_start_pos'], tnode['dst_end_pos']
                    if dst_start_pos is None: # deleted
                        minfo['keep_track'] = False 
                        print (f"[Log] mutant {mid} can't be tracked anymore as the target node is deleted")
                        continue
                    
                    srcode_dst[dst_start_pos:dst_end_pos] = mutant_value
                    ########################################### 
                    #           Compile and Testing           #
                    ########################################### 
                    compile_and_test() # need to implement
                    # ... 
                    
        update_files_to_track(initial_to_latest, latest_to_initial, files_to_track, fpath_to_update)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("-mut", "--mutfile", type = str)
    parser.add_argument("-r", "--repo_path", type = str)
    args = parser.parse_args()
    
    with open(args.mutfile, 'r') as f:
        import json 
        mutants_info = json.load(f)
            
    repo_path = Path(args.repo_path).absolute()
    end_commit_hash = ... 
    run(
        repo_path, 
        end_commit_hash, 
        mutants_info #:Dict[str, Dict[str, List[Dict[str, Union[str, int, None]]]]]
    )