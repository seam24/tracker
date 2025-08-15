""" 
The goal is to check whether the target is sufficiently changed to be used to do the change analysis 
    - Will follow the renaming 
    
    in the end, in additon to just changed together, we can furhter check the semantics of the content, the authors, etc. 
"""
import os, sys 
from typing import List, Dict, Tuple, Set, Optional
from track.gitHelper import GitRepoHelper
from utils import gumtreeUtils
from track.node import TMethod 
from track.changes import Change
from utils.fileUtils import save_content, get_content
import traceback 
import pickle 

GUMTREE_IMAGE = None
GUMTREE_PLATFORM = None 
INTERMEDIATE_RESULTFILE = "{0}/inter_results.pkl"
OUTPUT_DIR = None

def init(
    rpHelper:GitRepoHelper, 
    intermediate_dir:str, 
    target_files:Optional[List[str]]=None, 
    file_types:Optional[List[str]] = None,
    target_types:Optional[List[str]] = None, 
) -> Dict[str, Dict[str, TMethod]]:
    """_summary_

    Args:
        rpHelper (GitRepoHelper): _description_
        intermediate_dir (str): _description_
        target_files (Optional[List[str]], optional): _description_. Defaults to None.
        file_types (Optional[List[str]], optional): _description_. Defaults to None.
        target_types (Optional[List[str]], optional): _description_. Defaults to None.

    Returns:
        Dict[str, Dict[str, TMethod]]: _description_
    """
    def clean(temp_fpath:str):
        if os.path.exists(temp_fpath):
            os.unlink(temp_fpath)     
    # target_files: 
    #   - both source and test files should be tracked
    tracked_files = []
    for fpath, _ in rpHelper.repo.index.entries:
        if (file_types is None or 
            any([fpath.endswith(_ftype) for _ftype in file_types])):
            tracked_files.append(fpath)
        
    # check if all target files have been tracked by Git
    if target_files is not None: 
        for target_file in target_files: 
            if target_file not in tracked_files: 
                raise Exception(f"File {target_file} is not tracked by Git: OUT_OF_SCOPE")
    else: # target all tracked files 
        target_files = tracked_files 
    
    methods_by_file = {} 
    for target_file in target_files: 
        methods_by_file[target_file] = {}
        temp_fpath = os.path.join(intermediate_dir, os.path.basename(target_file))
        write_status = rpHelper.write_file_to_disk(rpHelper.head_commit, target_file, temp_fpath)#, encoding='utf-8')
        if not write_status:
            raise Exception(f"Failed to retrieve and save the content of {target_file} in {temp_fpath}")
    
        mth_declrs = gumtreeUtils.parse(
            GUMTREE_IMAGE, 
            temp_fpath, 
            GUMTREE_PLATFORM, 
            host_path=intermediate_dir, 
            container_path="/data", 
            target_types = target_types,
        )
        if mth_declrs:
            for idx, mth_declr in enumerate(mth_declrs):
                pos = int(mth_declr.get('pos'))
                length = int(mth_declr.get('length'))
                children = mth_declr.get('children')
                label = mth_declr.get('label')
                mth_node = TMethod(
                    file_path = target_file, 
                    pos = pos, 
                    length = length, 
                    children = children, 
                    label = label,
                    local_file_path = temp_fpath 
                )
                methods_by_file[target_file][idx] = mth_node
                # clean up 
            clean(temp_fpath) 
        elif mth_declrs is not None: # empty list 
            print (f"[Log] No methods in {target_file}")
        else:
            raise Exception(f"[Error] Failed to initialise method declarations in {target_file}") 
    return methods_by_file

def save_init_methods(methods_by_file:Dict[str, Dict[str, TMethod]], init_file:str):
    """_summary_

    Args:
        methods_by_file (Dict[str, Dict[str, TMethod]]): _description_
        init_file (str): _description_
    """
    d_methods_by_file = {}
    for fpath in methods_by_file:
        mth_nodes: Dict[str, TMethod] = methods_by_file[fpath]
        d_methods_by_file[fpath] = {}
        for mth_pos_key in mth_nodes:
            mth_node:TMethod = mth_nodes[mth_pos_key]
            d_methods_by_file[fpath][mth_pos_key] = mth_node.to_dict()  
    save_content(d_methods_by_file, init_file, as_is=True, encoding='utf-8') # b/c later need to be given as input to Gumtree

def construct_from_saved_init_methods(init_file:str, target_files:Optional[List[str]] = None) -> Dict[str, Dict[str, TMethod]]:
    """_summary_

    Args:
        init_file (str): _description_

    Returns:
        Dict[str, Dict[str, TMethod]]: _description_
    """
    import json
    
    dmethods_by_file = json.loads(get_content(init_file, as_is = True))
    if target_files is not None:
        dmethods_by_file = {target_file:dmethods_by_file[target_file] for target_file in target_files}
    methods_by_file: Dict[str, Dict[str, TMethod]] = {}
    
    for fpath, d_mths in dmethods_by_file.items():
        methods_by_file[fpath] = {}
        for lno_key, d_mth in d_mths.items():
            methods_by_file[fpath][lno_key] = TMethod.from_dict(d_mth, asTNodes=['javadoc', 'block'])
    return methods_by_file

def update_files_to_track(initial_to_latest:Dict[str, str], latest_to_initial:Dict[str,str], 
                          files_to_track:Set[str], to_update:Dict[str, Optional[str]]): 
    init_n = len(files_to_track)
    updated_fpaths = []
    for afpath, bfpath in to_update.items():
        if afpath != bfpath:
            init_path = latest_to_initial[afpath]
            initial_to_latest[init_path] = bfpath # can contain None
            del latest_to_initial[afpath]
            latest_to_initial[bfpath] = init_path
            files_to_track.remove(afpath)
            if bfpath is not None:
                files_to_track.add(bfpath)
                updated_fpaths.append(bfpath)
                if bfpath in updated_fpaths:
                    print(f"[WARN] Potential merge: {bfpath} already tracked")
            else:
                print (f"[STOP-TRACKING] {afpath} doesn't exist anymore") 
        else:
            updated_fpaths.append(bfpath)
            
    updated_n = len(files_to_track)
    print (f"[# OF FILES-TO-TRACK] From {init_n} to {updated_n}")
    
def update_method_to_track(fpath:str, mth_latest_lnokey_to_uid:Dict[str,Dict[str,str]], 
                           mth_uid_to_latest_lnokey:Dict[str, Dict[str,str]],
                           to_update_mid_new:List[Tuple[str,str]]):
    mth_uid_to_latest_lnokey[fpath] = dict()
    mth_latest_lnokey_to_uid[fpath] = dict()
    for mid, new_v in to_update_mid_new:
        mth_uid_to_latest_lnokey[fpath][mid] = new_v
        mth_latest_lnokey_to_uid[fpath][new_v] = mid 


def get_tracking_keys(methods_by_file: Dict[str, Dict[str, TMethod]]) -> Tuple[Dict, Dict, Dict, Dict]:
    """_summary_

    Args:
        methods_by_file (Dict[str, Dict[str, TMethod]]): _description_

    Returns:
        Tuple[Dict, Dict, Dict, Dict]: _description_
    """
    # Outer keys 
    initial_to_latest: Dict[str,str] = {}
    latest_to_initial: Dict[str,str] = {}
    # Global structure
    mth_uid_to_latest_lnokey: Dict[str, Dict[str, str]] = {}
    # maps file_path -> lno_key -> method_uid
    mth_latest_lnokey_to_uid: Dict[str, Dict[str, str]] = {}
    # maps file_path -> method_uid -> current lno_key
    for fpath, mth_nodes in methods_by_file.items():
        initial_to_latest[fpath] = fpath
        latest_to_initial[fpath] = fpath
        mth_uid_to_latest_lnokey[fpath] = {}
        mth_latest_lnokey_to_uid[fpath] = {}
        for uid, mth_node in mth_nodes.items():
            lno_key = mth_node.lno_key
            mth_uid_to_latest_lnokey[fpath][uid] = lno_key
            mth_latest_lnokey_to_uid[fpath][lno_key] = uid
    return (initial_to_latest, latest_to_initial, mth_uid_to_latest_lnokey, mth_latest_lnokey_to_uid)
    
def save_inter_result(
    methods_by_file:Dict[str, Dict[str, TMethod]], 
    files_to_track:Set[str], 
    initial_to_latest:Dict[str, Optional[str]], 
    latest_to_initial:Dict[str, Optional[str]], 
    mth_uid_to_latest_lnokey:Dict[str, Dict[str, Optional[str]]], 
    mth_latest_lnokey_to_uid:Dict[str, Dict[str, Optional[str]]], 
    curr_processed_modfiles:Set[str], 
    last_commit_idx:int
    ):
    with open(INTERMEDIATE_RESULTFILE, 'wb') as f:
        pickle.dump(
            {'output':methods_by_file, 
                'index':{
                    'initial_to_latest': initial_to_latest, 
                    'latest_to_initial': latest_to_initial,
                    'mth_uid_to_latest_lnokey': mth_uid_to_latest_lnokey, 
                    'mth_latest_lnokey_to_uid': mth_latest_lnokey_to_uid
                    }, 
                'files_to_track':files_to_track, 
                'curr_processed_modfiles': curr_processed_modfiles,
                'last_commit_idx':last_commit_idx
            }, f) 


def run(
    repo_path:str, 
    host_path:str, 
    target_files:Optional[List[str]] = None, 
    file_types:Optional[List[str]] = ['.java'], 
    time_window:Optional[int] = 365, 
    init_file:Optional[str] = None
) -> Dict[str, Dict[str, TMethod]]:
    """_summary_

    Args:
        repo_path (str): _description_
        host_path (str): _description_
        target_files (Optional[List[str]], optional): _description_. Defaults to None.
        file_types (Optional[List[str]], optional): _description_. Defaults to ['.java'].
        time_window (Optional[int], optional): _description_. Defaults to 365. Counts in days

    Returns:
        Dict[str, Dict[str, TMethod]]: _description_
    """
    global GUMTREE_PLATFORM, GUMTREE_IMAGE, INTERMEDIATE_RESULTFILE, OUTPUT_DIR
    from tqdm import tqdm
    from pathlib import Path
    
    host_path = Path(repo_path).parent.absolute()
    container_path = "/data"
    
    rpHelper = GitRepoHelper(repo_path, order="reverse", only_no_merge=True)
    # the intermediate directoy to save temporary source files (saved for gumtree operation)
    intermediate_dir = os.path.join(host_path, "temp_srcdir")
    os.makedirs(intermediate_dir, exist_ok=True)

    try:
        methods_by_file = construct_from_saved_init_methods(init_file, target_files=target_files) if os.path.exists(init_file) else None
    except Exception as e:
        print(f"[Warning] Failed to load init file '{init_file}': {type(e).__name__} - {e}")
        methods_by_file = None

    if methods_by_file is None: 
        methods_by_file = init(
            rpHelper, 
            intermediate_dir, 
            target_files = target_files,  
            file_types = file_types, 
            target_types = None # will focus on the default target type set, which is only MethodDeclaration
        )
        initial_to_latest: Dict[str,str] = {fpath:fpath for fpath in methods_by_file}
        latest_to_initial: Dict[str,str] = {fpath:fpath for fpath in methods_by_file}
        # need to convert methods_by_file into dictionary 
        save_init_methods(methods_by_file, init_file)
    initial_to_latest, latest_to_initial, mth_uid_to_latest_lnokey, mth_latest_lnokey_to_uid = get_tracking_keys(methods_by_file)
    files_to_track = set(list(methods_by_file.keys()))
    time_window = time_window * (60*60*24) # in seconds 
    start_time = None
    # initialisation and then start from the "initialised" commit
    # since we are doing in reverse, if the commit is the last one, we should go back once 
    commits = list(rpHelper.pydriller_repo.traverse_commits())
    print (f"[Start] Traversal starts from {commits[0].hash}")
    if os.path.exists(INTERMEDIATE_RESULTFILE): 
        with open(INTERMEDIATE_RESULTFILE, 'rb') as f:
            inter_results = pickle.load(f)
        methods_by_file = inter_results['output']
        initial_to_latest = inter_results['index']['initial_to_latest']
        latest_to_initial = inter_results['index']['latest_to_initial']
        mth_uid_to_latest_lnokey = inter_results['index']['mth_uid_to_latest_lnokey']
        mth_latest_lnokey_to_uid = inter_results['index']['mth_latest_lnokey_to_uid']
        files_to_track = inter_results['files_to_track']
        last_commit_idx = inter_results['last_commit_idx']
        curr_processed_modfiles = inter_results['curr_processed_modfiles']
        
        delta = last_commit_idx
    else: 
        last_commit_idx = 0
        delta = 0
        curr_processed_modfiles = set()
        
    if last_commit_idx is not None:
        start_time = commits[0].author_date
        commits = commits[last_commit_idx:]
    
    already_processed = curr_processed_modfiles
    curr_processed_modfiles = set()
    initial_targets = {fpath_key:[mth.lno_key for mth in mths.values() if mth.lno_key is not None] for fpath_key,mths in methods_by_file.items()}
    # chnage init start
    changes_pcommit = {}
    # change init end
    try:
        for i, commit in enumerate(tqdm(commits)):
            print (f"At commit: {commit.hash}")
            last_commit_idx = i + delta # temp 
            curr_processed_modfiles = set() # re-init # temp 
            
            prev_commit = commits[i+1] if i < len(commits) - 1 else None
            if start_time is None: # the first commit
                start_time = commit.author_date

            # Check the stop condition
            time_difference = abs((commit.author_date - start_time).total_seconds())
            if time_difference > time_window:
                print (f"[Out-of-time-window] {round(time_difference/(60*60*24),2)} > {time_window}. Stop tracking at {commit.hash}")
                break
            
            # set change start
            parsed_info = GitRepoHelper.parse_commit(commit)
            chg_inst = Change(
                commit.hash, 
                author=parsed_info['author_name'], 
                author_email=parsed_info['author_email'], 
                author_date=parsed_info['author_date'], 
                msg = parsed_info['msg'], 
                modified_fpaths=parsed_info['modified_fpaths']
            )
            # setchange set end 
            fpath_to_update:Dict[str, str] = {} # afpath(src) : bfpath(dst)
            #to_update_mid_new = []
            for modified_file in commit.modified_files:
                # skip those alreaady processed (while continuing with inter) 
                if i == 0:
                    if modified_file.filename in already_processed: # usage of filenmae is temp
                        continue # skip as it already has been handled
                ## inter end
                
                src_path = modified_file.new_path
                dst_path = modified_file.old_path
                #if src_path not in files_to_track or not src_path.endswith(".java"): # here, the order matter
                if not modified_file.filename.endswith(".java") or src_path not in files_to_track: 
                    continue
                #print (modified_file.new_path not in list(methods_by_file.keys()))
                # prepare the update for later 
                fpath_to_update[src_path] = dst_path
                # check whether we need to further process this file's changes (already did if dst_path is None) 
                if dst_path is None: 
                    # the file no longer exist in the past, meaning the current commit is the commit where the file is first introduced 
                    print (f"[Newly Introduced] A {src_path} is introduced at {commit.hash}")
                    print (f"                   End the tracking for {src_path}")
                    continue
                
                # We need to get the file as "as-is" instead of decoded string to match with the outputs provided by gumtree
                out = GitRepoHelper.get_file_content(rpHelper.get_git_commit(commit.hash), src_path)#, encoding='utf-8')
                if out is None:
                    raise ValueError(f"Failed to retrive {src_path}")
                srcode_src, _ = out 
                out = GitRepoHelper.get_file_content(rpHelper.get_git_commit(prev_commit.hash), dst_path)#, encoding='utf-8')
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
                updated_mth_range = gumtreeUtils.diff(
                    image=GUMTREE_IMAGE, 
                    srcfpath=local_src_path, #modified_file.new_path, # src_path  -> acutally this should be local path 
                    dstfpath=local_dst_path, #modified_file.old_path, # dst_path 
                    platform=GUMTREE_PLATFORM, 
                    host_path=str(host_path),
                    container_path=container_path
                ) 
                # old_path: not in the current commit (the previosu commit) 
                # new_path: in the current commit (and thereby, Method.start_lno and Method.end_lno are from the file of the curretn commit, and thereby form "new")
                fpath_key = latest_to_initial[src_path]
                methods_to_check = methods_by_file[fpath_key]
                
                # tmth_diff_range is for updated method, and mth_lno_key is before the updated
                # Get a list of modified methods 
                modified_mths = modified_file.changed_methods
                end_lno_of_modified_mths = {mth.end_line:mth.long_name for mth in modified_mths}
                #
                to_update_mid_new = []
                #
                _init_targets_to_replace = []
                for mth_lno_key, tmth_diff_range in updated_mth_range.items(): 
                    ####mth_at_last_mod = mth.long_name # from the src commit (not dst)
                    # update for the next parsing
                    next_path = dst_path
                    # get next information
                    next_start_pos = tmth_diff_range.get('start_pos')
                    next_end_pos = tmth_diff_range.get('end_pos')
                    next_start_lno = tmth_diff_range.get('start_lno')
                    next_end_lno = tmth_diff_range.get('end_lno')
                    #
                    new_label = updated_mth_range.get('label') # from the src commit (not dst) 
                    if next_start_pos is None: 
                        next_content = None
                        new_mth_lno_key = None
                    else:
                        next_content = srcode_dst[next_start_pos:next_end_pos]
                        new_mth_lno_key = f"{next_start_lno}-{next_end_lno}"

                    # update method node (TMethod)
                    try:
                        mth_uid = mth_latest_lnokey_to_uid[fpath_key][mth_lno_key]
                    except KeyError: # as e:
                        # "Currently" two cases are possible: 
                        #   - failure to map in the previous case
                        #   - or, the method was newly introduced at the commit (src_code side), so not in the original target
                        if fpath_key in mth_latest_lnokey_to_uid.keys():
                            if mth_lno_key in initial_targets[fpath_key]:
                                print (f"[Log] Failure on mapping at {commit.hash}: {mth_lno_key} from {src_path} is missing") 
                        else:
                            print (f"[Log] Failure on mapping at {commit.hash}: {fpath_key} representing {src_path} is missing")
                        continue
                    
                    # Start updating
                    tmth_node = methods_to_check[mth_uid]   
                    # set mth change before updating 
                    has_mth_modified = tmth_node.end_lno in end_lno_of_modified_mths
                    if has_mth_modified:
                        mth_raw_str_diff = Change.get_diff(tmth_node.content, next_content, reverse=True)
                        chg_inst.set_mth_diff(
                            src_path, 
                            end_lno_of_modified_mths[tmth_node.end_lno], 
                            tmth_node.init_lno_key, 
                            mth_raw_str_diff
                        )
                    # end 
                    tmth_node.update(
                        commit_hash=commit.hash if has_mth_modified else None,  # THIS SHOULD BE ONLY FOR THOSE MODIFIED METHODS
                        new_file_path=next_path, 
                        new_label=new_label,  
                        new_start=next_start_pos, 
                        new_end=next_end_pos, 
                        new_start_lno=next_start_lno, 
                        new_end_lno=next_end_lno, 
                        new_content=next_content, 
                        attribute_to_force='all'
                        #new_mth_name_at_last_mod=mth_at_last_mod, 
                    )
                    # prepare the updates of the global mth keys 
                    if new_mth_lno_key is None:
                        new_mth_lno_key = f"None-{mth_lno_key}-{commit.hash}"
                    else:
                        _init_targets_to_replace.append(new_mth_lno_key)
                        
                    to_update_mid_new.append((mth_uid, new_mth_lno_key))
                    print (f"[Update] Will updated from {mth_lno_key} -> {new_mth_lno_key} for {mth_uid}")
                    # mth_uid_to_latest_lnokey[fpath_key][mth_uid] = new_mth_lno_key # update 
                    # ##
                    # del mth_latest_lnokey_to_uid[fpath_key][mth_lno_key] # currently, accidently delete the correct keys
                    # print (f"[Update] Will updated from {mth_lno_key} -> {new_mth_lno_key} for {mth_uid}")
                    # if new_mth_lno_key is not None:
                    #     mth_latest_lnokey_to_uid[fpath_key][new_mth_lno_key] = mth_uid
                    # else: # the case where the method doesn't exist any more
                    #     mth_latest_lnokey_to_uid[fpath_key][f"None-{mth_lno_key}-{commit.hash}"] = mth_uid
                update_method_to_track(
                        fpath_key, 
                        mth_latest_lnokey_to_uid, #[fpath_key], 
                        mth_uid_to_latest_lnokey, #[fpath_key], 
                        to_update_mid_new
                )    
                curr_processed_modfiles.add(src_path)
                initial_targets[fpath_key] = _init_targets_to_replace
            # update files_to_track  and methods to track for each 
            update_files_to_track(initial_to_latest, latest_to_initial, files_to_track, fpath_to_update)
            changes_pcommit[commit.hash] = chg_inst # not sure whether will be used
            # save chg_inst
            save_file = os.path.join(OUTPUT_DIR, f"{commit.hash}.chgd_inst.json")
            chg_inst.save(save_file)
    except Exception:
        # save 
        with open(INTERMEDIATE_RESULTFILE, 'wb') as f:
            pickle.dump(
                {'output':methods_by_file, 
                 'index':{
                     'initial_to_latest': initial_to_latest, 
                     'latest_to_initial': latest_to_initial,
                     'mth_uid_to_latest_lnokey': mth_uid_to_latest_lnokey, 
                     'mth_latest_lnokey_to_uid': mth_latest_lnokey_to_uid
                     }, 
                 'files_to_track':files_to_track, 
                 'curr_processed_modfiles': curr_processed_modfiles,
                 'last_commit_idx':last_commit_idx
                }, f) 
            
        save_inter_result(
            methods_by_file, 
            files_to_track, 
            initial_to_latest, 
            latest_to_initial, 
            mth_uid_to_latest_lnokey, 
            mth_latest_lnokey_to_uid, 
            curr_processed_modfiles,
            last_commit_idx
            )
        
        traceback.print_exc()
        
    save_inter_result(
        methods_by_file, 
        files_to_track, 
        initial_to_latest, 
        latest_to_initial, 
        mth_uid_to_latest_lnokey, 
        mth_latest_lnokey_to_uid, 
        curr_processed_modfiles,
        last_commit_idx
        )
            
    return methods_by_file 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-out", "--output_dir", default = "results")
    parser.add_argument("-r", "--repo_dir", default = ".")
    parser.add_argument("-gumtree", "--gumtree_image", type = str, default = "gumtreediff/gumtree:latest")
    parser.add_argument("-gplt", "--gumtree_platform", type = str, default = 'linux/amd64')
    parser.add_argument("-tw", "--time_window", default = 30, help = "a time window for change-tracking; in days")
    args = parser.parse_args()
    
    # set 
    repo_path = "/Users/jeongjusohn/temp/lang_7b" #args.repo_dir
    GUMTREE_IMAGE = args.gumtree_image 
    GUMTREE_PLATFORM = args.gumtree_platform
    
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    host_path = "/Users/jeongjusohn/temp"
    init_file = os.path.join(OUTPUT_DIR, 'init_file.json')
    INTERMEDIATE_RESULTFILE  = INTERMEDIATE_RESULTFILE.format(OUTPUT_DIR)
    # testing 
    #target_types = None # Currently this run.py is for changeability analyses, meaning only for method-level [] # will look everything
    methods_by_file = run(
        repo_path, 
        host_path, 
        target_files = None, 
        file_types = ['.java'], 
        time_window = 365*2, 
        init_file = init_file 
    )
    with open(os.path.join(OUTPUT_DIR, 'test.pkl'), 'wb') as f:
        pickle.dump(methods_by_file, f)