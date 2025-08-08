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
from utils.fileUtils import save_content, get_content
import traceback 
import pickle 

GUMTREE_IMAGE = None
GUMTREE_PLATFORM = None 
INTERMEDIATE_RESULTFILE = os.path.join('results', "inter_results.pkl")

def init(
    rpHelper:GitRepoHelper, 
    intermediate_dir:str, 
    target_files:Optional[List[str]]=None, 
    file_types:Optional[List[str]] = None,
    target_types:Optional[List[str]] = None, 
) -> Dict[str, Dict[str, TMethod]]:
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
            #print (f"Failed to retrieve and save the content of {target_file} in {temp_fpath}")
            raise Exception(f"Failed to retrieve and save the content of {target_file} in {temp_fpath}")
    
        # sucessfully saved the content 
        # - next, parse the content     
        # need to init
        mth_declrs = gumtreeUtils.parse(
            GUMTREE_IMAGE, 
            temp_fpath, 
            GUMTREE_PLATFORM, 
            host_path=intermediate_dir, 
            container_path="/data", 
            target_types = target_types,
        )
        #print ("mth dec", mth_declrs)
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
                #methods_by_file[target_file][mth_node.lno_key] = mth_node 
                methods_by_file[target_file][idx] = mth_node
                # clean up 
            clean(temp_fpath) 
        elif mth_declrs is not None: # empty list 
            print (f"[Log] No methods in {target_file}")
        else:
            raise Exception(f"[Error] Failed to initialise method declarations in {target_file}") 
    return methods_by_file


def analyse_changes(content_bfr:str, content_aft):
    #from utils.gumtreeUtils import 
    pass 


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
            old_path = latest_to_initial[afpath]
            initial_to_latest[old_path] = bfpath # can contain None
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
    global GUMTREE_PLATFORM, GUMTREE_IMAGE
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
        #save_content(methods_by_file, init_file, as_binary=True)
        save_init_methods(methods_by_file, init_file)
    initial_to_latest, latest_to_initial, mth_uid_to_latest_lnokey, mth_latest_lnokey_to_uid = get_tracking_keys(methods_by_file)
    files_to_track = set(list(methods_by_file.keys()))

    time_window = time_window * (60*60*24) # in seconds 
    start_time = None
    # initialisation and then start from the "initialised" commit
    # since we are doing in reverse, if the commit is the last one, we should go back once 
    commits = list(rpHelper.pydriller_repo.traverse_commits())
    print (f"[Start] Traversal starts from {commits[0].hash}")
    # temporary 
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
    # temporary end
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

            if commit.hash in 'b168f29106d112da0bbdb3a070be30f9ee3cb36e': 
                for modified_file in commit.modified_files:
                    print (modified_file.filename)
                    print ("src", modified_file.new_path)
                    print ("dst", modified_file.old_path)
                    print (modified_file.change_type)
                    print ()

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
                sys.exit()    
            
            fpath_to_update:Dict[str, str] = {} # afpath(src) : bfpath(dst)
            for modified_file in commit.modified_files:
                ## INTER 
                if i == 0:
                    if modified_file.filename in already_processed: # usage of filenmae is temp
                        continue # skip as it already has been handled
                    
                src_path = modified_file.new_path
                dst_path = modified_file.old_path
                if not modified_file.filename.endswith(".java") or src_path not in files_to_track: # order matter
                    continue
                # prepare the update for later 
                fpath_to_update[src_path] = dst_path
                # check whether we need to further process this file's changes (already did if dst_path is None) 
                if dst_path is None: 
                    # the file no longer exist in the past, meaning the current commit is the commit where the file is first introduced 
                    print (f"[Newly Introduced] A {modified_file} is introduced at {commit.hash}")
                    print (f"                   End the tracking for {modified_file}")
                    continue
                
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
                updated_mth_range = gumtreeUtils.diff(
                    image=GUMTREE_IMAGE, 
                    srcfpath=local_src_path, #modified_file.new_path, # src_path  -> acutally this should be local path 
                    dstfpath=local_dst_path, #modified_file.old_path, # dst_path 
                    platform=GUMTREE_PLATFORM, 
                    host_path=str(host_path),
                    container_path=container_path
                ) # ... -> should it be changed to .. new-key to old-key? but at the same time .. we need to consider that we are visiting the commits in "reverse"     order. so, "start_lno-end_lno" -> so .. the (new) start_lno ane end_lno retried here is acutaly those of before
                # meaning that .. for gumtree-diff, we need to make sure that we are comparing in reverse older (or not...? need to thingk)
                # now thinking .. we are doing in reverse, and start_lno and end_lno is from that commit, and the info will be udpated, so, mdofied or not can be simply    cheked aby start_lno and end_lno of pydriller 
                # old_path: not in the current commit (the previosu commit) 
                # new_path: in the current commit (and thereby, Method.start_lno and Method.end_lno are from the file of the curretn commit, and thereby form "new")
                # if we do gumtree to be comparing backward: gumtreee {new_path} {old_path} (from the lastest to old), then "other_pos" will be about {old_path}
                # updating 
                fpath_key = latest_to_initial[src_path]
                methods_to_check = methods_by_file[fpath_key]

                # tmth_diff_range is for updated method, and mth_lno_key is before the updated
                # Get a list of modified methods 
                modified_mths = modified_file.changed_methods
                end_lno_of_modified_mths = set([mth.end_line for mth in modified_mths]) #
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
                    except KeyError:
                        if fpath_key in mth_latest_lnokey_to_uid.keys():
                            print (f"[Log] Failure on mapping at {commit.hash}: {mth_lno_key} from {src_path} is missing") 
                        else:
                            print (f"[Log] Failure on mapping at {commit.hash}: {fpath_key} representing {src_path} is missing")
                        continue
                        
                    tmth_node = methods_to_check[mth_uid]
                    tmth_node.update(
                        commit_hash=commit.hash if tmth_node.end_lno in end_lno_of_modified_mths else None,  # THIS SHOULD BE ONLY FOR THOSE MODIFIED METHODS
                        new_file_path=next_path, 
                        new_label=new_label,  
                        new_start=next_start_pos, 
                        new_end=next_end_pos, 
                        new_start_lno=next_start_lno, 
                        new_end_lno=next_end_lno, 
                        new_content=next_content, 
                        attribute_to_force='all'
                    )
                    # update global keys 
                    mth_uid_to_latest_lnokey[fpath_key][mth_uid] = new_mth_lno_key # update 
                    del mth_latest_lnokey_to_uid[fpath_key][mth_lno_key]
                    if new_mth_lno_key is not None:
                        mth_latest_lnokey_to_uid[fpath_key][new_mth_lno_key] = mth_uid
                    else:
                        mth_latest_lnokey_to_uid[fpath_key][f"None-{mth_lno_key}-{commit.hash}"] = mth_uid
                # temp 
                curr_processed_modfiles.add(src_path) 
            update_files_to_track(initial_to_latest, latest_to_initial, files_to_track, fpath_to_update)
    except Exception:
        # save 
        # initial_to_latest, latest_to_initial, mth_uid_to_latest_lnokey, mth_latest_lnokey_to_uid = get_tracking_keys(methods_by_file)
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
        traceback.print_exc()
    
    # for continuous colleciton 
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
            
    return methods_by_file 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-out", "--output_dir", default = "results")
    parser.add_argument("-r", "--repo_dir", default = ".")
    parser.add_argument("-gumtree", "--gumtree_image", type = str, default = "gumtreediff/gumtree:latest")
    parser.add_argument("-gplt", "--gumtree_platform", type = str, default = 'linux/amd64')
    args = parser.parse_args()
    
    # set 
    repo_path = "/Users/jeongjusohn/temp/lang_1b" #args.repo_dir
    GUMTREE_IMAGE = args.gumtree_image 
    GUMTREE_PLATFORM = args.gumtree_platform
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    host_path = "/Users/jeongjusohn/temp"
    init_file = os.path.join(output_dir, 'init_file.json')
    # testing 
    #target_types = None # Currently this run.py is for changeability analyses, meaning only for method-level [] # will look everything
    methods_by_file = run(
        repo_path, 
        host_path, 
        target_files = None, #['src/main/java/org/apache/commons/lang3/reflect/FieldUtils.java'],
        time_window = 365*3, 
        init_file = init_file 
    )
    with open(os.path.join(output_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(methods_by_file, f)