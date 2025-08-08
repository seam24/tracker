""" 
Maintain changes 
"""
from typing import List, Dict, Optional 
import json 

class Change:
    def __init__(self, commit_hash:str, filepath:Optional[str]=None, from_filepath:bool = False):
        """_summary_

        Args:
            commit_hash (str): _description_
            filepath (Optional[str], optional): _description_. Defaults to None.
            from_filepath (bool, optional): _description_. Defaults to False.
        """
        if from_filepath and bool(filepath):
            self.init_from_file(filepath) 
        else:
            self.commit_hash = commit_hash
            self.mod_files = {}
            
    def init_from_file(self, filepath:str):
        """_summary_

        Args:
            filepath (str): _description_

        Raises:
            Exception: _description_
        """
        try:
            with open(filepath) as f:
                data = json.load(filepath)
        except FileNotFoundError: 
            raise Exception(f"[FileNotFoundError] {filepath} does not exist")

        self.commit_hash = data.get('commit_hash')
        self.mod_files = data.get('chgspfile')

    def save(self, filepath:Optional[str]=None):
        """_summary_

        Args:
            filepath (Optional[str], optional): _description_. Defaults to None.
        """
        # save both the commit_hash and modifications 
        if filepath is None:
            filepath = f"{self.commit_hash}.changes.json"
        
        with open(filepath, 'w') as f: 
            json.dump({
                'commit_hash':self.commit_hash, 
                'chgspfile':self.mod_files,
                }, f)
         

class ChangeSet:
    def __init__(self):
        self.changes:Dict[str, Change] = {} # key (str) will be commit_has
    
    def update(self, chg_inst:Change):
        self.changes.update({chg_inst.commit_hash:chg_inst})
    
    def get_commit_hashes(self) -> List[str]:
        return list(self.changes)
    
    def save(self, filepath:Optional[str]): 
        # need to format change inst wihtin 
        try:
            with open(filepath, 'w') as f:
                pass 
        except FileNotFoundError:
            raise Exception(f"[FileNotFoundError] {filepath} does not exist") 
    