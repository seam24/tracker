""" 
Maintain changes 
"""
from typing import List, Dict, Optional, Union
import json 
import difflib 

class Change:
    def __init__(self, 
                 commit_hash:str, 
                 author:Optional[str]= None, 
                 author_email:Optional[str]= None, 
                 author_date:Optional[str]= None,
                 msg:Optional[str]= None,
                 modified_fpaths:Optional[List[str]] = None,
                 save_fpath:Optional[str]=None, 
                 from_saved:bool = False, 
                 ):
        """_summary_

        Args:
            commit_hash (str): _description_
            filepath (Optional[str], optional): _description_. Defaults to None.
            from_filepath (bool, optional): _description_. Defaults to False.
        """
        if from_saved and bool(save_fpath):
            self.init_from_file(save_fpath) 
        else:
            self.commit_hash = commit_hash
            self.author = author
            self.author_email = author_email
            self.author_date = author_date
            self.msg = msg
            self.modified_fpaths = modified_fpaths
        self.mths_diff = {fpath:dict() for fpath in self.modified_fpaths} # mth_longname:mth_diff
        self.mth_ids_of_chgds = {fpath:dict() for fpath in self.modified_fpaths}
    
    @staticmethod
    def get_diff(content_a:Optional[str], content_b:Optional[str], reverse:bool = False) -> str:
        # normalize newlines and remove trailing carriage returns
        def normalize(s: Optional[str]) -> list[str]:
            if s is None:
                return []
            return [line.rstrip('\r') for line in s.splitlines()]
    
        old_lines = normalize(content_a)
        new_lines = normalize(content_b)
        
        diff_lines = list(
            difflib.unified_diff(
                old_lines, 
                new_lines,
                fromfile="old" if not reverse else "new", tofile="new" if not reverse else "old",
                lineterm=""
            )
        )
        unwanted_prefixes = ("@@", "--- new", "+++ old")
        cleaned_diff = "\n".join(
            line for line in diff_lines
            if not line.startswith(unwanted_prefixes)
        )
        return cleaned_diff #if as_str else diff_lines
        
    def set_mth_diff(self, modified_fpath:str, mth_longname:str, mth_id:str, mth_diff:Optional[str]):
        if self.mths_diff.get(modified_fpath) is None:
            raise KeyError(f"[Failure] {modified_fpath} was not modified at {self.commit_hash}")
        else:
            #self.mth_ids_of_chgds.append(mth_id)
            self.mth_ids_of_chgds[modified_fpath][mth_id] = mth_longname
            self.mths_diff[modified_fpath][mth_longname] = mth_diff
    
    # def get_mths_diff_as_str(self) -> str:
    #     # format all method diff to string (for prompting .. not sure whether I will have this here)
    #     pass 
    
    def save(self, to_save_file:str):
        to_save = {
                'commit_hash': self.commit_hash, 
                'author': self.author, 
                'author_email': self.author_email, 
                'author_date': self.author_date, 
                'msg ': self.msg, 
                'modified_fpaths':self.modified_fpaths, 
                'mths_diff': self.mths_diff , 
                'mth_ids_of_chgds': self.mth_ids_of_chgds
        }
        with open(to_save_file, 'w', encoding = 'utf-8') as f:
            json.dump(to_save, f, indent = 2)
        print (f"[Log] Change instance of {self.commit_hash} saved to {to_save_file}")
    
    def init_from_file(self, save_file:str):
        """_summary_

        Args:
            save_file (str): _description_

        Raises:
            Exception: _description_
        """
        try:
            with open(save_file) as f:
                data = json.load(save_file)
        except FileNotFoundError as e: 
            raise type(e)(f"[FileNotFoundError] {save_file} does not exist") from e

        self.commit_hash = data.get('commit_hash')
        self.author = data.get('author')
        self.author_email = data.get('author_email')
        self.author_date = data.get('author_date')
        self.msg = data.get('msg')
        self.modified_fpaths = data.get('modified_fpaths')
        self.mths_diff = data.get('mths_diff')
        self.mth_ids_of_chgds = data.get('mth_ids_of_chgds')
        