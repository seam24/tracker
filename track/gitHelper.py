from pydriller import Repository
from pydriller.domain.commit import Commit as pydrillerCommit
from git import Repo, Commit
from typing import Optional, Tuple, Dict, List, Union

class GitRepoHelper:
    def __init__(self, repo_path:str, order:Optional[str]=None, only_no_merge:bool=False): 
        self.repo_path = repo_path
        self.repo = Repo(repo_path)
        self.head_commit = self.repo.head.commit.hexsha
        self.pydriller_repo = Repository(repo_path, order = order, only_no_merge=only_no_merge)
    # def get_commit_metadata(self, commit_hash):
    #     for commit in self.pydriller_repo.traverse_commits():
    #         if commit.hash == commit_hash:
    #             return {
    #                 'author': commit.author.name,
    #                 'date': commit.committer_date,
    #                 'message': commit.msg
    #             }
    #     return None    
    
    def get_git_commit(self, commit_hash):
        return self.repo.commit(commit_hash)
    
    def _get_file_content(self, commit_hash, file_path, encoding: Optional[str] = None) -> Optional[Tuple[str, str]]:
        commit = self.repo.commit(commit_hash)
        return self.get_file_content(commit, file_path, encoding)
 
    def write_file_to_disk(self, commit_hash, file_path, dest_path, encoding: Optional[str] = None):
        from utils.fileUtils import save_content
        content, _ = self._get_file_content(commit_hash, file_path, encoding = encoding)
        if content:
            try: 
                #with open(dest_path, 'w') as f:
                #    f.write(content)   
                save_content(content, dest_path, as_is=True, encoding = 'utf-8') # save as-is & b/c later need to be given as input to Gumtree
                return True
            except Exception as e:
                print (e)
                return False
        else:
            print (f'[Failure] Failed to get the content of {file_path} from {commit_hash}')
        return False
         
    @staticmethod
    def get_file_content(commit:Commit, file_path, encoding: Optional[str] = None) -> Optional[Tuple[str, str]]:
        try:
            blob = commit.tree / file_path
            raw = blob.data_stream.read()
            # new
            encodings_to_try = ['utf-8', 'cp1252', 'latin1'] if encoding is None else [encoding]
            out, enc_used = None, None
            for enc in encodings_to_try:
                try:
                    out = raw.decode(encoding=enc)
                    enc_used = enc
                    break
                except UnicodeDecodeError as e:
                    print (e)
                    continue
            if out is None: 
                # Final fallback: read as bytes and decode as UTF-8 (ignoring errors)
                enc_used = 'utf-8'
                out = raw.decode(encoding=enc_used, erros='ignore')
            return out, enc_used
        except Exception as e:
            print (f"While working with {file_path}")
            print (e)
            return None
        
    @staticmethod
    def parse_commit(commit:pydrillerCommit) -> Dict[str, Union[str,List[str]]]:
        author_email:str = commit.author.email
        author_name:str = commit.author.name
        msg:str = commit.msg 
        author_date = str(commit.author_date)
        modified_fpaths:List[str] = list(map(lambda v:v.new_path, commit.modified_files)) # path at this commit
        return {'author_email':author_email, 'author_name':author_name, 'msg':msg, 'author_date':author_date, 'modified_fpaths':modified_fpaths}
        
     
