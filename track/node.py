"""
Nodes  
""" 
from typing import List, Dict, Tuple, Optional, Union
from utils.fileUtils import get_content


class TNode: 
    def __init__(self, 
                 ntype:str, 
                 file_path:str, 
                 pos:int, 
                 length:int, 
                 children:List[Dict], 
                 label:Optional[str] = None, 
                 local_file_path:Optional[str] = None, 
                 file_content:Optional[str] = None,
                 set_lno_info:bool = False
        ):
        """_summary_
        - Currenlty self.label is from the commit that has been visited just before instead of the current commit

        Args:
            ntype (str): _description_
            pos (int): _description_
            length (int): _description_
            children (List[Dict]): _description_
            label (Optional[str], optional): _description_. Defaults to None.
        """
        self.type = ntype
        self.file_path = file_path
        self.label = label
        self.start = pos # start index 
        self.length = length 
        self.end = self.start + self.length # end index -> content[start:end (start + length)]
        self.pos_key = f"{self.start}-{self.end}"
        self.children = children
        # set content 
        self.content = None
        self.start_lno, self.end_lno = None, None
        self.lno_key = None
        if bool(file_path) | bool(file_content):
            status = self.set_content(local_file_path=local_file_path, file_content=file_content, set_lno_info = set_lno_info)
            if not status: 
                raise Exception(f"[Error] Failed to set the content of {self.type} ({self.pos_key})")
        # set change-related variables 
        self.chgd_ats = [] # a list of commits that modified the node 
        # set initial key attributes (these won't change)
        self.init_pos_key = self.pos_key
        self.init_lno_key = self.lno_key
        self.init_content = self.content
           
    @classmethod
    def from_dict(cls, d:Dict, asTNodes:Optional[List[str]] = None) -> "TNode":
        """_summary_

        Args:
            d (Dict): _description_
            asTNodes (Optional[List[str]], optional): _description_. Defaults to None.

        Returns:
            TNode: _description_
        """
        obj = cls.__new__(cls) # empty instance without calling __init__ 
        for k, v in d.items():
            if v is not None and asTNodes is not None and k in asTNodes:
                setattr(obj, k, TNode.from_dict(v)) 
            else:
                setattr(obj, k, v)
        return obj

    def to_dict(self) -> Dict:
        """
        Convert the instance to a serialisable object (dict)

        Returns:
            Dict: _description_
        """
        converted = {}
        for k, v in self.__dict__.items():
            if isinstance(v, TNode): # e.g., self.javadoc and self.block from TMethod
                converted[k] = v.to_dict()
            else:
                converted[k] = v
        return converted
    
    def set_content(self, 
                    local_file_path:Optional[str]=None, 
                    file_content:Optional[str] = None, 
                    idx_range:Optional[Tuple[int, int]] = None, 
                    set_lno_info:bool = False
        ) -> bool:
        """_summary_

        Args:
            file_path (Optional[str], optional): _description_. Defaults to None.
            file_content (Optional[str], optional): _description_. Defaults to None.
            idx_range (Optional[Tuple[int, int]], optional): a tuple of start and end line numbers. 
                If given, it will be prioritised. Defaults to None.
            set_lno_info (bool, optional): _description_. Defaults to False.

        Returns:
            bool: _description_
        """
        if local_file_path is not None:
            try:
                file_content = get_content(local_file_path, as_is = True)
            except FileNotFoundError:
                if file_content is None:
                    print ("[Warning] To set up the content, either `local_file_path` or `file_content` should be given") 
                    print (f"          - Invalid file_path: {local_file_path} does not exist") 
                    return False
            except Exception as e: # other exception (e.g., OS)
                print (e)
                return False
        elif file_content is None: 
            print ("[Warning] To set up the content, either `local_file_path` or `file_content` should be given")
            return False 

        _start = self.start if idx_range is None else idx_range[0]
        _end = self.end if idx_range is None else idx_range[1]
        #print ("currently", _end, len(file_content))#type(_end))
        #print (self.start, self.length, self.end)
        if len(file_content) >= _end:
            self.content = file_content[_start:_end] 
            if set_lno_info: 
                from utils.parseUtils import char_index_to_line 
                chr_to_line = char_index_to_line(file_content)
                self.start_lno = chr_to_line[_start]
                self.end_lno = chr_to_line[_end]
                self.lno_key = f"{self.start_lno}-{self.end_lno}"
            return True
        else:
            print (f"[Out-of-index] {_end} (last-index) vs. {len(file_content)} (file_content)")
            return False 
    
    def update_chgd_ats(self, commit_hash:Optional[str]):
        """_summary_

        Args:
            commit_hash (str): _description_
        """
        if commit_hash is not None:
            #print ("Updated", commit_hash)
            self.chgd_ats.append(commit_hash)

    def conditional_update(self, attr_name: str, new_value, force: bool = False):
        """_summary_

        Args:
            attr_name (str): _description_
            new_value (_type_): _description_
        """
        if new_value is not None or force:
            setattr(self, attr_name, new_value)

    def update(self,
               commit_hash:Optional[str] = None, 
               new_file_path:Optional[str] = None, 
               new_label:Optional[str] = None, 
               new_start:Optional[int] = None,  
               new_end:Optional[int] = None,  
               new_start_lno:Optional[int] = None,  
               new_end_lno:Optional[int] = None, 
               new_content:Optional[str] = None, 
               attribute_to_force:Optional[Union[str, List[str]]] = None
        ) -> bool:
        # this is processing for the next update (not the current one)
        force_flag = {
            'file_path':False, 
            'label':False, 
            'start':False, 
            'end':False, 
            'length':False, 
            'pos_key':False, 
            'start_lno':False, 
            'end_lno':False, 
            'lno_key':False, 
            'content':False
        }
        if attribute_to_force is not None:
            if attribute_to_force == 'all': 
                attribute_to_force = list(force_flag.keys())
            for attr in attribute_to_force:
                if attr in force_flag:
                    force_flag[attr] = True
        try:
            # update the chgdAt information 
            self.update_chgd_ats(commit_hash)
            # update 
            # - self.children will not updated for now 
            self.conditional_update('file_path', new_file_path, force=force_flag['file_path'])  
            self.conditional_update('label', new_label, force=force_flag['label'])  
            
            # Update position-related 
            self.conditional_update('start', new_start, force=force_flag['start'])  
            self.conditional_update('end', new_end, force=force_flag['end'])  
            new_length = new_end - new_start if new_start is not None else None
            self.conditional_update('length', new_length, force=force_flag['length'])
            new_pos_key = f'{new_start}-{new_end}' if new_length is not None else None
            self.conditional_update('pos_key', new_pos_key, force=force_flag['pos_key'])  
            # Update line-related
            self.conditional_update('start_lno', new_start_lno, force=force_flag['start_lno'])  
            self.conditional_update('end_lno', new_end_lno, force=force_flag['end_lno'])  
            new_lno_key = f'{new_start_lno}-{new_end_lno}' if self.start_lno is not None else None
            self.conditional_update('lno_key', new_lno_key, force=force_flag['lno_key'])
            
            # Update content
            self.conditional_update('content', new_content, force=force_flag['content'])
            
            return True
        except Exception as e:
            raise type(e)(f"[Failure] Something went wrong during the update\n\t{e}") from e


class TMethod(TNode):
    """A node with type "MethodDeclaration" 
    Args:
        TNode (_type_): _description_
    """
    def __init__(self, 
                 file_path:str, 
                 pos:int, 
                 length:int, 
                 children:List[Dict], 
                 label:Optional[str] = None, 
                 local_file_path:Optional[str] = None, 
                 file_content:Optional[str] = None, 
        ):
        """_summary_
        - Currenlty self.label and self.mth mth_name_at_last_mod are from the commit that has been visited just before instead of the current commit
        
        Args:
            file_path (str): _description_
            pos (int): _description_
            length (int): _description_
            children (List[Dict]): _description_
            label (Optional[str], optional): _description_. Defaults to None.
            local_file_path (Optional[str], optional): _description_. Defaults to None.
            file_content (Optional[str], optional): _description_. Defaults to None.
        """
        super().__init__(
            'MethodDeclaration', 
            file_path, 
            pos, 
            length, 
            children, 
            label, 
            local_file_path=local_file_path, 
            file_content=file_content, 
            set_lno_info = True
        ) 
        
        # additionally set javadoc and block 
        self.block, self.javadoc = None, None
        # for javadoc and block, during the initialisatin, Javadoc and Block's content will be set 
        for child in self.children:
            if child.get('type') == 'Javadoc':
                self.javadoc = TNode(
                    child.get('type'),
                    file_path,  
                    int(child.get('pos')), 
                    int(child.get('length')), 
                    child.get('childeren'), 
                    child.get('label'), 
                    local_file_path=local_file_path, 
                    file_content=file_content)
            elif child.get('type') == 'Block':
                self.block = TNode(
                    child.get('type'), 
                    file_path, 
                    int(child.get('pos')), 
                    int(child.get('length')), 
                    child.get('childeren'), 
                    child.get('label'), 
                    local_file_path=local_file_path, 
                    file_content=file_content)
        self.mth_name_at_last_mod = None # "at" commit that changed the method (for initilisation, this is None (due to the initialisation via pasring))
        
    def update(self,
               commit_hash:Optional[str] = None, 
               new_file_path:Optional[str] = None, 
               new_label:Optional[str] = None, 
               new_start:Optional[int] = None,  
               new_end:Optional[int] = None,  
               new_start_lno:Optional[int] = None,  
               new_end_lno:Optional[int] = None, 
               new_content:Optional[str] = None,
               new_mth_name_at_last_mod:Optional[str] = None, 
               attribute_to_force:Optional[List[str]] = None
        ) -> bool:
        # this is processing for the next update (not the current one)
        #try:
        super().update(
            commit_hash = commit_hash, 
            new_file_path = new_file_path, 
            new_label = new_label, 
            new_start = new_start, 
            new_end = new_end, 
            new_start_lno = new_start_lno,
            new_end_lno = new_end_lno, 
            new_content = new_content, 
            attribute_to_force = attribute_to_force
        )
        try:
            force_flag = True if (attribute_to_force is not None) and ('mth_name_at_last_mod' in attribute_to_force) else False
            self.conditional_update('mth_name_at_last_mod', new_mth_name_at_last_mod, force = force_flag)
            return True
        except Exception as e:
            raise type(e)(f"[Failure] Something went wrong during the method update\n\t{e}") from e
    # def set_mthID(self):
    #     if self.label is not None:
    #         return self.label
    #     else:
    #         child_labels = []
    #         for child in self.children:
    #             if child['type'] == 'Block':
    #                 break 
    #             child_label = child.get('label')
    #             if child_label is not None: 
    #                 child_labels.append(child_label) 
    #         label = " ".join(child_label)
    #         return label if bool(label) else None 
    