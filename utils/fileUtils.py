from typing import Union
import json 


def get_content(fpath: str, as_binary: bool = False, as_is: bool = False) -> Union[str, bytes]:
    """_summary_

    Args:
        fpath (str): _description_
        as_binary (bool, optional): _description_. Defaults to False.
        as_is (bool, optional): _description_. Defaults to False.

    Raises:
        FileNotFoundError: _description_
        OSError: _description_

    Returns:
        Union[str, bytes]: _description_
    """
    try:
        if as_is:
            encodings_to_try = ['utf-8', 'cp1252', 'latin1']
            for enc in encodings_to_try:
                try:
                    # reading text with newline='' to preserve character offsets exactly
                    with open(fpath, 'r', encoding=enc, newline='') as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            # final fallback: read as bytes and decode as UTF-8 (ignoring errors)
            with open(fpath, 'rb') as f:
                return f.read().decode('utf-8', errors='ignore')  
        elif as_binary:
            with open(fpath, 'rb') as f:
                return f.read()
        else:
            # force UTF-8 for normal text mode
            with open(fpath, 'r', encoding='utf-8') as f:
                return f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"[Error] File not found: {fpath}") from e
    except Exception as e:
        raise OSError(f"Failed to get content from a file: {fpath} -> {type(e).__name__}: {e}") from e


def save_content(fcontent: Union[str, bytes, dict], fpath: str, as_binary: bool = False, as_is: bool = False, encoding: str = 'utf-8') -> None:
    """_summary_

    Save content to file with various modes:
    - Text (default and force `utf-8` encoding)
    - Binary (if as_binary=True)
    - Exact offsets preserved (if as_is=True, disables newline translation)
    - In normal text mode (as_is=False, as_binary=False), content is always saved as UTF-8,
      regardless of the `encoding` argument.

    Args:
        fcontent (Union[str, bytes]): Content to save (str, bytes, or dict for JSON)
        fpath (str): Path to save to
        as_binary (bool, optional): Save in binary mode (writes raw bytes or pickled object). Defaults to False.
        as_is (bool, optional): Save text while preserving character offsets (newline=''). Defaults to False.
        encoding (str, optional): Encoding for text/JSON output (default: 'utf-8'). Defaults to 'utf-8'.
        
    Raises:
        FileNotFoundError: _description_
        OSError: _description_
    """
    if as_is and as_binary:
        raise ValueError("Cannot set both `as_is` and `as_binary` to True.")
    
    # actually need clean-up
    try:
        if as_is: # e.g., \r\n, \n, or \r
            save_text_preserve_offsets(fcontent, fpath, encoding = encoding)
        elif as_binary: 
            with open(fpath, 'wb') as f:
                if isinstance(fcontent, bytes): 
                    f.write(fcontent)
                else:
                    import pickle 
                    pickle.dump(fcontent, f)
        else:
            # force UTF-8 for normal text mode
            print(f"[Warning] Ignoring encoding='{encoding}' in text mode — saving as UTF-8")
            with open(fpath, 'w', encoding='utf-8') as f: # this translate newlines (b/c it is without "newline=''")
                if isinstance(fcontent, dict): 
                    json.dump(fcontent, f, ensure_ascii=False, indent = 2)
                else:
                    f.write(fcontent)
    except FileNotFoundError as e: 
        raise FileNotFoundError(f"Directory for {fpath} does not exist.") from e
    except Exception as e:
        import os 
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"[Failure-CleanUp] Cleaned up incomplete {fpath}")
        raise OSError(f"Failed to write file: {fpath} -> {type(e).__name__}: {e}") from e
    
def save_text_preserve_offsets(content: str, fpath: str, encoding:str = 'utf-8') -> None:
    """_summary_

    Args:
        content (str): _description_
        fpath (str): _description_
        encoding (str, optional): _description_. Defaults to 'utf-8'.
    """
    # writing text with newline='' to preserve character offsets exactly.
    with open(fpath, 'w', encoding=encoding, newline='') as f: # this translate newlines (b/c it is without "newline=''") -> so, keep \r\n, \r, \n, et
        ensure_ascii = False if encoding == 'utf-8' else True
        if isinstance(content, dict): 
            json.dump(content, f, ensure_ascii=ensure_ascii, indent=2)
        else:
            f.write(content)

