from typing import Dict

def char_index_to_line(content: str) -> Dict[int, int]:
    """_summary_

    Args:
        content (str): _description_

    Returns:
        Dict[int, int]: _description_
    """
    chr_to_line: Dict[int, int] = {}
    line_num = 1
    idx = 0
    while idx < len(content):
        chr_to_line[idx] = line_num
        # add mapping for **newline** character -> if not, we will end up with mismatching info 
        if content[idx] == '\r':
            if idx + 1 < len(content) and content[idx + 1] == '\n': # '\r\n'
                chr_to_line[idx + 1] = line_num  # include \n in the same line
                idx += 1 # skip '\n'
            line_num += 1
        elif content[idx] == '\n':
            line_num += 1
        idx += 1 # normal increase
    return chr_to_line
