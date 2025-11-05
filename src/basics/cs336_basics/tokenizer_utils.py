class Node:
    """
    helper class for linked list
    """
    def __init__(self):
        self.val = None
        self.next = None
        self.prev = None

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def create_linked_lst(word_bytes):
    head = None
    current = None
    for i in range(len(word_bytes)):
        node = Node()
        node.val = word_bytes[i:(i + 1)]
        if head is None:
            head = node
            current = head
        else:
            current.next = node
            node.prev = current
            current = node

    return head
