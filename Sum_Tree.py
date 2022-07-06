class Node:
    def __init__(self, left, right, is_leaf: bool = False, idx=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        """
        Creates a leaf node
        :returns: the leaf node with given value and index
        """
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf


def create_tree(input: list) -> tuple:
    """
    Create a SumTree
    :returns: top node of the SumTree and the leaf_nodes
    """
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]
    return nodes[0], leaf_nodes


def retrieve(value: float, node: Node) -> Node:
    """
    Retrieve a node given a float-value
    """
    if node.is_leaf:
        # leaf node reached
        return node
    if node.left.value >= value:
        # the value to retrieve is within the left side of the tree
        return retrieve(value, node.left)
    else:
        # go to right side of the tree, subtract sum of left side from value
        return retrieve(value - node.left.value, node.right)


def update(node: Node, new_value: float) -> None:
    """
    Update value of node and all parent nodes by that same difference
    """
    change = new_value - node.value
    node.value = new_value
    propagate_changes(change, node.parent)


def propagate_changes(change: float, node: Node):
    """
    Helper function for update to propagate up the SumTree
    """
    node.value += change
    if node.parent is not None:
        propagate_changes(change, node.parent)