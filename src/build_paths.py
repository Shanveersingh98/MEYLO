def build_paths(tree, children, current, path):
    new_path = path + [current]
    child_indexes = children.get(tree[current]["message_id"], [])

    if not child_indexes:
        return [new_path]

    return [
        p
        for child in child_indexes
        for p in build_paths(tree, children, child, new_path)
    ]


def make_children(tree):
    children = {}

    for i, item in enumerate(tree):
        parent = item["parent_id"]
        children.setdefault(parent, []).append(i)

    return children


def process_tree(indexes, clean):
    tree = clean.select(indexes)
    children = make_children(tree)

    paths = []

    for root in children.get(None, []):
        paths.extend(build_paths(tree, children, root, []))

    # Convert local indexes back to the original clean dataset indexes
    return [[indexes[i] for i in path] for path in paths]