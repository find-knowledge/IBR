class Node:
    def __init__(self, head):
        self.head = head

    def __str__(self):
        return str(self.head)

def get_proof_graph(proof_str, id):
    stack = []
    last_open = 0
    last_open_index = 0
    pop_list = []
    all_edges = []
    all_nodes = []

    proof_str = proof_str.replace("(", " ( ")
    proof_str = proof_str.replace(")", " ) ")
    proof_str = proof_str.split()

    new_proof_str = []
    str_dic = {}
    index = 0

    for x in proof_str:
        if x not in ['(', ')', '->', ' ( ', ' ) ', '[', ']']:
            new_proof_str.append(str(index))
            str_dic[str(index)] = x
            index += 1
        else:
            new_proof_str.append(x)
    proof_str = new_proof_str

    should_join = False
    for i in range(len(proof_str)):

        _s = proof_str[i]
        x = _s.strip()
        if len(x) == 0:
            continue

        if x == "(":
            stack.append((x, i))
            last_open = len(stack) - 1
            last_open_index = i
        elif x == ")":
            for j in range(last_open + 1, len(stack)):
                if isinstance(stack[j][0], Node):
                    pop_list.append((stack[j][1], stack[j][0]))

            stack = stack[:last_open]
            for j in range((len(stack))):
                if stack[j][0] == "(":
                    last_open = j
                    last_open_index = stack[j][1]

        elif x == '[' or x == ']':
            pass
        elif x == "->":
            should_join = True
        else:
            # terminal
            if x not in all_nodes:
                all_nodes.append(x)

            if should_join:

                new_pop_list = []
                # Choose which ones to add the node to
                for (index, p) in pop_list:
                    if index < last_open_index:
                        new_pop_list.append((index, p))

                    else:
                        all_edges.append((p.head, x))
                pop_list = new_pop_list

            stack.append((Node(x), i))

            should_join = False
    return all_nodes, all_edges, str_dic

def get_proof_graph_with_fail(proof_str, id):
    proof_str = proof_str[:-2].split("=")[1].strip()[1:-1]
    nodes = proof_str.split(" <- ")

    all_nodes = []
    all_edges = []
    for i in range(len(nodes)-1):
        all_nodes.append(nodes[i])
        if nodes[i+1] != "FAIL":
            all_edges.append((nodes[i+1], nodes[i]))
    # print(all_nodes)
    # print(all_edges)

    str_dic, str_dic_1 = {}, {}
    new_all_nodes, new_all_edges = [], []
    for index, node in enumerate(all_nodes):
        str_dic[str(index)] = node
        str_dic_1[node] = str(index)
        new_all_nodes.append(str(index))

    for index, edge in enumerate(all_edges):
        new_all_edges.append((str_dic_1[edge[0]], str_dic_1[edge[1]]))

    return new_all_nodes, new_all_edges, str_dic
