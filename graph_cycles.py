# for cycle detection I initially wanted to use python-graph-core
# but
#   >pip install --allow-unverified python-graph-core python-graph-core
# was kinda annoying with its need for this creepy --allow-unverified.
# Same for https://pypi.python.org/pypi/graph/0.4
# What did install easily was pip install altgraph, but this doesn't
# have cycle detection? 
# P.S.: pygraph aka python-graph does have cycle detection for digraphs
# with pygraph.algorithms.accessibility.mutual_accessibility finding
# strongly connected components in a digraph.
#
# Anyway, just use the usual
#   >pip install -r requirements.txt
#
# See also
# https://code.google.com/p/python-graph/issues/attachmentText?id=98&aid=980000000&name=find_all_cycles.py
# which I don't think is working correctly: emits different cycles
# every other time you run it. Then I tried to impl this myself for
# fun's sake, but that ended up being harder than I though: 
#
# General problem is described here:
#    http://en.wikipedia.org/wiki/Strongly_connected_component

import pdb

from pygraph.algorithms.cycles import find_cycle
from pygraph.classes.digraph import digraph
from pygraph.algorithms.accessibility import mutual_accessibility

def _graph2py_digraph(node2arc_targets):
    """ node2arc_targets is a simple digraph representation via dict() """
    graph = digraph()
    for node in node2arc_targets.keys():
        graph.add_node(node)
    for node, arc_targets in node2arc_targets.items():
        for arc_target in arc_targets:
            graph.add_edge((node, arc_target)) # insists on tuple
    return graph

def _find_one_pointless_cycle(node2arc_targets):
    # I guess find_cycle is cheaper than mutual_accessibility() if you
    # only care if there are cycles or not, but don't need the exact
    # strongly connected components """
    print(find_cycle(_graph2py_digraph(graph)))

def find_all_cycles(node2arc_targets):
    """ cycle aka strongly connected component of a digraph """
    result = mutual_accessibility(_graph2py_digraph(node2arc_targets))
    cycles = []
    nodes_in_cycles = set()
    for node, cycle in result.items():
        if len(cycle) <= 1:
            continue
        if node in nodes_in_cycles:
            #assert cycles.count(cycle) == 1
            continue
        cycles.append(set(cycle))
        for node in cycle:
            assert not node in nodes_in_cycles
            nodes_in_cycles.add(node)
    for cycle in cycles:
        print(node, cycle)
    pdb.set_trace()
    return cycles

# this is my own (pointless impl), written before I realized that
# mutual_accessibility does what I needed. Todo: find out what's the
# problem with my impl (fails with modularized boost deps). The problem
# is somewhere around build_spanning_tree, which I think lacks conns,
# and it's different conns since python's dict hashes are different in
# each process run: see 
#   http://stackoverflow.com/questions/15479928/why-is-the-order-in-python-dictionaries-and-sets-arbitrary
# "Note that as of Python 3.3, a random hash seed is used as well, 
# making hash collisions unpredictable to prevent certain types of 
# denial of service (where an attacker renders a Python server 
# unresponsive by causing mass hash collisions). "
def _find_all_cycles(graph):

    all_nodes = set(graph.keys())
    for src, dests in graph.items():
        for dest in dests:
            all_nodes.add(dest)
    print('input graph with', len(graph), 'src nodes,', 
        len(all_nodes), 'total nodes',
        sum(len(targets) for targets in graph.values()), 'edges')

    def is_connected(src, dest):
        """ return true if the 2 nodes are conncted directly or indirectly
            by the (directed) spanning tree
        """
        # Walk from dest to root, using reverse_spanning_tree.
        # If you encounter src then these are connected.
        if src == dest:
            return True
        node = dest
        while node in reverse_spanning_tree:
            node = reverse_spanning_tree[node]
            if node == src:
                return True
        return False

    def get_chain(src, dest):
        """ returns all nodes of of the spanning tree between the 2 given
            nodes.
            Precondition: is_connected == True, exception otherwise
        """
        assert is_connected(src, dest)
        chain = [dest]
        node = dest
        while node in reverse_spanning_tree:
            node = reverse_spanning_tree[node]
            chain.append(node)
            if node == src:
                assert chain[0] == dest and chain[-1] == src
                return chain 
        assert False

    def build_spanning_tree(graph):

        def dfs(src):
            visited.add(src)
            for dest in graph[src]:
                # something is buggy here, Todo
                if not dest in visited:
                    assert not dest in reverse_spanning_tree
                    reverse_spanning_tree[dest] = src
                    dfs(dest)

        visited = set() # nodes we did dfs from alrdy
        reverse_spanning_tree = {} # spanning subset of graph, dest node to src node
        for src in graph:
            if not src in visited:
                assert not src in reverse_spanning_tree
                #reverse_spanning_tree[src] = None
                dfs(src) # dfs or bfs, doesn't matter
        assert len(visited) == len(all_nodes)
        return reverse_spanning_tree

    # 'reverse' because it maps target nodes from the original graph
    # to source nodes, so this allows to cheaply walk the original graph
    # backwards
    reverse_spanning_tree = build_spanning_tree(graph)
    print('reverse_spanning_tree has', len(reverse_spanning_tree), 'edges')
    if False:
        print('edges of spanning tree')
        for dest, src in reverse_spanning_tree.items():
            assert dest in graph[src]
            assert is_connected(src, dest)
            print('  ', src, dest)

    # now walk over all graph edges again, identifying cycles
    # using the reverse_spanning_tree
    node2equivalence_class = {}
    for src in graph:
        for dest in graph[src]:
            #assert is_connected(src, dest) # direct or indirect conn via spanning tree
            if is_connected(dest, src):

                # reverse connection in spanning tree, so all these nodes 
                # are part of a cycle.
                #print('cycle between',src,dest)

                # detect maximum cycles now:
                print('getting chain for cycle edge', src, '->', dest)
                chain = get_chain(dest, src)
                print('cycle chain:', chain)
                assert chain[0] == src and chain[-1] == dest

                # there could be multiple cycles in the forest, map all 
                # nodes connected by a cycle to the same equivalence class.
                # So if [a, b, c] is detected as a cycle and also [a, f]
                # later on then we want [a, b, c, f] as one equivalence class.
                # Make the src of the chain the id for the new equivalence class
                # (any elem of the chain would do, not just src)
                new_class = src
                connected_equ_classes = set()
                for node in chain:
                    if node in node2equivalence_class:
                        old_class = node2equivalence_class[node]
                        connected_equ_classes.add(old_class)
                def merge_equivalence_classes(classes, new_class):
                    merged_mapping = {}
                    for node, clazz in node2equivalence_class.items():
                        merged_mapping[node] = new_class \
                            if clazz in connected_equ_classes \
                            else clazz
                    return merged_mapping
                if len(connected_equ_classes) == 1:
                    new_class = next(iter(connected_equ_classes))
                    print('extending class for', new_class)
                elif len(connected_equ_classes) > 1:
                    print('merging node sets for', connected_equ_classes)
                    connected_equ_classes = \
                        merge_equivalence_classes(connected_equ_classes, new_class)
                for node in chain:
                    node2equivalence_class[node] = new_class

    # extract cycles from equivalence classes
    class2cycle_members = {} # node 2 list
    for node, clazz in node2equivalence_class.items():
        def add_cycle_member(node, clazz):
            if not clazz in class2cycle_members:
                class2cycle_members[clazz] = []
            class2cycle_members[clazz].append(node)
        add_cycle_member(node, clazz)
   
    # dict hashing results in order changing between the found cycles
    # every time you run this func here. So let's bring the result into
    # an ordered normal form (just for easier visual inspection/comparison).
    cycles = list(class2cycle_members.values())
    for cycle in cycles:
        cycle.sort()
    cycles.sort(key = lambda cycle: cycle[0])

    # lastly assert that the cycles are pairwise disjoint
    nodes_in_cycles = set()
    for cycle in cycles:
        for node in cycle:
            assert not node in nodes_in_cycles
            nodes_in_cycles.add(node)

    return cycles
