import datetime
import networkx as nx


def visual_graph(g, defaults):

    def _get_spec(data, key, default_value=None):
        if key in data:
            return data[key]
        elif key in defaults:
            return defaults[key]
        else:
            return default_value

    graph = nx.DiGraph()
    graph.add_nodes_from(g.nodes())

    for (edge_src, edge_dst, edge_data) in g.edges(data=True):
        weight = _get_spec(edge_data, "weight")
        delay = _get_spec(edge_data, "delay", datetime.timedelta(0))
        if delay == datetime.timedelta(0):
            label = str(weight)
        else:
            label = "{0} ({1})".format(str(weight), int(delay.total_seconds()))
        graph.add_edge(edge_src, edge_dst, label=label)

    return graph


def output_graph(g, defaults, output):
    graph = visual_graph(g, defaults)
    ag = nx.nx_agraph.to_agraph(graph)
    ag.draw(output, prog='circo')
