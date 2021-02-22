import datetime

DEFAULT_DEFAULT_TYPE = "variable"


def default_setup(g, defaults):
    if "default_type" not in defaults:
        defaults["default_type"] = DEFAULT_DEFAULT_TYPE

    s_type = set([])
    for _, node_data in g.nodes(data=True):
        if "type" in node_data:
            s_type.add(node_data["type"])
        else:
            s_type.add(defaults["default_type"])

    if "variable_index" not in defaults:
        if "tsevent" in s_type:
            dt_base = datetime.datetime(2112, 9, 3)
            dt_interval = datetime.timedelta(minutes=1)
            index = [dt_base + dt_interval * diff for diff in range(24 * 60)]
        else:
            index = list(range(1000))
        defaults["variable_index"] = index

    if "variable" in s_type:
        defaults.update({"gaussian_scale": 1,
                         "gaussian_loc": 0})
    if "binary" in s_type:
        defaults.update({"prob": 0.5})
    if "countable" in s_type or "tsevent" in s_type:
        defaults.update({"lambd": 0.1})

    return defaults
