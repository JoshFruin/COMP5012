# pareto_utils.py

def dominates(u, v):
    """
    Checks if solution 'u' dominates solution 'v' in a multi-objective context.
    """
    return (u["Distance"] <= v["Distance"] and u["Time"] <= v["Time"]) and \
        (u["Distance"] < v["Distance"] or u["Time"] < v["Time"])



"""return (u["Distance"] <= v["Distance"] and u["Time"] <= v["Time"]) and \
           (u["Distance"] < v["Distance"] or u["Time"] < v["Time"])"""