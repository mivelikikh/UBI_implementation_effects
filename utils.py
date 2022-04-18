import os
import pandas as pd

from matplotlib import cm


COLORS_DICT = {'Alaska': cm.winter(0.5), #  #0080bf
               'Brazil': cm.winter(1.0), #  #00ff80
               'Russia': cm.cool(0.8), #  #cc33ff
               'None': cm.autumn(0.6)} # #ff9900


def get_colors(colors_dict, color_names=None):
    if color_names is None:
        return list(colors_dict.values())
    return [colors_dict[color_name] for color_name in color_names]


def lists_union(lists):
    united_list = []
    if isinstance(lists[0], list):
        united_list = list(set().union(*lists))
    else:
        united_list = list(set().union(lists))
    
    if len(united_list) == 1:
        return united_list[0]
    return united_list


def lists_intersection(lists):
    intersection_list = []
    if isinstance(lists[0], list):
        intersection_list = [element for element in lists[0]
                             if element in set(lists[0]).intersection(*map(set, lists))]
    else:
        intersection_list = list(lists)
    
    if len(intersection_list) == 1:
        return intersection_list[0]
    return intersection_list
