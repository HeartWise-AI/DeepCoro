"""Coronary artery labels and deterministic segment selection."""

import numpy as np

retinanet_artery_labels = { # https://syntaxscore.org/index.php/tutorial/definitions
    1: 'prox_rca', # 1: RCA proximal
    2: 'mid_rca', # 2: RCA mid
    3: 'dist_rca', # 3: RCA distal
    4: 'pda', # 4: Posterior descending
    5: 'leftmain', # 5: Left main
    6: 'lad', # 6: LAD proximal
    7: 'mid_lad', # 7: LAD mid
    8: 'dist_lad', # 8: LAD apical
    9: 'other', # 9: First diagonal
    10: 'other', # 9a: First diagonal a
    11: 'other', # 10: Second diagonal
    12: 'other', # 10a: Second diagonal a
    13: 'lcx', # 11: Proximal circumflex
    14: 'other', # 12: Intermediate/anterolateral
    15: 'other', # 12a: Obtuse marginal a
    16: 'dist_lcx', # 13: Distal circumflex
    17: 'other', # 14: Left posterolateral
    18: 'other', # 14a: Left posterolateral a
    19: 'other', # 15: Posterior descending
    20: 'posterolateral', # 16: Posterolateral from RCA
    21: 'posterolateral', # 16a: Posterolateral from RCA
    22: 'posterolateral', # 16b: Posterolateral from RCA
    23: 'posterolateral', # 16c: Posterolateral from RCA
    24: 'other', # 12b: Obtuse marginal b
    25: 'other', # 14b: Left posterolateral b
    26: 'other' # stenosis
}

which_artery = {"RCA": [1,2,3,4,20,21,22,23],
                "LCA": [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,24,25]}

def get_segment_region(region, object_value):
    if object_value not in which_artery:
        raise ValueError("Artery view must be RCA or LCA")
    unique, counts = np.unique(region, return_counts=True)
    # np.unique sorts labels; stable sorting preserves the original tie behavior.
    for index in np.argsort(-counts, kind="stable"):
        label = unique[index]
        if label in which_artery[object_value]:
            return retinanet_artery_labels[label]
    return "None"


def get_segment_center(region, object_value):
    """Search outward from the image center, including tiny/rectangular crops."""
    if object_value not in which_artery:
        raise ValueError("Artery view must be RCA or LCA")
    if np.ndim(region) != 2:
        raise ValueError("Segment region must be a 2D array")
    height, width = region.shape
    if height == 0 or width == 0:
        return "None"
    cy, cx = height // 2, width // 2
    for radius in range(max(height, width) + 1):
        if radius == 0:
            crop = region[cy:cy + 1, cx:cx + 1]
        else:
            crop = region[max(0, cy - radius):min(height, cy + radius),
                          max(0, cx - radius):min(width, cx + radius)]
        segment = get_segment_region(crop, object_value)
        if segment != "None":
            return segment
    return "None"
