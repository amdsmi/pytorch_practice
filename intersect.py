from shapely.geometry import Polygon


def check_intersection(polygons):
    """
    :param polygons: list of polygons list in shape [[x1, y1, x2, y2, x3, y3, x4, y4][...],...]
    """
    not_overlay = True
    while polygons:
        chosen_polygon = polygons.pop(0)
        chosen_poly = Polygon([(chosen_polygon[i * 2], chosen_polygon[i * 2 + 1]) for i in range(4)])
        intersect_poly = [poly for poly in polygons if
                          chosen_poly.intersects(Polygon([(poly[j * 2], poly[j * 2 + 1]) for j in range(4)]))]
        if len(intersect_poly) == 0:
            pass
        else:
            not_overlay = False
            break
    return not_overlay