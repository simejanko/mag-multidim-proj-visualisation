def rectangle_intersect(bbox1, bbox2):
    """
    Checks if two rectangles intersect. Bboxes given in (left_x, bottom_y, width, height).
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    return not (x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + h2)


def rectangle_circle_bbox_intersect(bbox, circle):
    """
    Checks if rectangle and circle's bounding box intersect.

    :param bbox: rectangle's bbox given in (left_x, bottom_y, width, height)
    :param circle: circle given in (center_x, center_y, radius)
    """

    x1, y1, w1, h1 = bbox
    cx, cy, r = circle

    x2, y2 = (cx - r, cy - r)
    w2, h2 = (2 * r, 2 * r)

    return rectangle_intersect((x1, y1, w1, h1), (x2, y2, w2, h2))
