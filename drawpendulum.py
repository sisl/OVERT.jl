
# get_line is taken from the http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm#Python
def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

# translated from the julia version:
def line(theta, sz):
    # theta is CCW from y axis
    theta += np.pi
 
    # length of line
    d = np.floor((min(sz) - 1) / 2)
    ylim, xlim = sz
    y0, x0 = [int(np.ceil(c/2) - 1) for c in sz]
    # if the matrix is even, it has no center, so the points need a nudge.
    s, c = np.sin(theta), np.cos(theta)
    y0 += (ylim % 2 == 0) & (s > 0)
    x0 += (xlim % 2 == 0) & (c > 0)
 
    x1 = np.clip(int(round(x0 + d*c)), 0, xlim-1)
    y1 = np.clip(int(round(y0 + d*s)), 0, ylim-1)
 
    pts = get_line((x0, y0), (x1, y1))
 
    A = np.zeros(sz, int)
    for p in pts:
        A[p] = 1
 
    return A
