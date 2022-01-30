import cv2 as cv
import numpy as np

from .classify import DigitClassifier

# ------------------------------------------------------------------------------

TARGET_SIZE = 512       ## Size of image to work on
NETWORK_SIZE = 28       ## Size of input to neural network
NOISE_SIZE_TH = 7       ## Components with size less than this are noise

# ------------------------------------------------------------------------------

def extract_puzzle(img: np.ndarray) -> np.ndarray:

    '''
    Given an BGR (or RGB) image, isolate the 9 x 9 Sudoku grid and return its
    filled-in values. Assumes that the puzzle grid is on a flat surface, and
    that the border of the puzzle grid is the largest contour in the image.

    Args:
        img (ndarray<uint8>): HxWx3 BGR (or RGB) color image.

    Returns:
        (ndarray<uint8>): 9x9 array representing Sudoku puzzle. 0 means unfilled
        square.
    '''

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ## Resize to target size while keeping original aspect ratio
    height, width = img.shape
    f = TARGET_SIZE * 1. / max(height, width)
    if f < 1:
        img = cv.resize(img, None, fx=f, fy=f, interpolation=cv.INTER_AREA)

    ## Create binary image
    cv.GaussianBlur(img, (3, 3), 0, img)
    cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 3, img)

    ## Get largest contour. Assume this is the puzzle.
    contour = _get_largest_contour(img)

    ## Mask out everything that is not the puzzle
    mask = np.zeros_like(img)
    cv.drawContours(mask, (contour,), -1, 255, cv.FILLED)
    img[mask == 0] = 0

    ## Get corners of the puzzle
    corners_ix = _get_four_corners(contour)
    corners = contour[corners_ix, 0]

    ## Perspective transform
    a = NETWORK_SIZE
    b = NETWORK_SIZE // 2   ## Buffer of 14 pixels on all sides
    dst = np.array([(b, b), (a*9+b, b), (a*9+b, a*9+b), (b, a*9+b)], dtype=np.float32)
    transform = cv.getPerspectiveTransform(corners.astype(np.float32), dst)
    img = cv.warpPerspective(img, transform, (a*10, a*10))

    ## For each square, determine digit
    puzzle = np.zeros((9, 9), dtype=np.uint8)
    classifier = DigitClassifier()

    centers = _isolate_digits(img)
    for x, y in centers:
        square = img[y-b:y+b, x-b:x+b]
        i = (y - b) // a
        j = (x - b) // a
        if 0 <= i and i < 9 and 0 <= j and j < 9:
            digit = classifier(square)
            puzzle[i, j] = digit

    return puzzle

# ------------------------------------------------------------------------------

def _get_largest_contour(img: np.ndarray) -> np.ndarray:
    '''Return largest contour in binary image, oriented clockwise.'''
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
    signed_areas = list(cv.contourArea(cnt, oriented=True) for cnt in contours)
    ## Find contour with the largest area
    largest_contour_ix = np.argmax(np.abs(signed_areas))
    largest_contour = contours[largest_contour_ix]
    ## If largest contour is oriented CCW, reverse it
    if signed_areas[largest_contour_ix] < 0:         
        largest_contour = largest_contour[::-1]
    return largest_contour

# ------------------------------------------------------------------------------

def _get_four_corners(contour: np.ndarray) -> np.ndarray:

    '''
    Given a square-shaped contour, get the indices corresponding to the four
    corners, in clockwise order starting from the top-left.
    '''

    ## Draw contour
    img = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    cv.drawContours(img, (contour,), -1, 255, 1)

    ## Use Harris corner detector
    harris = cv.cornerHarris(img, 2, 3, 0.04) 
    corner_flat_coords = np.argsort(harris.flat)[-4:] ## Four most probable corners
    corner_coords = np.unravel_index(corner_flat_coords, harris.shape)
    corners = np.stack((corner_coords[1], corner_coords[0]), axis=1)    ## Convert to cv conventions
    
    ## Sort four corners
    x_sorted = np.argsort(corners[:, 0])
    leftmost = corners[x_sorted[:2]]
    tl, bl = leftmost[np.argsort(leftmost[:, 1])]
    rightmost = corners[x_sorted[2:]]
    tr, br = rightmost[np.argsort(rightmost[:, 1])]
    sorted_corners = np.array([tl, tr, br, bl])

    ## Find closest points on the contour
    sorted_corners_ix = np.argmin(np.linalg.norm(contour - sorted_corners, axis=2), axis=0)
    return sorted_corners_ix

# ------------------------------------------------------------------------------

def _isolate_digits(img: np.ndarray) -> np.ndarray:

    '''
    Isolate the digits and remove the puzzle grid from the input image. Returns
    a list of center coordinates for each digit. Some noise in the input image
    is also removed. 

    Args:
        img (ndarray<uint8>): HxW binary image.

    Returns:
        (ndarray<int32>): Nx2 array of center coordiantes.
    '''

    _, labels, stats, _ = cv.connectedComponentsWithStats(img)

    ## Remove noise
    is_noise = (stats[:, cv.CC_STAT_WIDTH] <= NOISE_SIZE_TH) & (stats[:, cv.CC_STAT_HEIGHT] <= NOISE_SIZE_TH)
    is_noise_ix = np.argwhere(is_noise)
    mask = np.isin(labels, is_noise_ix)
    img[mask] = 0

    ## Isolate digits
    is_digit = (stats[:, cv.CC_STAT_WIDTH] < NETWORK_SIZE) * (stats[:, cv.CC_STAT_HEIGHT] < NETWORK_SIZE)
    is_digit_ix = np.argwhere(is_digit & ~is_noise)
    mask = np.isin(labels, is_digit_ix)
    img[~mask] = 0

    ## Get centers for each digit
    centers = []
    for ix in is_digit_ix[:, 0]:
        x = stats[ix, cv.CC_STAT_LEFT] + stats[ix, cv.CC_STAT_WIDTH] // 2
        y = stats[ix, cv.CC_STAT_TOP] + stats[ix, cv.CC_STAT_HEIGHT] // 2
        centers.append((x, y))

    return centers

# ------------------------------------------------------------------------------

# def _get_edge_vectors(
#     contour: np.ndarray, 
#     start_index: int,
#     end_index: int
# ) -> np.ndarray:
    
#     '''
#     Approximate the oriented edge of a square contour with 9 roughly
#     equal-length vectors.

#     `contour` is an N * 1 * 2 array of int32 coordinates.
#     `start_index` and `end_index` are non-negative integers less than N.
#     Returns an 9 * 2 array of floats.
#     '''
    
#     ## Note that the contour is clockwise-oriented
#     if start_index < end_index:
#         edge = contour[start_index:end_index+1, 0]
#     else:
#         edge = np.concatenate((contour[start_index:, 0], contour[:end_index+1, 0]))
        
#     ## Calculate distance between adjacent points
#     adj_distances = np.linalg.norm(edge[1:] - edge[:-1], axis=1)
#     total_length = np.sum(adj_distances)
    
#     vecs = []
#     cumulative_dist = 0
#     target_dist = total_length / 9
#     startpoint = edge[0]
#     for i, dist in enumerate(adj_distances):
#         while len(vecs) < 8 and cumulative_dist + dist > target_dist:
            
#             ## Find intermediate point on the contour
#             u = (target_dist - cumulative_dist) / dist ## Scalar in [0, 1)
#             endpoint = edge[i] + u * (edge[i+1] - edge[i])

#             vecs.append(endpoint - startpoint)
#             startpoint = endpoint
#             target_dist += total_length / 9
            
#         cumulative_dist += dist

#     ## Last vector points to the corner
#     vecs.append(edge[-1] - startpoint)
#     return np.array(vecs)

# ------------------------------------------------------------------------------

# def find_grid_intersections(contour) -> np.ndarray:

#     '''
#     Given the contour around the 9 x 9 Sudoku grid, find the coordinate of each
#     grid intersection, given left-to-right and top-to-bottom.

#     Input is an N * 1 * 2 array of int32 coordinates.
#     Returns a 10 * 10 * 2 array of int32 coordinates.
#     '''
    
#     corners_i = sudoku.detector.extract._get_four_corners(contour)
#     _get_edge_vectors = sudoku.detector.extract._get_edge_vectors
    
#     ## Approximate each edge with 9 roughly equal-length vectors
#     t_vecs = _get_edge_vectors(contour, corners_i[0], corners_i[1])
#     r_vecs = _get_edge_vectors(contour, corners_i[1], corners_i[2])
#     b_vecs = _get_edge_vectors(contour, corners_i[2], corners_i[3])
#     l_vecs = _get_edge_vectors(contour, corners_i[3], corners_i[0])
    
#     ## Reverse the bottom and left vectors
#     b_vecs = -b_vecs[::-1]
#     l_vecs = -l_vecs[::-1]
    
#     intersections = np.zeros((10, 10, 2), dtype=np.float32)
    
#     ## fill boundary
#     intersections[0, 0] = contour[corners_i[0]]
#     intersections[0, 9] = contour[corners_i[1]]
#     intersections[9, 9] = contour[corners_i[2]]
#     intersections[9, 0] = contour[corners_i[3]]
#     for i in range(8):
#         intersections[0, i+1] = intersections[0, i] + t_vecs[i]
#     for i in range(8):
#         intersections[i+1, 9] = intersections[i, 9] + r_vecs[i]
#     for i in range(8):
#         intersections[i+1, 0] = intersections[i, 0] + l_vecs[i]
#     for i in range(8):
#         intersections[9, i+1] = intersections[9, i] + b_vecs[i]
    
#     # fill top triangle
#     for i in range(4):
#         for j in range(i+1, 9-i):
#             intersections[i+1, j] = intersections[i, j] + ((9-j)*l_vecs[i] + j*r_vecs[i])/9
#             intersections[8-i, j] = intersections[9-i, j] - ((9-j)*l_vecs[8-i] + j*r_vecs[8-i])/9
            
#     return intersections.astype(np.int32)
