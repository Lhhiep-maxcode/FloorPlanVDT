from skimage import io
from skimage import morphology,feature,transform,measure
from pathlib import Path
from scipy import stats
from scipy import ndimage
from shapely import geometry
from g2p.label import DOOR, EXTERIOR_BOUNDARY, OTHERS
import numpy as np

def extract_boundary(boundary_image):
    '''
    boundary_image: numpy array
    '''
    # Make sure we know the front door
    front_door_mask = boundary_image == DOOR
    region = measure.regionprops(front_door_mask.astype(int))[0]
    front_door = np.array(region.bbox,dtype=int)

    exterior_boundary = []

    # Get bounding box of the whole building from boundary mask
    min_h,max_h = np.where(np.any(boundary_image,axis=1))[0][[0,-1]]
    min_w,max_w = np.where(np.any(boundary_image,axis=0))[0][[0,-1]]

    # Expand the bounding box slightly (+10) to avoid cutting edges
    h, w = boundary_image.shape
    min_h = max(min_h-10,0)
    min_w = max(min_w-10,0)
    max_h = min(max_h+10,h)
    max_w = min(max_w+10,w)

    # ---- Boundary tracing algorithm ----
    # search direction: 0(right), 1(down), 2(left), 3(up)
    # Find the first interior pixel (top-left corner inside the house)
    flag = False
    for h in range(min_h, max_h):
        for w in range(min_w, max_w):
            if boundary_image[h, w] == EXTERIOR_BOUNDARY:
                exterior_boundary.append((h, w, 0))
                flag = True
                break
        if flag:
            break
    
    # ---- Boundary walking loop ----
    # Rules: 
    #   - left/top edge = "inside"
    #   - right/bottom edge = "outside"
    #   - corner_sum trick (sum == odd number) decides to turn direction
    while(flag):
        if exterior_boundary[-1][2] == 0:
            for w in range(exterior_boundary[-1][1]+1, max_w):
                corner_sum = 0
                if boundary_image[exterior_boundary[-1][0], w] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if boundary_image[exterior_boundary[-1][0]-1, w] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if boundary_image[exterior_boundary[-1][0], w-1] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if boundary_image[exterior_boundary[-1][0]-1, w-1] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if corner_sum == 1:
                    new_point = (exterior_boundary[-1][0], w, 1)
                    break
                if corner_sum == 3:
                    new_point = (exterior_boundary[-1][0], w, 3)
                    break
        
        if exterior_boundary[-1][2] == 1:      
            for h in range(exterior_boundary[-1][0]+1, max_h): 
                corner_sum = 0                
                if boundary_image[h, exterior_boundary[-1][1]] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if boundary_image[h-1, exterior_boundary[-1][1]] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if boundary_image[h, exterior_boundary[-1][1]-1] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if boundary_image[h-1, exterior_boundary[-1][1]-1] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if corner_sum == 1:
                    new_point = (h, exterior_boundary[-1][1], 2)
                    break
                if corner_sum == 3:
                    new_point = (h, exterior_boundary[-1][1], 0)
                    break

        if exterior_boundary[-1][2] == 2:   
            for w in range(exterior_boundary[-1][1]-1, min_w, -1):
                corner_sum = 0                     
                if boundary_image[exterior_boundary[-1][0], w] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if boundary_image[exterior_boundary[-1][0]-1, w] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if boundary_image[exterior_boundary[-1][0], w-1] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if boundary_image[exterior_boundary[-1][0]-1, w-1] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if corner_sum == 1:
                    new_point = (exterior_boundary[-1][0], w, 3)
                    break
                if corner_sum == 3:
                    new_point = (exterior_boundary[-1][0], w, 1)
                    break

        if exterior_boundary[-1][2] == 3:       
            for h in range(exterior_boundary[-1][0]-1, min_h, -1):
                corner_sum = 0                
                if boundary_image[h, exterior_boundary[-1][1]] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if boundary_image[h-1, exterior_boundary[-1][1]] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if boundary_image[h, exterior_boundary[-1][1]-1] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if boundary_image[h-1, exterior_boundary[-1][1]-1] == EXTERIOR_BOUNDARY:
                    corner_sum += 1
                if corner_sum == 1:
                    new_point = (h, exterior_boundary[-1][1], 0)
                    break
                if corner_sum == 3:
                    new_point = (h, exterior_boundary[-1][1], 2)
                    break

        # Stop when loop closes back at start point
        if new_point != exterior_boundary[0]:
            exterior_boundary.append(new_point)
        else:
            flag = False
    exterior_boundary = [[row,col,direction,0] for row,col,direction in exterior_boundary]
    
    # ---- Insert the front door into boundary ----
    door_y1,door_x1,door_y2,door_x2 = front_door
    door_h,door_w = door_y2-door_y1,door_x2-door_x1
    is_vertical = door_h>door_w or door_h==1 # 

    insert_index = None
    door_index = None
    new_p = []
    th = 3

    # Iterate over boundary edges to see where the door intersects
    for i in range(len(exterior_boundary)):
        y1,x1,d,_ = exterior_boundary[i]
        y2,x2,_,_ = exterior_boundary[(i+1)%len(exterior_boundary)] 
        if is_vertical!=d%2: continue       # check down/up for vertical or left/right for non-vertical first
        if is_vertical and (x1-th<door_x1<x1+th or x1-th<door_x2<x1+th): # 1:down 3:up
            l1 = geometry.LineString([[y1,x1],[y2,x2]])    
            l2 = geometry.LineString([[door_y1,x1],[door_y2,x1]])  
            l12 = l1.intersection(l2)
            if l12.length>0:
                dy1,dy2 = l12.xy[0] # (y1>y2)==(dy1>dy2)
                insert_index = i
                door_index = i+(y1!=dy1)
                if y1!=dy1: new_p.append([dy1,x1,d,1])
                if y2!=dy2: new_p.append([dy2,x1,d,1])
        elif not is_vertical and (y1-th<door_y1<y1+th or y1-th<door_y2<y1+th):
            l1 = geometry.LineString([[y1,x1],[y2,x2]])    
            l2 = geometry.LineString([[y1,door_x1],[y1,door_x2]])  
            l12 = l1.intersection(l2)
            if l12.length>0:
                dx1,dx2 = l12.xy[1] # (x1>x2)==(dx1>dx2)
                insert_index = i
                door_index = i+(x1!=dx1)
                if x1!=dx1: new_p.append([y1,dx1,d,1])
                if x2!=dx2: new_p.append([y1,dx2,d,1])                

    if len(new_p)>0:
        exterior_boundary = exterior_boundary[:insert_index+1]+new_p+exterior_boundary[insert_index+1:]
    exterior_boundary = exterior_boundary[door_index:]+exterior_boundary[:door_index]

    exterior_boundary = np.array(exterior_boundary,dtype=int)
    return exterior_boundary