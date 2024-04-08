
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from GeometryUtility import *
import random


class Camera:
   def __init__(self,K: np.array,Rot: np.array,tvec:np.array,rvec: np.array,pos: np.array,dist_coeffs: np.array):
       self.K = K
       self.K_inv = np.linalg.inv(K)
       self.Rot = Rot
       self.R_inv = np.linalg.inv(Rot)
       self.pos = pos
       self.tvec = tvec
       self.rvec = rvec
       self.dist_coeffs = dist_coeffs
      

def GetRandomColor():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    return (b,g,r)
# draw the provided lines on the image
def drawLines(img, lines,colors):
    _, c, _ = img.shape
    cnt = 0
    for r in lines:
        color = colors[cnt]
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img, (x0, y0), (x1, y1), color, 1)
        cnt += 1

def DrawRect(img,p0,p1,color=(255, 0, 0),thickness=2):
    start_point = (5, 5)
    end_point = (220, 220)
    #p0 has to be the left most point in the image
    cv2.rectangle(img, p0, p1, color, thickness)
  
def ProcessPoints(uv_cam1: np.array,uv_cam2:np.array,cam1: Camera , cam2: Camera):
    lines1  = []
    lines2  = []
    pixels1 = []
    pixels2 = []
    inter_points = []
    for i in range(uv_cam1.shape[0]):
        dir_1 = GeometryUtilities.GetPointCameraRayFromPixel(uv_cam1[i,:],cam1.K_inv,cam1.R_inv,cam1.tvec)
        dir_2 = GeometryUtilities.GetPointCameraRayFromPixel(uv_cam2[i,:],cam2.K_inv,cam2.R_inv,cam2.tvec)
        intersection = GeometryUtilities.RayRayIntersection(cam1.pos,dir_1,cam2.pos,dir_2)
        intersection = -1 * intersection
        pixel_cam1 = ProjectPointsCV(intersection,cam1.K,cam1.rvec,cam1.tvec,cam1.dist_coeffs)
        pixel_cam2 = ProjectPointsCV(intersection,cam2.K,cam2.rvec,cam2.tvec,cam2.dist_coeffs)
        
        p = pixel_cam1[0,0,:]
        pixels1.append(p)
        #PlotPointOnImage(im1,p[0],p[1],(0,0,255))
        p = pixel_cam2[0,0,:]
        pixels2.append(p)
        #PlotPointOnImage(im7,p[0],p[1],color=(0,0,255))
        error_pixel1 = np.abs(pixel_cam1 - uv_cam1[i,:])
        error_pixel2 = np.abs(pixel_cam2 - uv_cam2[i,:])
        print(f"Pixel cam 1 {uv_cam1[i,:]}, Pixel cam 7 {uv_cam2[i,:]}, World Position recovered {intersection}")
        print(f" Error Reprojection pixel cam 1 {error_pixel1} Error Reprojection pixel cam 2 {error_pixel2}\n")
        inter_points.append(intersection)

        lines1.append(-1*cam1.pos)
        lines2.append(-1*cam2.pos)
        #lines1.append(-1*intersection)
        #lines7.append(-1*intersection)
        lines1.append(-1*cam1.pos + 200 * dir_1)
        lines2.append(-1*cam2.pos + 200 * dir_2)
    
    return np.array(pixels1),np.array(pixels2),np.array(lines1),np.array(lines2),np.array(inter_points)


def ProcessPointsMine(uv_cam1: np.array,uv_cam2:np.array,cam1: Camera , cam2: Camera):
    lines1  = []
    lines2  = []
    pixels1 = []
    pixels2 = []
    inter_points = []
    for i in range(uv_cam1.shape[0]):
        dir_1 = GeometryUtilities.GetPointCameraRayFromPixel(uv_cam1[i,:],cam1.K_inv,cam1.R_inv,cam1.tvec)
        dir_2 = GeometryUtilities.GetPointCameraRayFromPixel(uv_cam2[i,:],cam2.K_inv,cam2.R_inv,cam2.tvec)
        intersection = GeometryUtilities.RayRayIntersection(cam1.pos,dir_1,cam2.pos,dir_2)
        intersection = -1 * intersection
        pixel_cam1 = ProjectPointsCV(intersection,cam1.K,cam1.rvec,cam1.tvec,cam1.dist_coeffs)
        pixel_cam2 = ProjectPointsCV(intersection,cam2.K,cam2.rvec,cam2.tvec,cam2.dist_coeffs)
        
        p = pixel_cam1[0,0,:]
        pixels1.append(p)
        #PlotPointOnImage(im1,p[0],p[1],(0,0,255))
        p = pixel_cam2[0,0,:]
        pixels2.append(p)
        #PlotPointOnImage(im7,p[0],p[1],color=(0,0,255))
        error_pixel1 = np.abs(pixel_cam1 - uv_cam1[i,:])
        error_pixel2 = np.abs(pixel_cam2 - uv_cam2[i,:])
        print(f"Pixel cam 1 {uv_cam1[i,:]}, Pixel cam 7 {uv_cam2[i,:]}, World Position recovered {intersection}")
        print(f" Error Reprojection pixel cam 1 {error_pixel1} Error Reprojection pixel cam 2 {error_pixel2}\n")
        inter_points.append(intersection)

        lines1.append(-1*cam1.pos)
        lines2.append(-1*cam2.pos)
        #lines1.append(-1*intersection)
        #lines7.append(-1*intersection)
        lines1.append(-1*cam1.pos + 200 * dir_1)
        lines2.append(-1*cam2.pos + 200 * dir_2)
    
    return np.array(pixels1),np.array(pixels2),np.array(lines1),np.array(lines2),np.array(inter_points)
def PlotPointsAndLines(xyz_points, intersections,lines1,lines2):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.scatter3D(xline1, yline1, zline1, c=zline1, cmap='Blues');
    ax.plot3D(lines1[:,0], lines1[:,1], lines1[:,2], c='blue');
    ax.plot3D(lines2[:,0], lines2[:,1], lines2[:,2], c='red');
    
    #ax.scatter(x_cam,y_cam,z_cam, color="r")
    ax.scatter(intersections[:,0],intersections[:,1],intersections[:,2], color="y")
    ax.scatter(xyz_points[:,0],xyz_points[:,1],xyz_points[:,2], color="black")  
    plt.show()

def ReadJsonAnnotation(json_file: str) -> dict:
    jsonObj = None
    with open(json_file) as f:
       lines = f.readlines()
       json_str  = ''.join(lines)
       jsonObj = json.loads(json_str)
    return jsonObj

def GetKeypointsAndBbox(json_obj):
    num_keypoints = json_obj['annotations'][0]['num_keypoints']
    keypoints = {}
    bboxes = {}
    for  a in json_obj['annotations']:
        keypoints[a['id']] = np.zeros((num_keypoints,2),dtype = 'double')
        bboxes[a['id']] = np.zeros((4,2),dtype = 'int')
        ctr = 0
        for i in range(0, len(a['keypoints']),3):
            keypoints[a['id']][ctr][0] = a['keypoints'][i]
            keypoints[a['id']][ctr][1] = a['keypoints'][i+1]
            ctr += 1

        # clockwise order
        x = a['bbox'][0]
        y = a['bbox'][1]
        w = a['bbox'][2]
        h = a['bbox'][3]
        bboxes[a['id']][0,0] = int(x)
        bboxes[a['id']][0,1] = int(y)

        bboxes[a['id']][1,0] = int(x) + w
        bboxes[a['id']][1,1] = int(y)
        
        bboxes[a['id']][2,0] = int(x) 
        bboxes[a['id']][2,1] = int(y) + h
           
        bboxes[a['id']][3,0] = int(x) + w
        bboxes[a['id']][3,1] = int(y) + h

    return keypoints,bboxes


#utility function
def PlotPointOnImage(img,x,y,color=(0,255,0)):
    cv2.circle(img, (int(x), int(y)), 3, color, -1)

def PlotPointsOnImages(im1,im2,pixels1,pixels2,color=(0,0,255),bUseMatlplotlib = False,figsize = (10,10)):
    for i in range(pixels1.shape[0]):
        PlotPointOnImage(im1,pixels1[i][0],pixels1[i][1],color)
        PlotPointOnImage(im2,pixels2[i][0],pixels2[i][1],color)
    if(bUseMatlplotlib):
        cv2.imshow("Output_cam1", im1)
        cv2.imshow("Output_cam7", im2)
        cv2.waitKey(0)
    else:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        plt.figure(figsize = figsize)
        plt.imshow(im1)
        plt.show()

        plt.figure(figsize = figsize)
        plt.imshow(im2)
        plt.show()

def ProjectPointsCV(xyz,K,rvec,tvec,dist_coeffs):
  p_uvs, _= cv2.projectPoints(xyz,rvec,tvec,K,dist_coeffs)
  return p_uvs


def ProjectPoints(xyz,K,Rot,tvec,dist_coeffs):
  uvs =[]
  for p in xyz:
    p_camera_space = Rot.dot( p)+ tvec[:,0]
    p_uv = K @ p_camera_space
    uvs.append(p_uv/p_uv[2])
  return np.array(uvs,dtype='double')

def slope(p1, p2) :
   return (p2[1] - p1[1]) * 1. / (p2[0] - p1[0])
   
def y_intercept(slope, p1) :
   return p1[1] - 1. * slope * p1[0]
   
def intersect(line1, line2) :
   min_allowed = 1e-5   # guard against overflow
   big_value = 1e10     # use instead (if overflow would have occurred)
   m1 = slope(line1[0], line1[1])
   print( 'm1: %d' % m1 )
   b1 = y_intercept(m1, line1[0])
   print( 'b1: %d' % b1 )
   m2 = slope(line2[0], line2[1])
   print( 'm2: %d' % m2 )
   b2 = y_intercept(m2, line2[0])
   print( 'b2: %d' % b2 )
   if abs(m1 - m2) < min_allowed :
      x = big_value
   else :
      x = (b2 - b1) / (m1 - m2)
   y = m1 * x + b1
   y2 = m2 * x + b2
   print( '(x,y,y2) = %d,%d,%d' % (x, y, y2))
   return (int(x),int(y))

def FindEpilineExtreme(line: np.array,size) -> np.array:
    #find all possible intersection with edges of the image x=0,y=0, x=w,y=h 
    edge_points = np.zeros((4,2))
    # x = 0
    edge_points[0,0] = 0
    edge_points[0,1] = -line[2]/line[1]

    # y = 0 
    edge_points[1,0] = -line[2]/line[0]
    edge_points[1,1] = 0

    # x = w
    edge_points[2,0] = size[1]
    edge_points[2,1] = - (size[1]*line[0]+line[2])/line[1] # y = -(ax+c)/b

    # y = 0 
    edge_points[3,0] = -(size[0]*line[1]+line[2])/line[0]
    edge_points[3,1] = size[0]

    return edge_points



# converts from ax + by +c to AB points 
def GetLine(line: np.array,size) -> np.array: 
   points = FindEpilineExtreme(line,size)
   out_points = []
   for i in range(points.shape[0]):
       if(points[i,0] < 0 or points[i,0] >  size[1] or points[i,1] < 0 or points[i,1] >  size[0]):
           continue
       out_points.append(points[i,:])
   
   return np.array(out_points)



def FindMatch(keypointsCam1: dict, keypointsCam2: dict,id_cam1,cam1: Camera,cam2: Camera,bUseVieww = False,verbose = False):
    keypoints1 = keypointsCam1[id_cam1] # goalkeeper
    best_id = -1
    best_3dpoints = []
    min_mse = 10e10
    count = 0
    for key in keypointsCam2:
        keypoints2 = keypointsCam2[key]
        points3d = [] 
        mse = 0
        bSkip = False
        for i in range(len(keypoints1)):         
            dir_1 = GeometryUtilities.GetPointCameraRayFromPixel(keypoints1[i,:],cam1.K_inv,cam1.R_inv,cam1.tvec)
            dir_2 = GeometryUtilities.GetPointCameraRayFromPixel(keypoints2[i,:],cam2.K_inv,cam2.R_inv,cam2.tvec)
            if bUseVieww == True:
                l1p1 = cam1.pos
                l1p2 = cam1.pos + 200 * dir_1

                l2p1 = cam2.pos
                l2p2 = cam2.pos + 200 * dir_2
                dot,intersection = GeometryUtilities.DistanceBetweenTwoLinesPow2(l1p1,l1p2,l2p1,l2p2)
            else:
                intersection  = GeometryUtilities.RayRayIntersection(cam1.pos,dir_1,cam2.pos,dir_2)
            if(intersection is None):
                print("BAD POINT No Intersection")
                count += 1
                bSkip = True
                break
            intersection = -1 * intersection
            points3d.append(intersection)

            pixel_cam1 = ProjectPointsCV(intersection,cam1.K,cam1.rvec,cam1.tvec,cam1.dist_coeffs)
            pixel_cam2 = ProjectPointsCV(intersection,cam2.K,cam2.rvec,cam2.tvec,cam2.dist_coeffs)
            error = np.abs(pixel_cam2 - keypoints2[i,:])
            error_sqr = np.linalg.norm((pixel_cam2 - keypoints2[i,:]))
            mse += error_sqr
            if(verbose):
                print(f"id: {i} rect id cam 2: {key}, {intersection} {keypoints2[i,:]} reproj: {pixel_cam2} err: {error}, err_sqrt {error_sqr}")
            if(intersection[2] < 0 or intersection[0] < -60 or intersection[0] > 60 or  intersection[1] < -10 or intersection[1] > 60 ):
                if(verbose):
                    print("BAD POINT")
                count += 1
                bSkip = True
                break

        if( min_mse > mse and bSkip == False):
            best_id = key
            min_mse = mse
            best_3dpoints = points3d
        if(verbose):
            print(f"Bad Points count {count}")
            print("-----------------------------------------------------------------------------------")

    return best_id, best_3dpoints,min_mse

# A ------- B
# |         |
# |         |
# |         |
# C ------- D
    
def GetLinesFromAABB(aabb: np.array) -> np.array:
   bb_lines =  np.zeros((4,2,2))
   # AB
   bb_lines[0,0,:] = aabb[0,:]
   bb_lines[0,1,:] = aabb[1,:]

   # AC
   bb_lines[1,0,:] = aabb[0,:]
   bb_lines[1,1,:] = aabb[2,:]

   # BD
   bb_lines[2,0,:] = aabb[1,:]
   bb_lines[2,1,:] = aabb[3,:]

   # CD
   bb_lines[3,0,:] = aabb[2,:]
   bb_lines[3,1,:] = aabb[3,:]

   return bb_lines

def line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not(0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
 
    return x, y

def line_intersect_np(line1,line2):
    """ returns a (x, y) tuple or None if there is no intersection """

    Ax1 = line1[0,0]  
    Ay1 = line1[0,1]
    Ax2 = line1[1,0]
    Ay2 = line1[1,1]

    Bx1 = line2[0,0]  
    By1 = line2[0,1]
    Bx2 = line2[1,0]
    By2 = line2[1,1]
 
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not(0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
    
    return x, y

def FindMatchEpipoles(epilines: np.array,boundingBoxes:np.array,verbose=False):
    results = {}  # id bbox -> number of hits
    for key in boundingBoxes:
        if(verbose == True):
            print(f'-------------------------------id {key}--------------------------------------')
        bbox_lines = GetLinesFromAABB(boundingBoxes[key])
        results[key] = 0
        for i in range(epilines.shape[0]):
            epiline = epilines[i,:,:]
            points = []
            color = GetRandomColor()
            cnt = 0
            for j in range(bbox_lines.shape[0]):
                bb_line = bbox_lines[j,:,:]
                int_point  = line_intersect_np(epiline,bb_line)
                if int_point != None:
                    points.append(np.array(int_point))
                    # check point inside bounding box
                   
                    cnt += 1
       
                if(verbose == True):
                   print(f"{int_point} suca")
            if(cnt > 0):
                results[key] += 1
        
        if(verbose == True):
            print('------------------------------------------------------------------------')
    if(verbose == True):
       print('------------------------------------------------------------------------')
    results_sorted =dict(sorted(results.items(),key= lambda x:x[1],reverse=True))
    if(verbose == True):
       print(results_sorted)

    return results_sorted

def ComputeAndPlotEiplines(imgL,imgR,F,uv1,uv2):
    colors = [ GetRandomColor() for i in range(uv1.shape[0])]
    
    # find epilines corresponding to points in left image and draw them on the right image
    epilinesL = cv2.computeCorrespondEpilines(uv1.reshape(-1, 1, 2), 1, F)
    epilinesL = epilinesL.reshape(-1, 3)
    
    epilinesR = cv2.computeCorrespondEpilines(uv2.reshape(-1, 1, 2), 2, F)
    epilinesR = epilinesR.reshape(-1, 3)
    

    drawLines(imgL, epilinesR, colors)
    drawLines(imgR, epilinesL, colors)

    return epilinesL,epilinesR

def DrawBbox(img,bbox,color=None):
    if(color == None):
        color = GetRandomColor()
    DrawRect(img,bbox[0,:],bbox[3,:],color)

# takes an numpy array of lines ax+by+c = 0 
# returns 2 points array on that line usually the intersection with the image borders 
def TransformEpilines(epilines: np.array,img):
    lines = []
    for line in epilines:
        p = GetLine(line,img.shape)
        lines.append(p)
    lines = np.array(lines)
    return lines

def EpilineBBoxIntersections(epilines,bbox,img,verbose=False):
    bb_lines = GetLinesFromAABB(bbox)
    out_points = []
    for i in range(epilines.shape[0]):
        epiline = epilines[i,:,:]
        points = []
        for j in range(bb_lines.shape[0]):
            bb_line = bb_lines[j,:,:]
            int_point  = line_intersect_np(epiline,bb_line)
            if int_point != None:
                points.append(np.array(int_point))
            
            if(verbose == True):
                print(f"{int_point}")
      
        if(len(points) > 0):
            out_points.append(np.array(points))
        else:
            out_points.append(None)

    return out_points # intersections

def ValidKeypoints(keypoints):
    for i in range(keypoints.shape[0]):
        if(keypoints[i,0] == 0  and keypoints[i,1] == 0  ):
            return False

    return True
                

def FindMatchesDualCamera(cameras,cln_imgs,keypoints,id_1,id_2,bboxes,badkeys,verbose=False,bUSeVieww = False):
    colors = {}
    cam1 = cameras[id_1]
    cam2 = cameras[id_2]
    keys1 = keypoints[id_1]
    keys2 = keypoints[id_2]
    img1 = cln_imgs[id_1].copy()
    img2 = cln_imgs[id_2].copy()
    bboxes1 = bboxes[id_1]
    bboxes2 = bboxes[id_2]
    badkeys1 = badkeys[id_1]
    badkeys2 = badkeys[id_2]
    results = {}
    
    print(f'--------------Starting Cam: {id_1} -- Cam: {id_2}-------------------------------------------')
    res_key = (id_1,id_2)
    results[res_key] = {}
    results[res_key]['img'] = None
    results[res_key]['pairs'] = []
    results[res_key]['mse'] = []
    results[res_key]['keys_3d'] = []
    # iterate through 
    for key in keys2:
        id,key3d,mse = FindMatch(keys2,keys1,key,cam2,cam1,bUSeVieww,verbose)
        results[res_key]['pairs'].append((key,id))
        results[res_key]['mse'].append(mse)
        results[res_key]['keys_3d'].append(key3d)
        if(mse > 0 and mse < 100):
            color =  GetRandomColor()
            colors[(key,id)] = color
            colors[(id,key)] = color
            for p in keys1[id]:
                PlotPointOnImage(img1,p[0],p[1],color)
        print(f'{key} {id} {mse}')

    print('-----------------------------------------------------------------')
    for key in keys1:
        id,key3d,mse = FindMatch(keys1,keys2,key,cam1,cam2,bUSeVieww,verbose)
        results[res_key]['pairs'].append((key,id))
        results[res_key]['mse'].append(mse)
        results[res_key]['keys_3d'].append(key3d)
        if(mse > 0 and mse < 100):
            color = None
            if (key,id) in colors:
                color = colors[(key,id)]
            elif (id,key) in colors:
                color = colors[(id,key)]
            else:
                color = GetRandomColor()
                colors[(key,id)] = color
                colors[(id,key)] = color
            for p in keys2[id]:
                PlotPointOnImage(img2,p[0],p[1],color)


        print(f'{key} {id} {mse}')
    print(f'--------------Ending Cam: {id_1} -- Cam: {id_2}-------------------------------------------')
    for key in bboxes1:
        if key in badkeys1:
            continue
        r = bboxes1[key]
        DrawBbox(img1,r)
        cv2.putText(img1, text='box_id: '+str(key), org=r[0,:],
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0),
                thickness=1, lineType=cv2.LINE_AA)
    for key in bboxes2:
        if key in badkeys2:
            continue
        r = bboxes2[key]
        DrawBbox(img2,r)
        cv2.putText(img2, text='box_id: '+str(key), org=r[0,:],
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0),
                thickness=1, lineType=cv2.LINE_AA)

    h_img = cv2.hconcat([img1, img2])
    results[res_key]['img'] = h_img
    #cv2.imshow(f"Output_cam{id_1}", img1)
   # cv2.imshow(f"Output_cam{id_2}", img2)
    #cv2.imshow("Output_Conb", h_img)
    #plt.figure(figsize = (80,60))
    #plt.imshow(h_img)
    #cv2.waitKey(0)
    #plt.show()


    return results

                

def FindMatchesDualCameraEx(cameras,cln_imgs,keypoints,id_1,id_2,bboxes,badkeys,verbose=False,bUSeVieww = False):
    colors = {}
    cam1 = cameras[id_1]
    cam2 = cameras[id_2]
    keys1 = keypoints[id_1]
    keys2 = keypoints[id_2]
    img1 = cln_imgs[id_1].copy()
    img2 = cln_imgs[id_2].copy()
    bboxes1 = bboxes[id_1]
    bboxes2 = bboxes[id_2]
    badkeys1 = badkeys[id_1]
    badkeys2 = badkeys[id_2]
    results = {}
    
    print(f'--------------Starting Cam: {id_1} -- Cam: {id_2}-------------------------------------------')
    res_key = (id_1,id_2)
    results[res_key] = {}
    results[res_key]['img'] = None
    results[res_key]['pairs'] = []
    results[res_key]['mse'] = []
    results[res_key]['keys_3d'] = []
    # iterate through 
    for key in keys2:
        id,key3d,mse = FindMatchEx(keys2,keys1,key,cam2,cam1,bUSeVieww,verbose)
        results[res_key]['pairs'].append((key,id))
        results[res_key]['mse'].append(mse)
        results[res_key]['keys_3d'].append(key3d)
        if(mse > 0 and mse < 100):
            color =  GetRandomColor()
            colors[(key,id)] = color
            colors[(id,key)] = color
            for p in keys1[id]:
                PlotPointOnImage(img1,p[0],p[1],color)
        print(f'{key} {id} {mse}')

    print('-----------------------------------------------------------------')
    for key in keys1:
        id,key3d,mse = FindMatchEx(keys1,keys2,key,cam1,cam2,bUSeVieww,verbose)
        results[res_key]['pairs'].append((key,id))
        results[res_key]['mse'].append(mse)
        results[res_key]['keys_3d'].append(key3d)
        if(mse > 0 and mse < 100):
            color = None
            if (key,id) in colors:
                color = colors[(key,id)]
            elif (id,key) in colors:
                color = colors[(id,key)]
            else:
                color = GetRandomColor()
                colors[(key,id)] = color
                colors[(id,key)] = color
            for p in keys2[id]:
                PlotPointOnImage(img2,p[0],p[1],color)


        print(f'{key} {id} {mse}')
    print(f'--------------Ending Cam: {id_1} -- Cam: {id_2}-------------------------------------------')
    for key in bboxes1:
        if key in badkeys1:
            continue
        r = bboxes1[key]
        DrawBbox(img1,r)
        cv2.putText(img1, text='box_id: '+str(key), org=r[0,:],
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0),
                thickness=1, lineType=cv2.LINE_AA)
    for key in bboxes2:
        if key in badkeys2:
            continue
        r = bboxes2[key]
        DrawBbox(img2,r)
        cv2.putText(img2, text='box_id: '+str(key), org=r[0,:],
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0),
                thickness=1, lineType=cv2.LINE_AA)

    h_img = cv2.hconcat([img1, img2])
    results[res_key]['img'] = h_img
    #cv2.imshow(f"Output_cam{id_1}", img1)
   # cv2.imshow(f"Output_cam{id_2}", img2)
    #cv2.imshow("Output_Conb", h_img)
    #plt.figure(figsize = (80,60))
    #plt.imshow(h_img)
    #cv2.waitKey(0)
    #plt.show()


    return results

def FindMatchEx(keypointsCam1: dict, keypointsCam2: dict,id_cam1,cam1: Camera,cam2: Camera,bUseVieww = False,verbose = False):
    keypoints1 = keypointsCam1[id_cam1] # goalkeeper
    best_id = -1
    best_3dpoints = []
    min_mse = 10e10
    count = 0
    for key in keypointsCam2:
        keypoints2 = keypointsCam2[key]
        points3d = [] 
        mse = 0
        bSkip = False
        for i in range(len(keypoints1)):         
            dir_1 = GeometryUtilities.GetPointCameraRayFromPixel(keypoints1[i,:],cam1.K_inv,cam1.R_inv,cam1.tvec)
            dir_2 = GeometryUtilities.GetPointCameraRayFromPixel(keypoints2[i,:],cam2.K_inv,cam2.R_inv,cam2.tvec)
            if bUseVieww == True:
                l1p1 = cam1.pos
                l1p2 = cam1.pos + 200 * dir_1

                l2p1 = cam2.pos
                l2p2 = cam2.pos + 200 * dir_2
                dot,intersection = GeometryUtilities.DistanceBetweenTwoLinesPow2(l1p1,l1p2,l2p1,l2p2)
            else:
                intersection  = GeometryUtilities.RayRayIntersectionEx(cam1.pos,dir_1,cam2.pos,dir_2)
                intersectiona  = GeometryUtilities.RayRayIntersectionEx(cam2.pos,dir_2,cam1.pos,dir_1)
                a,b =  GeometryUtilities.RayRayIntersectionExDual(cam1.pos,dir_1,cam2.pos,dir_2)
                print(f'{intersection} {intersectiona} {a} {b}' )
            if(intersection is None):
                print("BAD POINT No Intersection")
                count += 1
                bSkip = True
                break
            intersection = -1 * intersection
            points3d.append(intersection)

            pixel_cam1 = ProjectPointsCV(intersection,cam1.K,cam1.rvec,cam1.tvec,cam1.dist_coeffs)
            pixel_cam2 = ProjectPointsCV(intersection,cam2.K,cam2.rvec,cam2.tvec,cam2.dist_coeffs)
            error = np.abs(pixel_cam2 - keypoints2[i,:])
            error_sqr = np.linalg.norm((pixel_cam2 - keypoints2[i,:]))
            mse += error_sqr
            if(verbose):
                print(f"id: {i} rect id cam 2: {key}, {intersection} {keypoints2[i,:]} reproj: {pixel_cam2} err: {error}, err_sqrt {error_sqr}")
            if(intersection[2] < 0 or intersection[0] < -60 or intersection[0] > 60 or  intersection[1] < -10 or intersection[1] > 60 ):
                if(verbose):
                    print("BAD POINT")
                count += 1
                bSkip = True
                break

        if( min_mse > mse and bSkip == False):
            best_id = key
            min_mse = mse
            best_3dpoints = points3d
        if(verbose):
            print(f"Bad Points count {count}")
            print("-----------------------------------------------------------------------------------")

    return best_id, best_3dpoints,min_mse

def FindMatchMedium(keypointsCam1: dict, keypointsCam2: dict,id_cam1,cam1: Camera,cam2: Camera,verbose = False):
    keypoints1 = keypointsCam1[id_cam1] # goalkeeper
    best_id = -1
    best_3dpoints = []
    min_mse = 10e10
    count = 0
    for key in keypointsCam2:
        keypoints2 = keypointsCam2[key]
        points3d = [] 
        mse = 0
        bSkip = False
        for i in range(len(keypoints1)):         
            dir_1 = GeometryUtilities.GetPointCameraRayFromPixel(keypoints1[i,:],cam1.K_inv,cam1.R_inv,cam1.tvec)
            dir_2 = GeometryUtilities.GetPointCameraRayFromPixel(keypoints2[i,:],cam2.K_inv,cam2.R_inv,cam2.tvec)
            a,b =  GeometryUtilities.RayRayIntersectionExDual(cam1.pos,dir_1,cam2.pos,dir_2)
            if(a is None or b is None):
                print("BAD POINT No Intersection")
                count += 1
                bSkip = True
                break
            intersection = (a+b) * 0.5
            intersection = -1 * intersection
            points3d.append(intersection)

            pixel_cam1 = ProjectPointsCV(intersection,cam1.K,cam1.rvec,cam1.tvec,cam1.dist_coeffs)
            pixel_cam2 = ProjectPointsCV(intersection,cam2.K,cam2.rvec,cam2.tvec,cam2.dist_coeffs)
            error = np.abs(pixel_cam2 - keypoints2[i,:])
            error_sqr = np.linalg.norm((pixel_cam2 - keypoints2[i,:]))
            mse += error_sqr
            if(verbose):
                print(f"id: {i} rect id cam 2: {key}, {intersection} {keypoints2[i,:]} reproj: {pixel_cam2} err: {error}, err_sqrt {error_sqr}")
            if(intersection[2] < 0 or intersection[0] < -60 or intersection[0] > 60 or  intersection[1] < -10 or intersection[1] > 60 ):
                if(verbose):
                    print("BAD POINT")
                count += 1
                bSkip = True
                break

        if( min_mse > mse and bSkip == False):
            best_id = key
            min_mse = mse
            best_3dpoints = points3d
        if(verbose):
            print(f"Bad Points count {count}")
            print("-----------------------------------------------------------------------------------")

    return best_id, best_3dpoints,min_mse




def FindMatchesDualCameraMedium(cameras,cln_imgs,keypoints,id_1,id_2,bboxes,badkeys,verbose=False):
    colors = {}
    cam1 = cameras[id_1]
    cam2 = cameras[id_2]
    keys1 = keypoints[id_1]
    keys2 = keypoints[id_2]
    img1 = cln_imgs[id_1].copy()
    img2 = cln_imgs[id_2].copy()
    bboxes1 = bboxes[id_1]
    bboxes2 = bboxes[id_2]
    badkeys1 = badkeys[id_1]
    badkeys2 = badkeys[id_2]
    results = {}
    
    print(f'--------------Starting Cam: {id_1} -- Cam: {id_2}-------------------------------------------')
    res_key = (id_1,id_2)
    results[res_key] = {}
    results[res_key]['img'] = None
    results[res_key]['pairs'] = []
    results[res_key]['mse'] = []
    results[res_key]['keys_3d'] = []
    # compare all each set of keypoints from img2 to the ones in the image 1 
    # using find match...
    for key in keys2:
        id,key3d,mse = FindMatchMedium(keys2,keys1,key,cam2,cam1,verbose)
        results[res_key]['pairs'].append((key,id))
        results[res_key]['mse'].append(mse)
        results[res_key]['keys_3d'].append(key3d)
        if(mse > 0 and mse < 100):
            color =  GetRandomColor()
            colors[(key,id)] = color
            colors[(id,key)] = color
            for p in keys1[id]:
                PlotPointOnImage(img1,p[0],p[1],color)

            for p in keys2[key]:
                PlotPointOnImage(img2,p[0],p[1],color)
        print(f'{key} {id} {mse}')

   
    print(f'--------------Ending Cam: {id_1} -- Cam: {id_2}-------------------------------------------')
    # render bounding box 
    for key in bboxes1:
        if key in badkeys1:
            continue
        r = bboxes1[key]
        DrawBbox(img1,r)
        cv2.putText(img1, text='box_id: '+str(key), org=r[0,:],
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0),
                thickness=1, lineType=cv2.LINE_AA)
    for key in bboxes2:
        if key in badkeys2:
            continue
        r = bboxes2[key]
        DrawBbox(img2,r)
        cv2.putText(img2, text='box_id: '+str(key), org=r[0,:],
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0),
                thickness=1, lineType=cv2.LINE_AA)

    h_img = cv2.hconcat([img1, img2])
    results[res_key]['img'] = h_img
    #cv2.imshow(f"Output_cam{id_1}", img1)
   # cv2.imshow(f"Output_cam{id_2}", img2)
    #cv2.imshow("Output_Conb", h_img)
    #plt.figure(figsize = (80,60))
    #plt.imshow(h_img)
    #cv2.waitKey(0)
    #plt.show()


    return results



def find_matches_dual_camera_medium(cameras,keypoints,id_1,id_2,bboxes,badkeys,verbose=False):
    cam1 = cameras[id_1]
    cam2 = cameras[id_2]
    keys1 = keypoints[id_1]
    keys2 = keypoints[id_2]
    bboxes1 = bboxes[id_1]
    bboxes2 = bboxes[id_2]
    badkeys1 = set(badkeys[id_1])
    badkeys2 = set(badkeys[id_2])
    results = {}
    #if(verbose):
    if(True):
        print(f'--------------Starting Cam: {id_1} -- Cam: {id_2}-------------------------------------------')
    res_key = (id_1,id_2)
    results[res_key] = {}
    results[res_key]['pairs'] = []
    results[res_key]['mse'] = []
    results[res_key]['keys_3d'] = []
    # compare all each set of keypoints from img2 to the ones in the image 1 
    # using find match...
    for key in keys2:
        if key in badkeys2:
            if(verbose):
                print("Badkey")
            continue 
        id,key3d,mse = find_match_avg(keys2,keys1,key,cam2,cam1,badkeys1,verbose)
        results[res_key]['pairs'].append((key,id))
        results[res_key]['mse'].append(mse)
        results[res_key]['keys_3d'].append(key3d)
        

    return results


def find_match_avg(keypointsCam1: dict, keypointsCam2: dict,id_cam1,cam1: Camera,cam2: Camera,badkeys: set,verbose = False):
    keypoints1 = keypointsCam1[id_cam1] # goalkeeper
    best_id = -1
    best_3dpoints = []
    min_mse = 10e10
    count = 0
    for key in keypointsCam2:
        if key in badkeys:
            if(verbose):
                print("Badkey find_match_avg")
            continue
        print(f"Comparing {key} with {id_cam1}") # id_cam1 is the left camera 4-5 is camera 5
        keypoints2 = keypointsCam2[key]
        points3d = [] 
        mse = 0
        bSkip = False
        for i in range(len(keypoints1)):         
            dir_1 = GeometryUtilities.GetPointCameraRayFromPixel(keypoints1[i,:],cam1.K_inv,cam1.R_inv,cam1.tvec,cam1.dist_coeffs,cam1)
            dir_2 = GeometryUtilities.GetPointCameraRayFromPixel(keypoints2[i,:],cam2.K_inv,cam2.R_inv,cam2.tvec,cam2.dist_coeffs,cam2)
            #a,b,sin =  GeometryUtilities.RayRayIntersectionExDual(cam1.pos,dir_1,cam2.pos,dir_2)
            a,b,sin =  GeometryUtilities.RayRayIntersectionExDualUpdated(cam1.pos,dir_1,cam2.pos,dir_2,eps=1e-4)
           # a,b,sin =  GeometryUtilities.RayRayIntersectionExDualVieww(cam1.pos,dir_1,cam2.pos,dir_2,eps=1e-4)
            if(a is None or b is None):
                print("BAD POINT No Intersection")
                count += 1
                bSkip = True
                break
            intersection = (a+b) * 0.5
            
            #Uncomment for Vieew space case 
            #intersection = -1 * intersection
            points3d.append(intersection)

            pixel_cam1 = ProjectPointsCV(intersection,cam1.K,cam1.rvec,cam1.tvec,cam1.dist_coeffs)
            pixel_cam2 = ProjectPointsCV(intersection,cam2.K,cam2.rvec,cam2.tvec,cam2.dist_coeffs)
            error = np.abs(pixel_cam2 - keypoints2[i,:])
            error_sqr = np.linalg.norm((pixel_cam2 - keypoints2[i,:]))
            error_sqr_1 = np.linalg.norm((pixel_cam1 - keypoints1[i,:]))
            # this should be the average 1st error 
            err_sqr = (error_sqr + error_sqr_1)/2.0
            
            mse += err_sqr #error_sqr
            if(verbose):
                print(f"id: {i} rect id cam 2: {key}, {intersection} {keypoints2[i,:]} reproj: {pixel_cam2} err: {error}, err_sqrt {error_sqr}")
            if(intersection[2] < 0 or intersection[0] < -60 or intersection[0] > 60 or  intersection[1] < -10 or intersection[1] > 60 ):
                if(verbose):
                    print("BAD POINT")
                count += 1
                bSkip = True
                break

        if( min_mse > mse and bSkip == False):
            best_id = key
            min_mse = mse
            best_3dpoints = points3d
        if(verbose):
            print(f"Bad Points count {count}")
            print("-----------------------------------------------------------------------------------")

    return best_id, best_3dpoints,min_mse