import numpy as np
import cv2 # opencv
#import glm
class GeometryUtilities():
    @staticmethod
    def CalculateIntrinsicMatrix(camera_params: dict,image_size: tuple,pixel_size_mm):
        """Calculate the Initrinsic/ Projection matrix from the marapeters provided
         Args:
            camera_params: dictionary 
            image_size: size of the image in pixels
            pixel_size_mm: size of the pixels in mm
        """
        pixel_size_mm = camera_params["pixel_size_mm"]
        #[sx, sy] are the number of pixels per world unit in the x and y direction respectively
        sx = 1/pixel_size_mm # rg
        sy = 1/pixel_size_mm
        f = camera_params['f']
        fx = f*sx
        fy = f*sy
        center = (image_size[1]/2, image_size[0]/2)
        cx = center[0]
        cy = center[1]
       
        camera_matrix = np.array(
                         [[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0,  1]], dtype = "double")
        return camera_matrix

    def CalculateIntrinsicMatrixNew(camera_params: dict,image_size: tuple,pixel_size_mm):
        """Calculate the Initrinsic/ Projection matrix from the marapeters provided
         Args:
            camera_params: dictionary 
            image_size: size of the image in pixels
            pixel_size_mm: size of the pixels in mm
        """
        pixel_size_mm = camera_params["pixel_size_mm"]
        sx = 1/pixel_size_mm
        sy = 1/pixel_size_mm
        f = camera_params['f']
        fx = f*sx
        fy = f*sy
        center = (image_size[1]/2, image_size[0]/2)
        cx = center[0]
        cy = center[1]
        camera_matrix = np.array(
                         [[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0,  1]], dtype = "double")
        return camera_matrix

    @staticmethod
    def GetCameraInformationFromPoints(xyz: np.array ,uv: np.array,camera_matrix: np.array,dist_coeffs:np.array):
        """Providing at n-points in 3d space and the relative images points in the image plane. 
           Return a distionary containing the Extrinsic matric, the translation vector and the camera position 
           
        Args:
            xyz: nx3 points in World space
            xyz: nx3 points in World space
        Returns:
            nx4 points in Homogeneous by appending 1
           """
        (success, rotation_vector, translation_vector) = cv2.solvePnP(xyz, uv, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rotation_matrix,_ = cv2.Rodrigues(rotation_vector) #3x3 matrix
        camera_pos = np.linalg.inv( rotation_matrix) @ translation_vector

        return rotation_matrix,rotation_vector,translation_vector,camera_pos


    # https://stackoverflow.com/questions/23275877/opencv-get-perspective-matrix-from-translation-rotation

    @staticmethod
    def CalculateRotationMatrixRHS(camera_params: dict, is_row_major: bool,is_opencv = True) -> np.array:
        """  
        Calculate the rotatation matrix for a rhs space and the translation vector 
        Args:
            xyz: nx3 points in World space
            xyz: nx3 points in World space
        Returns:
            nx4 points in Homogeneous by appending 1 """
        R = np.zeros((3,3))
        eye = camera_params['pos']
        #eye = M_xyz @ eye
        target = camera_params['target']
        #target = M_xyz @ target
        up = camera_params['up'] 
        #up = M_xyz @ camera_params['up'] 
        forward = (eye - target)
        forward /= np.linalg.norm(forward)
        right = np.cross(up,forward) 
        # rinormalize the up vector
        up = up/np.linalg.norm(up)
        
        right = right/np.linalg.norm(right)
        up = np.cross(forward,right)
      
        up = up/np.linalg.norm(up)
        R = np.zeros((3,3))

        # row major vs column major stuff here to be checked 
        if not is_row_major:
            R[0,:] = right
            R[1,:] = up
            R[2,:] = forward
            # row major
        else:
            R[:,0] = right
            R[:,1] = up 
            R[:,2] = forward
       
        tvec = np.zeros((3,),dtype='double')
        tvec[0] = -np.dot(right,eye)
        tvec[1] = -np.dot(up,eye)
        tvec[2] = -np.dot(forward,eye)
      

        return R, tvec

    @staticmethod
    def CalculateRotationMatrix(camera_params: dict, is_row_major: bool,is_opencv = True) -> np.array:
        """  
        Calculate the rotatation matrix for a rhs space and the translation vector 
        Args:
            xyz: nx3 points in World space
            xyz: nx3 points in World space
        Returns:
            nx4 points in Homogeneous by appending 1 """
        R = np.zeros((3,3))
        eye = camera_params['pos']
        #eye = M_xyz @ eye
        target = camera_params['target']
        #target = M_xyz @ target
        up = camera_params['up'] 
        #up = M_xyz @ camera_params['up'] 
        forward = (eye - target)
        forward /= np.linalg.norm(forward)
        right = np.cross(up,forward) 
        # rinormalize the up vector
        up = up/np.linalg.norm(up)
        
        right = right/np.linalg.norm(right)
        up = np.cross(forward,right)
      
        up = up/np.linalg.norm(up)
        R = np.zeros((3,3))

        # row major vs column major stuff here to be checked 
        if not is_row_major:
            R[0,:] = right
            R[1,:] = up
            R[2,:] = forward
            # row major
        else:
            R[:,0] = right
            R[:,1] = up 
            R[:,2] = forward
       
        tvec = np.zeros((3,),dtype='double')
        tvec[0] = np.dot(right,eye)
        tvec[1] = np.dot(up,eye)
        tvec[2] = np.dot(forward,eye)
      

        return R, tvec
        
    @staticmethod
    def CalculateRotationMatrixLHS(camera_params: dict, is_row_major: bool):
        """  
        Calculate the rotatation matrix for a rhs space and the translation vector 
        Args:
            xyz: nx3 points in World space
            xyz: nx3 points in World space
        Returns:
            nx4 points in Homogeneous by appending 1 """
        R = np.zeros((3,3))
        M_xyz =  np.array([[1, 0, 0],
                    [0 ,0,1],
                    [0,1, 0]], dtype ='double')

        eye = camera_params['pos']
        #eye = M_xyz @ eye
        target = camera_params['target']
        #target = M_xyz @ target
        up = camera_params['up'] 
        #up = M_xyz @ camera_params['up'] 
        forward = (target -eye)
        forward /= np.linalg.norm(forward)
        right = np.cross(up,forward) 
        # rinormalize the up vector
        up = up/np.linalg.norm(up)
        
        right = right/np.linalg.norm(right)
        up = np.cross(forward,right)
      
        up = up/np.linalg.norm(up)
        R = np.zeros((3,3))

        # row major vs column major stuff here to be checked 
        if not is_row_major:
            R[0,:] = right
            R[1,:] = up
            R[2,:] = forward
            # row major
        else:
            R[:,0] = right
            R[:,1] = up 
            R[:,2] = forward    
       
        tvec = np.zeros((3,),dtype='double')
        tvec[0] = -np.dot(right,eye)
        tvec[1] = -np.dot(up,eye)
        tvec[2] = -np.dot(forward,eye)

        return R, tvec
    @staticmethod
    def NormalizeVector(v: np.array) -> np.array:
        """
            Given oa vector of the form return the normalized 
            vector

            Args:
            v: input vector
           
        Returns:
            The normalized vector"""
      
        return v/np.linalg.norm(v)

    @staticmethod
    def GetPointCameraRayFromPixel(pixel,k_inv,rot_inv,tvec,dist_coeffs=None,cam=None):
        # fix for pixel distortion 
        # https://groups.google.com/g/vsfm/c/IcbdIVv_Uek/m/Us32SBUNK9oJ?pli=1
        # https://newbedev.com/opencv-distort-back
        # https://programming.vip/docs/detailed-explanation-of-camera-model-and-de-distortion-method.html

        # this part checks the reprojection of the vector in 
        # finding ray going from camera center to pixel coord
        hom_pt = np.array([pixel[0],pixel[1],1],dtype='double')
        #hom_pt = np.array([721,391,1],dtype='double')
        dir_cam_space = k_inv @ hom_pt  # this bit transform points in camera space 

        # https://answers.opencv.org/question/148670/re-distorting-a-set-of-points-after-camera-calibration/ # <- this
        # TODO: apply distortion coefficients
        
        #if(cam.dist_coeffs is not None ) and (np.sum(cam.dist_coeffs) != 0):
        #    #print("APPLYING DIST Coefficients")
        #    x,y = GeometryUtilities.distort_point(dir_cam_space[0],dir_cam_space[1],cam,hom_pt)
        #    #print(f"{x} {y} {dir_cam_space}")
        #    dir_cam_space[0] = x
        #    dir_cam_space[1] = y
        
        dir_in_world  = rot_inv @ dir_cam_space #+ bb
        return dir_in_world
  
    @staticmethod
    def GetPointCameraRayFromPixelEx(pixel,k_inv,rot_inv,tvec):
        # this part checks the reprojection of the vector in 
        # finding ray going from camera center to pixel coord
        hom_pt = np.array([pixel[0],pixel[1],1],dtype='double')
        #hom_pt = np.array([721,391,1],dtype='double')
        dir_cam_space = k_inv @ hom_pt 
        
        dir_in_world  = rot_inv @ dir_cam_space + rot_inv @ tvec
        return GeometryUtilities.NormalizeVector(dir_in_world)

    @staticmethod
    # view matrix for the rhs system with y = up
    def CalculateViewMatrix(eye,target, up):
        forward = eye-target
        forward /= np.linalg.norm(forward)
        right = np.cross(up,forward)
        right /= np.linalg.norm(right)
       # up = np.cross(forward,right)
        #up /= np.linalg.norm(up)

        R = np.zeros((3,3),dtype='double')
        R[:,0] = right
        R[:,1] = up 
        R[:,2] = forward

        tvec = np.zeros((3,),dtype = 'double')
        tvec[0] = np.dot( right,eye )
        tvec[1] = np.dot( up,eye )
        tvec[2] = np.dot( forward,eye )

        return R,tvec
    @staticmethod
    #https://math.stackexchange.com/questions/270767/find-intersection-of-two-3d-lines
    # M= C± ||f X CD||/||f×e|| e.
    # TODO: missing error checking in case of parallel rays 
    def RayRayIntersection(origin_ray1,dir_ray1,origin_ray2,dir_ray2,eps=10e-2):
        g = origin_ray2 - origin_ray1
        #g /= np.linalg.norm(g)
        cross_r2g = np.cross(dir_ray2,g)
        cross_r2r1 = np.cross(dir_ray2,dir_ray1)
        
        val = np.linalg.norm( np.cross(dir_ray2,g))/np.linalg.norm( np.cross(dir_ray2,dir_ray1))
        val_Konst = val * dir_ray1
        # check if choosing the plus or the minus 
        cross_fg = np.cross(dir_ray2,g)
        cross_fe = np.cross(dir_ray2,dir_ray1)
        cross_fe /= np.linalg.norm(cross_fe)
        cross_fg /= np.linalg.norm(cross_fg)
        dot_prod  = np.dot(cross_fg,cross_fe)
        intersection = np.zeros((3,))
        intersection = None

        if(dot_prod > 0 and np.abs(dot_prod -1) <= eps):
            intersection = origin_ray1 + val_Konst
        elif (dot_prod < 0 and np.abs(dot_prod + 1) <= eps):
            intersection = origin_ray1 - val_Konst
        else :
            print("Error indirections")
            return None

        #print(f'{intersection}')
        return intersection


    # https://math.stackexchange.com/questions/270767/find-intersection-of-two-3d-lines
    # M = C± ||f X CD||/||f×e|| e.
    # Updated version takes into account the medium point amomng the two calculations
    def RayRayIntersectionEx(origin_ray1,dir_ray1,origin_ray2,dir_ray2,eps=10e-2):
        g = origin_ray2 - origin_ray1
        cross_r2g = np.cross(dir_ray2,g)
        cross_r2r1 = np.cross(dir_ray2,dir_ray1)
        len_r2r1 = np.linalg.norm(cross_r2r1)
        len_r2g = np.linalg.norm(cross_r2g)
        cross_r2r1_norm = cross_r2r1/len_r2r1
        cross_r2g_norm = cross_r2g/len_r2g
        val = len_r2g/len_r2r1
        val_Konst = val * dir_ray1

        # check if choosing the plus or the minus 
        dot_prod  = np.dot(cross_r2g_norm,cross_r2r1_norm)
        intersection1 = None


        if(dot_prod > 0 and np.abs(dot_prod -1) <= eps):
            intersection1 = origin_ray1 + val_Konst
        elif (dot_prod < 0 and np.abs(dot_prod + 1) <= eps):
            intersection1 = origin_ray1 - val_Konst
        else :
            print("Error in directions")
            return None

        #print(f'{intersection}')
        return intersection1

    def RayRayIntersectionExDualVieww(origin_ray1,dir_ray1,origin_ray2,dir_ray2,eps=1e-3):
        #float CRzWorld3D::DistanceBetweenTwoLinesPow2(vec3 l1p1, vec3 l1p2, vec3 l2p1, vec3 l2p2, vec3 &mpt)
        k_dir = 10
        p1 = origin_ray1 
        p2 = origin_ray1 + k_dir * dir_ray1
        p3 = origin_ray2 
        p4 = origin_ray2 + k_dir * dir_ray2

        l1dir = p2 - p1
        l2dir = p4 - p3
        a11 = np.dot(l1dir, l1dir)
        a22 =  np.dot(l2dir, l2dir)
        a12 = -np.dot(l1dir, l2dir)
        r = origin_ray2 - origin_ray1
        b1 =  np.dot(l1dir, r);
        b2 = -np.dot(l2dir, r);
        # Cramer determinants
        d0 = a11 * a22 - a12 * a12;
        d1 = b1 * a22 - b2 * a12;
        d2 = b2 * a11 - b1 * a12;
        if (((d0>0) and ((0<d1 and d1<d0) or (0<d2 and d2<d0))) or ((d0<0) and ((0>d1 and d1>d0) or (0<d2 and d2>d0)))):
            
            x1 = origin_ray1 + l1dir*(d1 / d0)
            x2 = origin_ray2 + l2dir*(d2 / d0)
            mpt = (x1 + x2) * 0.5
            dx = x1 - x2
            return x1,x2 #np.dot(dx,dx)

        return None,None,None



    ## Possible fix: https://math.stackexchange.com/questions/2738535/intersection-between-two-lines-3d
    def RayRayIntersectionExDualUpdated(origin_ray1,dir_ray1,origin_ray2,dir_ray2,eps=1e-3):
        #Vector3 line2Point1, Vector3 line2Point2, out Vector3 resultSegmentPoint1, out Vector3 resultSegmentPoint2)
        #Algorithm is ported from the C algorithm of 
        #Paul Bourke at http://local.wasp.uwa.edu.au/~pbourke/geometry/lineline3d/
        resultSegmentPoint1 = np.zeros(3)
        resultSegmentPoint2 = np.zeros(3)
        k_dir = 10
        p1 = origin_ray1 
        p2 = origin_ray1 + k_dir * dir_ray1
        p3 = origin_ray2 
        p4 = origin_ray2 + k_dir * dir_ray2
        
        p13 = p1 - p3
        p43 = p4 - p3
 
        if ( np.linalg.norm( p43) < eps):
            return None,None
   
        p21 = p2 - p1;
        if ( np.linalg.norm( p21) < eps):
            return None,None
   
        d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2]
        d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2]
        d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2]
        d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2]
        d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2]
 
        denom = d2121 * d4343 - d4321 * d4321
        if (np.abs(denom) < eps):
            return None,None
   
        numer = d1343 * d4321 - d1321 * d4343;
 
        mua = numer / denom;
        mub = (d1343 + d4321 * (mua)) / d4343;
 
        resultSegmentPoint1[0] = p1[0] + mua * p21[0]
        resultSegmentPoint1[1] = p1[1] + mua * p21[1]
        resultSegmentPoint1[2] = p1[2] + mua * p21[2]
        resultSegmentPoint2[0] = p3[0] + mub * p43[0]
        resultSegmentPoint2[1] = p3[1] + mub * p43[1]
        resultSegmentPoint2[2] = p3[2] + mub * p43[2]

        cross = np.cross(dir_ray1,dir_ray2)
        norm_cross  = np.linalg.norm(cross)
        v1_norm  = np.linalg.norm(dir_ray1)
        v2_norm  = np.linalg.norm(dir_ray2)
        sin_theta = norm_cross/(v1_norm * v2_norm) 

        return resultSegmentPoint1,resultSegmentPoint2,sin_theta



    # https://math.stackexchange.com/questions/270767/find-intersection-of-two-3d-lines
    # M = C± ||f X CD||/||f×e|| e.
    # Updated version takes into account the medium point amomng the two calculations
    def RayRayIntersectionExDual(origin_ray1,dir_ray1,origin_ray2,dir_ray2,eps=1e-3):
        k_dir = 10
        p1 = origin_ray1 
        p2 = origin_ray1 + k_dir * dir_ray1
        p3 = origin_ray2 
        p4 = origin_ray2 + k_dir * dir_ray2
        
        
        
        A = p1-p3
        B = p2-p1
        C = p4-p3
        #
        ## Line p1p2 and p3p4 unit vectors
        uv1 =  (p2-p1)/np.linalg.norm(p2-p1)
        uv2 =  (p4-p3)/np.linalg.norm(p4-p3)
        
        ## Check for parallel lines
        cp12 =  np.cross(uv1, uv2)
        _cp12_ = np.linalg.norm(cp12)
        Pa = None
        Pb = None
        inters_dist = None
        #
        if round(_cp12_, 6) != 0.0:         
            ma = ((np.dot(A, C)*np.dot(C, B)) - (np.dot(A, B)*np.dot(C, C)))/ \
                 ((np.dot(B, B)*np.dot(C, C)) - (np.dot(C, B)*np.dot(C, B)))
            mb = (ma*np.dot(C, B) + np.dot(A, C))/ np.dot(C, C)
            
            # Calculate the point on line 1 that is the closest point to line 2
            Pa = p1 + ma*B
            
            # Calculate the point on line 2 that is the closest point to line 1
            Pb = p3 + mb*C
            
            # Distance between lines            
            inters_dist = np.linalg.norm(Pa-Pb)
            
            return Pa,Pb
        else:
            return None,None


        g = origin_ray2 - origin_ray1
        cross_r2g = np.cross(dir_ray2,g)
        cross_r2r1 = np.cross(dir_ray2,dir_ray1)
        len_r2r1 = np.linalg.norm(cross_r2r1)
        len_r2g = np.linalg.norm(cross_r2g)
        cross_r2r1_norm = cross_r2r1/len_r2r1
        cross_r2g_norm = cross_r2g/len_r2g
        val = len_r2g/len_r2r1
        val_Konst = val * dir_ray1

        # check if choosing the plus or the minus 
        dot_prod  = np.dot(cross_r2g_norm,cross_r2r1_norm)
        intersection1 = None


        if(dot_prod > 0 and np.abs(dot_prod -1) <= eps):
        #if(dot_prod > 0 and np.abs(dot_prod -1) <= eps):
            intersection1 = origin_ray1 + val_Konst
        #elif (dot_prod < 0 and np.abs(dot_prod + 1) <= eps):
        elif (dot_prod < 0 and np.abs(dot_prod + 1) <= eps):
            intersection1 = origin_ray1 - val_Konst
        else :
            print("Error in directions")
            return None,None

        # find second intersection 
        cross_r1g = np.cross(dir_ray1,-g)
        len_r1g = np.linalg.norm(cross_r1g)
        cross_r1g_norm = cross_r1g/len_r1g
        val = len_r1g/len_r2r1
        val_Konst = val * dir_ray2
        dot_prod  = np.dot(cross_r1g_norm,-cross_r2r1_norm)
        intersection2 = None
        
        if(dot_prod > 0 and np.abs(dot_prod -1) <= eps):
            intersection2 = origin_ray2 + val_Konst
        elif (dot_prod < 0 and np.abs(dot_prod + 1) <= eps):
            intersection2 = origin_ray2 - val_Konst
        else :
            print("Error in directions")
            return None,None



        #print(f'{intersection}')
        return intersection1,intersection2

    @staticmethod
    def GetFov(f=35e-3,psize=4.8e-6,h=972.0):
       
        sy = 1/psize
        fy = sy*f
        const = h/(2 * fy)
        fovy_rad = 2 * np.arctan(const)
        fov_deg = np.degrees(fovy_rad)
        return fov_deg

    @staticmethod
    def CalculateProjection(camera_params: dict, img_size,nk = 0.75,fk=1.25):

        # calculate fov 
        # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
        pixel_size_mm = camera_params["pixel_size_mm"]
        sx = 1/pixel_size_mm
        sy = 1/pixel_size_mm
        f = camera_params['f']
        fx = f*sx
        fy = f*sy
        center = (img_size[1]/2, img_size[0]/2)
        cx = center[0]
        cy = center[1]
        eye = camera_params["pos"]
        
        dist = np.linalg.norm(eye)
        fovY = 2 * np.arctan(np.deg2rad(img_size[0])/2 * fy)
        fovX = 2 * np.arctan(np.deg2rad(img_size[1])/2 * fx)

        pGlm = glm.perspective(fovY,img_size[1]/img_size[0],nk * dist,fk * dist)
        return pGlm

    @staticmethod
    def LineRectIntersection2D(line: np.array,rect: np.array) -> bool:
        pass

    @staticmethod
    def DistanceBetweenTwoLinesPow2( l1p1: np.array,  l1p2:  np.array, l2p1:  np.array, l2p2: np.array) -> np.array:
        l1dir = l1p2 - l1p1
        l2dir = l2p2 - l2p1
        
        a11 = np.dot(l1dir, l1dir)
        a22 = np.dot(l2dir, l2dir)
        a12 = -np.dot(l1dir, l2dir);
        r = l2p1 - l1p1;
        b1 = np.dot(l1dir, r);
        b2 = -np.dot(l2dir, r);

        # Cramer determinants
        d0 = a11 * a22 - a12 * a12;
        d1 = b1 * a22 - b2 * a12;
        d2 = b2 * a11 - b1 * a12;
        if (((d0 > 0) and ((0 < d1 and d1 < d0) or (0 < d2 and d2 < d0))) or ((d0 < 0) and ((0 > d1 and d1>d0) or (0<d2 and d2>d0)))):

            x1 = l1p1 + l1dir*(d1 / d0);
            x2 = l2p1 + l2dir*(d2 / d0);
            mpt = (x1 + x2) * 0.5   #vec3(0.5f, 0.5f, 0.5f);
            dx = x1 - x2;
            return np.dot(dx,dx),mpt;

        return -1.0,None # error

    def LineRectIntersection2D(line: np.array,rect: np.array) -> bool:
        pass

    # https://yangyushi.github.io/code/2020/03/04/opencv-undistort.html
    #static inline vec2 RadialDistortion(vec2 &pt/*pixel position*/, const int2 &imgDim, const vec3 &radDist)
    #{
    #    vec2 cp = vec2(imgDim.x / 2, imgDim.y / 2);
    #    vec2 r = pt - cp;
    #    r *= GetRadialDistortionFactor(r, cp, radDist);
    #    return r + cp;
    #}

    #static inline float GetRadialDistortionFactor(vec2 &r, vec2 &cp, const vec3 &radDist)
    #{
    #float max_r2 = cp.x*cp.x + cp.y*cp.y;
    #float nr2 = (r.x*r.x + r.y*r.y) / max_r2;
    #return 1.0f + radDist.x*nr2 + radDist.y*nr2*nr2 + radDist.z*nr2*nr2*nr2;
    #}




    @staticmethod
    def distort_point(x,y,cam,hom_pt): # pixel in camera position 
        cx = cam.K[0][2]
        cy = cam.K[1][2]
        fx = cam.K[0][0]
        fy = cam.K[1][1] 
        # view version
        cp = np.zeros(2);
        cp[0] = cx
        cp[1] = cy

        r = hom_pt[:2] - cp
        max_r2 = cp[0]*cp[0] + cp[1]*cp[1];
        nr2 = (r[0]*r[0] + r[1]*r[1]) / max_r2;
        k_dist = 1.0 + cam.dist_coeffs[0]*nr2 #+ radDist.y*nr2*nr2 + radDist.z*nr2*nr2*nr2;
        new_pixel = np.zeros(2)
        new_pixel = r + cp
       
        #return new_pixel[0],new_pixel[1]
        
        p = np.zeros((2,),dtype='float32')
        p[0] = hom_pt[0]
        p[1] = hom_pt[1]
        p_norm = cv2.undistortPoints(p, cam.K, None) # this has been  already done 
        p_norm_hom = cv2.convertPointsToHomogeneous( p_norm );
        rtemp = ttemp = np.array([0,0,0], dtype='float32') # no rotation not translation for projection 
        p_image_dist,_ = cv2.projectPoints( p_norm_hom, rtemp, ttemp, cam.K, cam.dist_coeffs);
        p_image_dist  = cv2.convertPointsToHomogeneous( p_image_dist );
        p_image_dist = p_image_dist.reshape((3,))
        #p_img_dir = cam.K_inv @ p_image_dist 
        
        
        # test 2
        # To relative coordinates <- this is the step you are missing.
      
         
        x1 = ( hom_pt[0] - cx) / fx;
        y1 = ( hom_pt[1] - cy) / fy;
        x0 = x1
        y0 = y1
        # use iterative function to compensate distortion
        r2 = x1*x1 + y1*y1;
        k1 = cam.dist_coeffs[0]
        
        iter_num = 10
        
        for i in range(iter_num):
            r2 = x1 *x1 + y1*y1
            k_inv = 1 / (1 + k1 * r2)
            #Radial distorsion
            x1 = x0 *k_inv
            y1 = y0 *k_inv

        xDistort = x1
        yDistort = y1
        #xDistort = x1 * (1 + k1 * r2 )#+ k2 * r2 * r2 + k3 * r2 * r2 * r2);
        #yDistort = y1 * (1 + k1 * r2) #+ k2 * r2 * r2 + k3 * r2 * r2 * r2);


        # Tangential distorsion
        #xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
        #yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

        # Back to absolute coordinates.
        xDistort_p = xDistort * fx + cx;
        yDistort_p = yDistort * fy + cy;

   
        
        # calculate new camera matrix
        #K_new = cv2.

        r2 = x*x + y*y
        k1 = cam.dist_coeffs[0]  
        kr2 = r2*k1
       # x_dist = x/(1 - kr2)
        x_dist = x*(1 + kr2)
        y_dist = y*(1 + kr2)
       # y_dist = y/(1 - kr2)
        #p_out = cam.K_inv @ np.array([])
        return xDistort,yDistort

