"""Conservative camera-view shift heuristic, not a guaranteed camera-motion detector."""
import cv2
import numpy as np
from scene_math import displacement_warning


class SceneMonitor:
    def __init__(self):
        self.orb=cv2.ORB_create(nfeatures=1200)
        self.reference=None
        self.suspicion=0
        self.warned=False
    def update(self,frame,boxes):
        if self.warned:
            return True
        h,w=frame.shape[:2]
        width=640
        height=max(100,round(h*width/w))
        gray=cv2.cvtColor(cv2.resize(frame,(width,height)),cv2.COLOR_BGR2GRAY)
        mask=np.full(gray.shape,255,np.uint8)
        for b in boxes:
            a,c,e,f=[int(v*s) for v,s in zip(b,(width,height,width,height))]
            cv2.rectangle(mask,(max(0,a-8),max(0,c-8)),(min(width,e+8),min(height,f+8)),0,-1)
        keys,desc=self.orb.detectAndCompute(gray,mask)
        if desc is None or len(keys)<30:
            self.suspicion=0
            return False
        if self.reference is None:
            self.reference=(keys,desc,width,height)
            return False
        old,old_desc,rw,rh=self.reference
        if rw!=width or rh!=height:
            self.warned=True
            return True
        pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(old_desc,desc,k=2)
        matches=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.7*pair[1].distance]
        if len(matches)<25:
            self.suspicion=0
            return False
        source=np.float32([old[m.queryIdx].pt for m in matches])
        target=np.float32([keys[m.trainIdx].pt for m in matches])
        matrix,inliers=cv2.findHomography(source,target,cv2.RANSAC,3)
        if matrix is None or inliers is None or float(inliers.mean())<.65:
            self.suspicion=0
            return False
        good=source[inliers.ravel().astype(bool)]
        cells={(min(2,int(x/width*3)),min(2,int(y/height*3))) for x,y in good}
        # Local object motion should not be enough: require spread across the image.
        distributed=(len(good)>=25 and len(cells)>=4 and np.ptp(good[:,0])>width*.35 and np.ptp(good[:,1])>height*.35)
        moved=distributed and displacement_warning(matrix,width,height)
        self.suspicion=self.suspicion+1 if moved else 0
        self.warned=self.suspicion>=2
        return self.warned
