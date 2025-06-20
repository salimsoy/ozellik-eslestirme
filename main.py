import cv2
import numpy as np
from matplotlib import pyplot as plt

class FeatureMatching:
    
    def orb_matching(self,image):
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(image, None)
        return kp, des
    def flann_based_matcher(self, des1, des2):
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm= FLANN_INDEX_LSH,
                            table_number = 6,
                            key_size = 12,
                            multi_probe_level = 1)
        search_params = dict(checks= 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.matches = flann.knnMatch(des1, des2, k=2)
        matchesMask = [[0, 0] for i in range(len(self.matches))]
        for i, val in enumerate(self.matches):
            if len(self.matches[i]) < 2:
                continue
            m, n = val
            if m.distance < 0.65 * n.distance:
                matchesMask[i] = [1,0]
        
        self.draw_params = dict(matchColor = (0, 255, 0),
                           singlePointColor = None,
                           matchesMask = matchesMask,
                           flags = 2)
        
        
if __name__ == '__main__':
    
    img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('box_test.png', cv2.IMREAD_GRAYSCALE)
    proses = FeatureMatching()
    kp1, des1 = proses.orb_matching(img1)
    kp2, des2 = proses.orb_matching(img2)
    proses.flann_based_matcher(des1, des2)
    
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, proses.matches, None, **proses.draw_params)
    
    cv2.imwrite("output.jpg", img3)
    plt.imshow(img3,),plt.show()