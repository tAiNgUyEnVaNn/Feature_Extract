import cv2
import numpy as np
import time
from sklearn.cluster import DBSCAN

# object_detector = cv2.createBackgroundSubtractorMOG2()


def initial(frame):
    r = cv2.selectROI("select the area", frame)
    roi_x1 = r[0]
    roi_x2 = r[2] + r[0]
    roi_y1 = r[1]
    roi_y2 = r[3] + r[1]
    cv2.destroyWindow('select the area')
    return frame[roi_y1:roi_y2, roi_x1:roi_x2]



# Reading frame
# frame_prev = cv2.imread('frame_0004.jpg')
# frame_prev = frame_prev[640:727, 565:622]
# # roi_x, roi_y, roi_width, roi_height = 565, 640, 57, 87
# frame_next = cv2.imread('frame_0005.jpg')

def findBox(frame_prev, frame_next):
    # Create Feature Extracting algorithm
    detector = cv2.SIFT_create()
    kps_prev, des_prev = detector.detectAndCompute(frame_prev, None)  
    kps_next, des_next = detector.detectAndCompute(frame_next, None) 

    kps_prev_coord = [(kps.pt[0], kps.pt[1]) for kps in kps_prev] 
    kps_next_coord = [(kps.pt[0], kps.pt[1]) for kps in kps_next] 

    # Create a Brute-Force matcher
    matcher = cv2.BFMatcher()

    # Match the descriptors between the two frames
    matches = matcher.match(des_prev, des_next)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Select the top N matches (e.g., 100)
    top_matches = matches[:100]

    # Extract the matching keypoints from both frames
    matched_kps_prev = np.array([kps_prev[match.queryIdx] for match in top_matches])
    matched_kps_next = np.array([kps_next[match.trainIdx] for match in top_matches])

    kps_prev_coord = [(kps.pt[0], kps.pt[1]) for kps in matched_kps_prev] 
    kps_next_coord = [(kps.pt[0], kps.pt[1]) for kps in matched_kps_next] 
    # matched_keypoints_next

    # Draw bouding box around densest area of bouding box
    dbscan = DBSCAN(eps=20, min_samples=5)
    clusters_prev = dbscan.fit_predict(kps_prev_coord)  
    clusters_next = dbscan.fit_predict(kps_next_coord) 

    # Find the cluster with the highest density in previous frame
    unique_clusters_prev, cluster_counts_prev = np.unique(clusters_prev, return_counts=True)
    densest_cluster_prev = unique_clusters_prev[np.argmax(cluster_counts_prev)] 
    # Find the cluster with the highest density in next frame
    unique_clusters_next, cluster_counts_next = np.unique(clusters_next, return_counts=True)
    densest_cluster_next = unique_clusters_next[np.argmax(cluster_counts_next)]


    # Get the kps belonging to the previous densest cluster
    densest_cluster_kps_prev = [kps for kps, cluster in zip(matched_kps_prev, clusters_prev) if cluster == densest_cluster_prev]
    # Get the kps belonging to the previous densest cluster
    densest_cluster_kps_next = [kps for kps, cluster in zip(matched_kps_next, clusters_next) if cluster == densest_cluster_next]
    # Compute the bounding box coordinates for the densest cluster
    box_prev = cv2.boundingRect(np.array([kps.pt for kps in densest_cluster_kps_prev], dtype=np.float32)) #xp, yp, wp, hp
    box_next = cv2.boundingRect(np.array([kps.pt for kps in densest_cluster_kps_next], dtype=np.float32)) #xp, yp, wp, hp
    
    # box ~ xp, yp, wp, hp
    
    return box_prev, box_next, kps_prev, densest_cluster_kps_next, top_matches

# Draw the bounding box on the image
# kps_prev, kps_next, top_matches = findBox(frame_prev, frame_next)

# (xp, yp, wp, hp), (xn, yn, wn, hn), kps_prev, kps_next, top_matches  = findBox(frame_prev, frame_next)

videoPath = 'person.mp4'
cap = cv2.VideoCapture(0)

ret, firstFrame = cap.read()
if ret:
    FRAME_H, FRAME_W, _ = firstFrame.shape


# Calculate the scale up 
def deltaxy(w, h):
    delta_x = int(w/2)
    delta_y = int(h/2)
    return delta_x, delta_y

    

start_time = time.time()
frame_count = 0
fps = 0
frame_prev = None
frame_curr = None
run = False
delta = 10
count = 0

while True:
    ret, frame = cap.read()
    # frame = object_detector.apply(frame)
    # cv2.imwrite("frame_curr", )
    # if count == 1:
    #     cv2.imwrite("frame_prev.jpg", frame_prev)
    #     cv2.imwrite("frame_curr.jpg", frame_curr)
    #     break
    if not ret:
        break

    press = cv2.waitKey(1) & 0xFF
    if press == ord('b'):
        frame_prev = initial(frame)
        run = True
    elif press == ord('q'):
        break

    cv2.imshow("frame", frame) 
    if run:
        count +=1 
        if count % 2 != 0:
            continue
        frame_curr = frame.copy()
        cv2.imwrite('frame_prev.jpg', frame_prev)
        cv2.imwrite("frame_curr.jpg", frame_curr)
        _, box_next, _, kps_prev, _ = findBox(frame_prev, frame_curr)
        print("no prob in here")
        xn, yn, wn, hn = box_next
        print("box", box_next)
        dx, dy = deltaxy(wn, hn)
        # print(dx, dy)
        output_image = cv2.rectangle(frame, (xn, yn), (xn + wn, yn + hn), (255, 255, 255), 2)
        output_image = cv2.drawKeypoints(frame, kps_prev, None)

        frame_prev = frame_curr[yn-dy:yn+hn+dy, xn-dx:xn+wn+dx]
        # cv2.imwrite('frame.jpg', frame)
        # cv2.imwrite('frame_curr.jpg', frame_curr)
        # cv2.imwrite('frame_output.jpg', output_image)
        # break
        cv2.imshow("track", output_image)

        frame_count += 1
        # calculate elaspe time
        elapsed_time = time.time() - start_time
        # update FPS every second
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            print(fps)

        # print("count", count)
        continue

    # frame_count += 1
    
    # # Calculate the elapsed time
    # elapsed_time = time.time() - start_time
    
    # # Update FPS every second
    # if elapsed_time >= 1.0:
    #     fps = frame_count / elapsed_time
    #     frame_count = 0
    #     start_time = time.time()
    #     print(fps)

cap.release()
cv2.destroyAllWindows()