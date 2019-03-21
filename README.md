# python_cv2_cam_parallel
* python : 3.5.6
* opencv : 3.4.3

  * threading 사용시의 fps
  
  ![fps_threading](./img/threading.PNG)

  * multiprocessing 사용시의 fps
  
  ![fps_multiprocessing](./img/multiprocessing.PNG)
  
  I am not sure why. The biggest different is that detection thread of threading is faster than that of multiprocessing.  The others are almost the same.
