# comparision of frames rates(fps) for fetching / object detection / display of camera feed between using threading and using multiprocessing of python
* python : 3.5.6
* cv2 : 3.4.3

  * Both frame rates of using threading and multiprocessing in my codes are almost the same. (I expected that multiprocessing would be fast. Point me if my usage of threading and/or multiprocessing is wrong !!).  The typical frame rates are as below.  
    * fetch : 8.5
    * detection : 29.9
    * display : 3.4
    
  ![fps_threading](./img/threading.PNG)
<!--  

  * frame rates when using multiprocessing
    * fetch : 8.5
    * detection : 29.9
    * display : 1.6
  
  ![fps_multiprocessing](./img/multiprocessing.PNG)
  
  I am not sure why. The biggest different is that detection thread of threading is faster than that of multiprocessing.  The others are almost the same.
-->
