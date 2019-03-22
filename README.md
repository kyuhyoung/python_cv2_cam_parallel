# comparision of frames rates(fps) of fetching / object detection / display of camera feed between using threading and multiprocessing
* python : 3.5.6
* opencv : 3.4.3

  * frame rate when using threading
    * fetch : 8.5
    * detection : 29.9
    * display : 3.4
    
  
  ![fps_threading](./img/threading.PNG)

  * frame rates when using multiprocessing
    * fetch : 8.5
    * detection : 29.9
    * display : 
  
  ![fps_multiprocessing](./img/multiprocessing.PNG)
  
  I am not sure why. The biggest different is that detection thread of threading is faster than that of multiprocessing.  The others are almost the same.
