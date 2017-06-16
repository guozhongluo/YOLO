# YOLO

>>YOLO  VS2013 c++ code， Only need opencv, do not rely on the Caffe Library

# How to run?

command line： darknet detector test cfg/voc.data cfg/tiny-yolo-voc.cfg tiny-yolo-voc.weights data/dog.jpg

# model down

https://pjreddie.com/media/files/tiny-yolo-voc.weights

# Result:

# input picture

![image](https://github.com/guozhongluo/YOLO/blob/master/darknet_lgz/person.jpg)

# out result

![image](https://github.com/guozhongluo/YOLO/blob/master/darknet_lgz/predictions1.png)


# Program running speed test:


  1 Intel(R) Core(TM) i7-4790 CPU @3.6GHZ 8G(RAM)    run a picture cost 0.9 second

  2 use one Titan GPU, can run  20/fps;

