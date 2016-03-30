#**Image Processing Projects**

This repository is a collection of all things fun in image processing achieved with opencv and python. Projects and implementations are ever so on random topics but interesting ones nevertheless.

 1. Image_Lib - contains common code files that is reused by most projects.
 2. PyImageSearchProblems - Kudos to Adrian for his awesome blog on image processing in opencv - [PyImageSearch](http://www.pyimagesearch.com/). The files in this folder mostly follow some of his blogs with my flavor to the problems here and there.
 3. PythonProjects - My playground! Every time someone mentions or I think of an interesting problem, it ends up here.
 4. SelfProjectUtils - Code that I usually use to understand images for tuning parameters in other projects and such.
 
 
Few example results:
 - With the availability of displays in various sizes, image retargeting or image resizing with content awareness is something that's done frequently nowadays.  A simple implementation of seam carving a well known method and it's result(width reduction by 20%) is as below.  Note that the content of the image is not scaled or cropped.
   
![Input image](https://github.com/shekkizh/ImageProcessingProjects/blob/master/Dog.jpg)                  ![Seam reduced image](https://github.com/shekkizh/ImageProcessingProjects/blob/master/results/20PercentWidthReduction.jpg)

-  Panoramic image stitching using SIFT/ SURF features.

<img src="https://github.com/shekkizh/ImageProcessingProjects/blob/master/Image1.jpg" width = "400" height = "300"/>            <img src="https://github.com/shekkizh/ImageProcessingProjects/blob/master/Image2.jpg" width = "400" height = "300"/>
<img src="https://github.com/shekkizh/ImageProcessingProjects/blob/master/results/ImageStitiching.jpg" width = "800" height = "300"/>  
-  Image Cartooning. 

![Image](https://github.com/shekkizh/ImageProcessingProjects/blob/master/results/CartoonishImaging.jpg)

-  Color transforms and compression.

<img src="https://github.com/shekkizh/ImageProcessingProjects/blob/master/results/10ClusterImage.jpg" width = "400" height = "300"/>       <img src="https://github.com/shekkizh/ImageProcessingProjects/blob/master/results/ImageDithering.jpg" width = "400" height = "300"/>       <img src="https://github.com/shekkizh/ImageProcessingProjects/blob/master/results/OldSchoolSadnessFilter.jpg" width = "400" height = "300"/>       <img src="https://github.com/shekkizh/ImageProcessingProjects/blob/master/results/StatisticalColorTransform.jpg" width = "400" height = "300"/>

-   Auto detect size of objects in images given a reference object. In the example below the height of iPhoneSE (Reference object in this case) was the only value that was provided to the algorithm.

<img src="https://github.com/shekkizh/ImageProcessingProjects/blob/master/results/AutoDetectedPhoneSize.jpg" width = "449" height = "290"/>   <img src="https://github.com/shekkizh/ImageProcessingProjects/blob/master/results/ActualPhoneSize.jpg" width = "400" height = "175"/>
