# Visual-Odometry
This project explains how to implement a visual odometry for a stereo camera system. Stereo Matching of the images is done using Semi Global Block Matching.



```
├── Code
|  ├── stereovision.py
|  ├── helper.py
├── Docs
|  ├── VisualOdometry_Report.pdf
├── Results
|  |  ├── .png files
```
### Input Dataset 


Please downlaod the dataset from this [drive link](https://drive.google.com/drive/folders/1XbLQADGrB_WN5mZ-QMd1rzsP1wD1nOR2?usp=sharing).

### Packages needed 

- Opencv2
- Numpy
- Numba
- Matplotlib
- scipy


### Steps to Run the code 


```
git clone --recursive https://github.com/karanamrahul/Visual-Odometry.git
cd Visual-Odometry
python3 stereovision.py

```

You need to select the dataset and the method you want to use for stereo matching. ( Using Semi Global might take a little more time compared to the first method as it checks for four directions around the pixel)
```
Please choose the dataset number (1) - Curule (2) - Octagon (3) - Pendulum : 


Please choose the method number (1) - Simple Block Matching  (2) - Semi Global Block Matching :
```

### Results


##### Dataset - 1 Simple block matching algorithm
![alt test](https://github.com/karanamrahul/Visual-Odometry/blob/main/results/curule_out_60.png)
##### Dataset - 2
![alt test](https://github.com/karanamrahul/Visual-Odometry/blob/main/results/octagon_out_80.png )
##### Dataset - 3
![alt test](https://github.com/karanamrahul/Visual-Odometry/blob/main/results/pendulum_out_180.png)



#### Dataset - 1 Semi-global block matching algorithm
![alt test](https://github.com/karanamrahul/Visual-Odometry/blob/main/results/curule_semi_60.png)
##### Dataset - 2
![alt test](https://github.com/karanamrahul/Visual-Odometry/blob/main/results/octagon_semi_80.png )
##### Dataset - 3
![alt test](https://github.com/karanamrahul/Visual-Odometry/blob/main/results/pendulum_semi_180.png)



