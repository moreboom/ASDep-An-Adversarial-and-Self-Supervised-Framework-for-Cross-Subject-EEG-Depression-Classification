# ASDep: An Adversarial and Self-Supervised Framework for Cross-Subject EEG-Based Depression Classification

1. **read EDF:**  
```bash
run mdd_with_giter.py
```

2. **please go to dir '19':**  
```bash
if you want to input to classfication model
```
3. **please run as follow : mdd_with_giter.py for_gan_Mdd.py **  
```bash
if you want to input to GAN to make data
```

**our mainline is follow by this:**
process edf,turn to mat and NPZ, 
a.mat add label ,and it can input for GAN
b.GAN output NPZ, and it needs to hug with numpy.
c. finally make a collect data for classification

**final architecture**

![result](11.png)

