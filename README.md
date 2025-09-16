# ASDep: An Adversarial and Self-Supervised Framework for Cross-Subject EEG-Based Depression Classification

1. **use dataprocess:**  
```bash
2 goals: produce mat for GAN; process NPZ for classification
```

2. **use GAN:**  
```bash
Input the mat for GAN(one can be run for long time, so it needs add one by one)
```
3. **use classification model**  
```bash
After produce mat, use dataprocess to hug with origin dataset
```

**our mainline is follow by this:**
process edf,turn to mat and NPZ, 
a.mat add label ,and it can input for GAN.
b.GAN output NPZ, and it needs to hug with numpy.
c. finally make a collect data for classification in the model.

**thanks for dataset source:**

(https://figshare.com/articles/dataset/EEG_Data_New/4244171/2)
