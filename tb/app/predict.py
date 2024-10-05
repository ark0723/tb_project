import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

"""
todo: 
1. predict letters and digit 
2. predict circle marked or not 
3. save the predicted data into database
4. evaluation 
"""
