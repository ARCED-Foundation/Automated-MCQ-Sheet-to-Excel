# Lets cut to the chase! The output will look like this.
![Loading Image Output](https://github.com/ARCED-Foundation/Automated-MCQ-Sheet-to-Excel/blob/master/temp%20pic.png)

# Table of Contents
- [Environmental Factor to be kept in check](#environmental-factor-to-be-kept-in-check)
- [Steps on how to set up in your local computer](#steps-on-how-to-set-up-in-your-local-computer)
- [How to Run](#how-to-run)




# Environmental Factor to be kept in check 
> [!NOTE]
> - Take picture with CAM SCANNER
> - Picture should be taken roughly from the top not from an angle.
> - While taking picture make sure there is sufficient light

# Steps on how to set up in your local computer 
> [!WARNING]
> There is a requirement.txt file. Do not run it, The version will vary which will cause it to not work. (It is there to show you which packages you will need).

### 1. Your python Version
Your Python Version must be less than 3.4 
> [!TIP]
> Best to use python version 3.10

### 2. Clone this repo
```
git https://github.com/AdnanSalazzar/Check-Box-Detector.git
```

### 3. Make a Virtual Environment and activate it
```
py -3.10 -m venv myenv
myenv\Scripts\activate
```

### 4. Install Numpy
```
pip install numpy==1.23.5
```

### 5. Download the ultralytics by the version given below 
```
pip install ultralytics==8.0.230
```

# How to Run 

### Make a Virtual Environment and activate it
```
py -3.10 -m venv myenv
myenv\Scripts\activate
```

### Upload images in the imgFolder
> [!NOTE]
> - The images will be renamed with numbers 

### Finally run the YoloPlusTextFolder.py 
> [!IMPORTANT]
> Make sure the directory path are correct (For models and your folder)
```
python "Your python code address to the file”
```








   












