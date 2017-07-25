# Python-FaceTemplateMatching
Using Python and OpenCV to achieve template matching and detection based on an input template and a webcam's video stream. I used a seperate thread for the webcam frame capture so as to improve performance. Using the LBP cascade also adds a couple extra frames per second, yet at the loss of accuracy (unless trained).

## Usage:

To run from command prompt, use:
```
FaceTemplateMatching.py <optionalArg>
```
while pointing to your root directory, where the optionalArg is the path to your template image. If left blank, it will default to a file named "template.png" in your root directory. Problems arise when no such file exists or no path is given.

If you provide a template file with more than one face in it, only one will be selected and checked against, and the rest will be discarded.

