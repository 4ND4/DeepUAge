# DeepUAge
DCA - Dlib Contour Artistic Approach for face preprocessing.

DCA uses facial proportions to reconstruct the face and obtain landmarks that are close to the hairline. 
It operates with [dlib] and extends to the prediction of the hairline with a facial proportion artistic approach
based on the work of Andrew Loomis[1].

[dlib]: http://dlib.net/





@inproceedings{plummerCITE2018,
Author = {Bryan A. Plummer and Paige Kordas and M. Hadi Kiapour and Shuai Zheng and Robinson Piramuthu and Svetlana Lazebnik},
Title = {Conditional Image-Text Embedding Networks},
Booktitle  = {ECCV},
Year = {2018}
}


You can crop an image using:

python dca.py --f image_file_name --o image_file_path

Here's our logo:

Reference-style: 
![alt text][logo]

[logo]: https://github.com/4ND4/DeepUAge/raw/master/src/common/images/icon48.png "Logo Title Text 2"
