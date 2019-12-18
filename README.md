# DeepUAge
DCA - Dlib Contour Artistic Approach for face preprocessing.

DCA uses facial proportions to reconstruct the face and obtain landmarks that are close to the hairline. 
It operates with [dlib] and extends to the prediction of the hairline with a facial proportion artistic approach
based on the work of Andrew Loomis[1].

DCA approach for landmark detection:

![alt text][logo]

You can crop an image using:

python dca.py --f image_file_name --o image_file_path


If you find this code useful in your research, please consider citing:

```
@article{anda2020UnderageAgeEstimation,
	author={Anda, Felix and Le-Khac, Nhien-An and Scanlon, Mark},
	title="{DeepUAge: Improving Underage Age Estimation Accuracy to Aid CSEM Investigation}",
	journal="{Forensic Science International: Digital Investigation}",
	year="2020",
	month="03",
	publisher={Elsevier}
  }
```

[logo]: https://github.com/4ND4/DeepUAge/blob/master/fgnet-artistic_contour.png "fgnet artistic contour"
[dlib]: http://dlib.net/
