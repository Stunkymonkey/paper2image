# paper2image
detect papers in pictures and cuts of the useless infos

```
Usage: paper2image.py [options]

Options:
  -h, --help            show this help message and exit
  -i IMAGE, --image=IMAGE
                        Path of image which will be scanned
  -g, --grayscale       only having black and white
  -v, --verbose         showing the results
  ```
The original file:
<br>
<img src="https://github.com/Stunkymonkey/paper2image/blob/master/demo.jpeg" width="256">

The output of `python2.7 paper2image.py -i demo.jpeg` (only the paper):
<br>
<img src="https://github.com/Stunkymonkey/paper2image/blob/master/demo-image.jpeg" width="256">

The output of `python2.7 paper2image.py -i demo.jpeg -g` (converting colors to black and white):
<br>
<img src="https://github.com/Stunkymonkey/paper2image/blob/master/demo-gray.jpeg" width="256">
