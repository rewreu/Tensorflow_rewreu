## protobuf files ready for transfer learning
Both pb files here can feed in with multiple images
#### The inception V4:

Input tensor name: "InputImage:0", shape ?,299,299,3

Output tensor name: "InceptionV4/Logits/Logits/MatMul:0", shape ?1,2048

The file is too large for github, using dropbox: https://www.dropbox.com/s/ee0cgl7ui2ytavj/inception_v4_batch.pb?dl=1

#### The inception V3 has

Input tensor name: "Placeholder:0", shape ?,299,299,3

Output tensor name: "InceptionV3/Logits/Conv2d_1c_1x1/Conv2D:0"), shape ?1,2048