# EVA
All assignments of EVA TSAI
S11<br>
Write a code which uses this new ResNet Architecture for Cifar10:<br>
PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]<br>
Layer1 -<br>
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
Add(X, R1)
Layer 2 -<br>
Conv 3x3 [256k]
MaxPooling2D
BN
ReLU
Layer 3 -<br>
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
Add(X, R2)
MaxPooling with Kernel Size 4
FC Layer
SoftMax<br>

Uses One Cycle Policy such that:
Total Epochs = 24
Max at Epoch = 5
LRMIN = FIND
NO Annihilation<br>
Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
Batch size = 512
Target Accuracy: 90%.
