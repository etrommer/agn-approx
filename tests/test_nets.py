from agnapprox.nets import AlexNet, LeNet5, MobileNetV2, ResNet, VGG

def test_alexnet():
    dut = AlexNet(pretrained=False)

def test_lenet5():
    dut = LeNet5()

def test_mobilenetv2():
    dut = MobileNetV2(pretrained=False)

def test_resnet():
    dut = ResNet()

def test_vgg():
    dut = VGG(pretrained=False)