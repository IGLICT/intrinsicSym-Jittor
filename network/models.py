import jittor as jt
from jittor import Module, nn
import jittor as jt
jt.flags.use_cuda = 1

class SignNet(Module):
    """ ConvNet baseline, input is BxNxK gray image """
    def __init__(self, num_point, dim_input=4):
        super().__init__()
        self.conv1 = nn.Conv(in_channels=1, out_channels=64, kernel_size=(1, dim_input), stride=(1, 1), padding=0)
        self.bn1 = nn.BatchNorm(64)
        self.conv2 = nn.Conv(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn2 = nn.BatchNorm(128)
        self.conv3 = nn.Conv(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn3 = nn.BatchNorm(256)
        self.conv4 = nn.Conv(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn4 = nn.BatchNorm(512)
        self.conv5 = nn.Conv(in_channels=512, out_channels=4096, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn5 = nn.BatchNorm(4096)  # B  * 4096 * num_point * dim_input
        # TODO: it seems that jittor do not have modulelist?

        self.conv_net = nn.Sequential([self.conv1, self.bn1, nn.Relu(), self.conv2, self.bn2, nn.Relu(),
                                       self.conv3, self.bn3, nn.Relu(), self.conv4, self.bn4, nn.Relu(),
                                       self.conv5, self.bn5, nn.Relu()])
        self.pool = nn.Pool(kernel_size=(num_point, 1), stride=(2, 2), padding=0, op='maximum')
        # classification network
        self.fc1 = nn.Linear(in_features=4096, out_features=512)
        self.fc_bn1 = nn.BatchNorm(512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc_bn2 = nn.BatchNorm(128)
        self.fc3 = nn.Linear(in_features=128, out_features=32)
        self.fc_bn3 = nn.BatchNorm(32)
        self.fc4 = nn.Linear(in_features=32, out_features=2)
        # self.fc_bn4 = nn.BatchNorm(2)
        self.fc_net = nn.Sequential([self.fc1, self.fc_bn1, nn.Relu(),
                                     nn.Dropout(p=0.7),
                                     self.fc2, self.fc_bn2, nn.Relu(),
                                     nn.Dropout(p=0.7),
                                     self.fc3, self.fc_bn3, nn.Relu(),
                                     self.fc4  # , self.fc_bn4, nn.Relu(),
                                     ])

        self.criterion = nn.CrossEntropyLoss()

    def execute(self, model_input, model_sign=None):
        # TODO: report this
        model_input = jt.float32(model_input)
        if model_sign is not None:
            model_sign = jt.float32(model_sign)
        input_image = jt.unsqueeze(model_input, 1)  # B * num_point * dim_input * 1
        conv_result = self.conv_net(input_image)   # B * 4096 * num_point * dim_input

        max_result = self.pool(conv_result)
        # classification
        x = jt.reshape(max_result, [-1, 4096])
        # import pdb
        # pdb.set_trace()
        my_sign_score = self.fc_net(x)
        # TODO: just p my_sign_score cause error
        _, my_sign = jt.argmax(my_sign_score, dim=1)
        if model_sign is not None:
            # loss
            loss = self.criterion(my_sign_score, model_sign)
            return my_sign, loss
        else:
            return my_sign