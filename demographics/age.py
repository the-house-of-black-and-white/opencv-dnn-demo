from . import CaffeClassifier


class AgeClassifier(CaffeClassifier):

    def __init__(self, prototxt='models/caffe/age_gender-0.0.2/deploy_age.prototxt', caffemodel='models/caffe/age_gender-0.0.2/age_net.caffemodel'):
        CaffeClassifier.__init__(self, prototxt, caffemodel, ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100'])

