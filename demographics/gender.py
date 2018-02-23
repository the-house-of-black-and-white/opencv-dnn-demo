from . import CaffeClassifier


class GenderClassifier(CaffeClassifier):

    def __init__(self, prototxt='models/caffe/age_gender-0.0.2/deploy_gender.prototxt',
                 caffemodel='models/caffe/age_gender-0.0.2/gender_net.caffemodel'):
        CaffeClassifier.__init__(self, prototxt, caffemodel, ['M', 'F'])


class DexGenderClassifier(CaffeClassifier):

    def __init__(self, prototxt='models/caffe/dex/gender.prototxt', caffemodel='models/caffe/dex/gender.caffemodel'):
        CaffeClassifier.__init__(self, prototxt, caffemodel, ['M', 'F'])

    def dims(self):
        return 224, 224
