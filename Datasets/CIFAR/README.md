# CIFAR Dataset
 
* Test: ./test/test  10k files
* Train: ./train/train/[neg/pos] 10k files if each class (*_MARK.txt)

To import class in your project:
~~~~
# import CIFAR dataset with class selection

sys.path.insert(0, 'full_path_to_the_directory/CIFAR')
from CIFAR import CIFARDataset
~~~~
