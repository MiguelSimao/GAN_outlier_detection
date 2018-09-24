# GAN-based detection of out-of-vocabulary gestures

Improvement of the detection rate of gesture outliers with a feed-forward neural network trained as the discriminator in the Generative Adversatial Network.
![enter image description here](https://github.com/MiguelSimao/GAN_outlier_detection/blob/master/pictures/samples.png?raw=true)

## Description
This is a way of solving the issue of classification of out-of-vocabulary gestures. Very often, these gestures are classified as an existing class and are difficult to remove with a classification threshold. The proposed solution has two components: (1) the use of a generative model (GAN) to augment the data set with new generated samples, (2) the use of noisy labels to decrease the average probability of predictions, thus facilitating threshold tuning.

## Usage
Use the file **dualmyo_gan_generator_train.py** to train the generator and discriminator networks. The networks can then be tested with the scripts on **tests_discriminator.py** and **tests_generator.py**.  The description of the methodology is to be published soon.

The networks are implemented in [Keras](https://github.com/keras-team/keras). Aditional libraries used are: numpy, scikit-learn and tensorflow.
## Contributors

Miguel Simão:
 - [Google Scholar](https://scholar.google.com/citations?user=_xkTazsAAAAJ&hl=pt-PT)
 - [Linkedin](https://www.linkedin.com/in/miguels1mao/)

## License

This software is distributed under a MIT License.

Copyright (c) 2018 


## Acknowledgements
This work was partially funded by the Portuguese Foundation for Science and Technology (FCT) throught grant SFRH/BD/105252/2014.
