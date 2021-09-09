# Anomaly Detection from Normal Images


Main Ideas Taken From and Inspired By - Image Anomaly Detection Using Normal Data Only by
Latent Space Resampling

link : https://doi.org/10.3390/app10238660


_______________________________________________________________

# Example 1 


| Anomoulous Image | Direct Recon | Resampled Recon | Anomaly Score Map | 
| --- | --- | --- | --- |
| <img src="images/original_anomalous_screw.png" height="256" width="256"> | <img src="images/direct_recon_screw.jpeg" height="256" width="256"> | <img src="images/resampled_recon_screw.jpeg" height="256" width="256"> | <img src="images/anomaly_score_map.jpeg" height="256" width="256"> |



_________________________________________________________________

# Example 2

| Anomoulous Image | Direct recon | Resampled recon | Anomaly score map | 
| --- | --- | --- | --- |
| <img src="images/original.jpeg"> | <img src="images/direct.jpeg"> | <img src="images/resampled.jpeg"> | <img src="images/difference.jpeg"> |


______________________________________________________________________

VQ-VAE architecture was inspired by :  Aäron van den Oord et al. in PyTorch (https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb)
SimplePixelCNN architecture was inspired by :  https://github.com/pilipolio/learn-pytorch/blob/master/201708_ToyPixelCNN.ipynb




