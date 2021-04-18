# Master's thesis: Label inference for data clusters in point-based visualizations

<img src="https://user-images.githubusercontent.com/6630566/115135658-53c5d500-a01a-11eb-9d7f-8a1aa21111b7.png" width="720">


Abstract:

Two-dimensional point-based visualizations of multidimensional data may reveal data structures and clusters that require further interpretation. We present an approach that can automatically annotate the clusters in these visualisations. Our method extends the existing procedure for automatic annotation of two-dimensional representations of text documents and enables it for general attribute-value data. We propose to finds groups of points on scatterplot visualisations and assign them labels that describe a group’s characteristics in a language of the attributes of the original data. The approach uses DBSCAN clustering algorithm to find groups of points in the scatterplots. Statistical tests are used to determine labels for each of the groups. The proposed approach also features an interactive exploration of arbitrary subgroups manually chosen by the user. We analyze three datasets to demonstrate the usefulness of our approach. We show that the proposed method is sufficiently fast to support interactive analysis and that the group annotations found by our approach are meaningful.

## Repository structure:
* **data** : Datasets 
* **papers** : Related works recap (SLO language)
* **lib** : Source code / library
* **showcases** : Usage examples as jupyter notebooks

## Installing requirements
With Python3.6+ installed, run:
```
sudo apt-get install python3-dev
pip3 install -r requirements.txt
```

## Full thesis (SLO language)
https://repozitorij.uni-lj.si/Dokument.php?id=123767&lang=eng
