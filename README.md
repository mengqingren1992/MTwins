# MTwins
The R funtion was uesd to find the phenotype-associated features
#### Install package dependency
* tidyverse
* fdrtool
* qvalue

#### Usage
```R
source(file = "Pair_Find.R")
pair_find(data=data,RAN_num=15,RAP_num=30,k="euclidean",rawoff=0.05)
```
#### Exaples
```R
set.seed(123)
data <- matrix(rnorm(120),nrow = 10,ncol = 12)
colnames(data) <- paste("Feature",1:12,sep="")
rownames(data) <- c(paste("Ctrl",1:5,sep=""),paste("Disease",1:5,sep=""))
head(data)
```

```R
#### Run Pair_find function
pair_find(data=data,RAN_num=5,RAP_num=5,k="euclidean",rawoff=0.01)
```

