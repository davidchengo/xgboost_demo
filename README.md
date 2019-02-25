# xgboost in R demo <br/>
Kaggle Otto Group Product Classification Challenge 2015. Multi-class product classification.

## Example results <br/>
### Feature ranking (relative importance) <br/>
Top 10 Features
![](img/top10.png)
First 10 Spits of the Tree
> model[1:10]
 [1] "booster[0]"                                                       
 [2] "0:[f16<1.5] yes=1,no=2,missing=1,gain=274.435059,cover=12222.8145"
 [3] "1:[f77<2.5] yes=3,no=4,missing=3,gain=88.3581543,cover=11424"     
 [4] "3:leaf=-0.450779825,cover=11219.5557"                             
 [5] "4:leaf=-0,cover=204.444443"                                       
 [6] "2:[f15<0.5] yes=5,no=6,missing=5,gain=34.6999054,cover=798.814819"
 [7] "5:leaf=0.329302341,cover=357.333344"                              
 [8] "6:leaf=-0,cover=441.481476"                                       
 [9] "booster[1]"                                                       
[10] "0:[f13<1.5] yes=1,no=2,missing=1,gain=7751.69873,cover=12222.8145"
Each line represents a branch, there is the tree ID, the feature ID, the point where it splits, and information regarding the next branches (left, right, when the row for this feature is N/A).
