# Self-matching
## dataset：都存在ttssvv里面
### nor:　还原词根,删除token小于5，删除有url的句子
### one:  还原词根,删除token小于5，删除有url的句子，删除词频小于１
### 剩下的是什么都没有处理的

##loaddata:
###不考虑test所以testData.tsv是随便乱写的句子

###　optimizer:Adam比rmspop更好
###  word embedding:300维比100维更好
###  梯度裁剪结果会更好
