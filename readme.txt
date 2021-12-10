一、数据生成阶段
 1、Network.py      将原网络数据文件转化为networkx文件

 2、Centrality_Rank.py     生成中心性特征以及节点的中心性排名。中心性特征保存路径：./Data/TrainingFeatures/               排名文件保存路径：./Data/SortedCentrality/

 3、Get_single_IC_influence.py         基于IC生成单个节点的影响力，结果保存路径：./Data/SortedSingleInfluence/

 4、InfluenceMaximitionAlgorithm.py       基于影响力最大化算法计算7组影响力最大节点组。结果保存路径：./Data/InfluenceMaximizationGroup/
     
    参数说明：

	--size         group节点数量，默认值为5
	
	--p             传播概率，默认值为0.05
	
	--mc          传播轮数，默认值为100

	--ta            mia算法中需要生成的节点组数量，默认值为0.1


 5、Group_construction.py             基于前面的结果文件，通过三种策略（变异，topN随机生成，全图随机生成）组建groups数据集，结果保存路径：./Data/GroupData/
     
    参数说明：

	--size         group节点数量，默认值为5
	
	--times      变异次数，默认值为100

	--topN       策略二中选取排名前topN个节点，默认值为50

	--num1      策略二中需要生成的节点组数量，默认值为1000

	--num2      策略三中需要生成的节点组数量，默认值为5000

 6、ImModelMC.py         基于IC，生成所有节点组的影响规模，结果保存路径：./Data/GroupInfluence/

    参数说明：

	--p         传播概率，默认值为0.05
	
	--mc      传播轮数，默认值为100


二、训练阶段

 1、data_load.py         包含了原data_load函数，以及将节点表示转化为节点组表示的聚合函数，模型调用时需 from data_load import *

 2、train.py                 调用对应模型文件和数据进行训练和测试

    新增参数说明：

	--aggregate_mode         聚合类型，通过该参数来选择不同的聚合函数（在data_load.py中定义），默认值为'average'，即使用平均计算策略

 3、GCN.py                说明：已修改完成，在网络email-Eu-core上测试通过。使用命令  python train.py email-Eu-core  来实现在该网络上GCN模型的训练和测试。 









