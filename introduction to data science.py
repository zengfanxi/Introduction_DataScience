"""
主题：导入离线数据对样本进行数据分析
时间：2019-07-08
分析工具：Python
"""
# -*- coding:utf-8 -*-
import pandas as pd
# 读入数据
m = pd.read_csv('.../house_info.csv',encoding = 'utf-8')
# 删除重复数据
print('当前数据条数是：%.0f' % m.shape[0])
m= m.drop_duplicates(keep = 'first')
print('删除重复值后剩余数据条数是：%.0f' % m.shape[0])

# 为了方便统计分析，我们去除样本中“合租“的样本
print(m['rentmodename'].value_counts())
m = m.loc[m.rentmodename =='整租'] #这剔除掉356个样本
print('剔除合租样本后的剩余数据条数是：%.0f'% m.shape[0])

"""
1.查看杭州各区域租赁房源数及均价分布
"""
test = m[['qyname','price']].groupby('qyname').agg(['count','mean']).reset_index()
levels = test.columns.levels
labels = test.columns.labels
test.columns = levels[0][labels[0]]+levels[1][labels[1]]
print(test.columns)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize =(10,6))
ax.set_ylabel('房源数量：套')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
#坐标轴1
ax.bar(test.qyname,test.pricecount,linewidth=2)
#坐标轴2
ax2 = ax.twinx()
ax2.plot(test.qyname,test.pricemean,'r',linewidth=2)
ax2.set_ylabel('房租均价：元／平米')
# plt.legend(loc='upper right')
plt.title('杭州各区域租赁房源数及均价分布')
plt.show()

"""
2.探索数据分析
影响租赁价格的因素很多，比如面积、位置、楼层、朝向、装修程度等等；
基于以上的变量，根据描述性数据分析，我们来验证下租赁价格的走势情况；
数据可视化部分，本文借助seaborn模块
"""
# 我们看看杭州不同区房屋租赁数量及整体租赁均价水平
# 容易发现区域命名这里存在很多不规范，比如“上城区”，有的样本中没有带上“区”字，直接是“上城”
print(m[['qyname','price']].groupby('qyname',as_index = False).agg(['count','mean']).reset_index())
# 将区域名称变量统一整理
m['qyname'] = m['qyname'].str.replace('区|市|经济开发|新','').astype(str) # 这里直接用replace不起作用
print(m[['qyname','price']].groupby('qyname',as_index = False).agg(['count','mean']).reset_index())

"""
2.1面积和租赁价格水平分布
   a)由于海宁和钱塘新区样本较少，绘图时暂时剔除两个类目
   b)和我们认知一样，面积和总价格呈现明显的线性关系
"""
m1 = m.loc[(m.qyname != '海宁')&(m.qyname != '钱塘')]
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")
sns.relplot(x='area', y='price',
            col= 'qyname',
            col_wrap=5,height=3, aspect=.8,
            #hue = 'decoratelevel',
            linewidth=1.5,
            #kind = 'line',
            data=m1[['price','area','qyname','decoratelevel']])
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# # 除了面积之外，当在分析其他特征和因变量关系时，才用租赁总价格有些不妥，因为租赁总价格本身就和面积关系很大，这里我们用每平米均价更合适
m['per_price'] = m.price/m.area
# 不同装修程度租赁水平分布
# 我们利用箱线图来观察不同装修水平和价格的关系
# 下图中很明显看到随着装修级别越高，房屋价格趋势也很高；同时箱线图也反映来样本中存在很多异常数据
plt.figure(figsize=(15, 7))
#plt.title('装修程度和价格的关系')
sns.boxplot(y='price', x='qyname', hue ='decoratelevel',
            data=m[['price','qyname','decoratelevel']].loc[(m.qyname !='海宁')&(m.qyname !='钱塘')])

"""
2.2 楼层与租金水平的分布
    这里要选取同一个小区的情况，不同小区的不同楼层之前价格不具备可比性
"""
# 先统计不同小区租赁信息排序
k = m['communityname'].value_counts().to_frame().reset_index().sort_values(by = 'communityname', ascending = False)
print(k.head(5))
# 取前5个小区来观察楼层和租金价格分布
l = m.loc[m['communityname'].str.contains('保利湾天地|紫西花语城' ,regex = True)]
plt.figure(figsize=(3, 7))
sns.relplot(x='buildingfloor', y='price',
            #col= 'qyname',
            height=6, aspect=.9,
            hue = 'communityname',
            linewidth=1.5,
            kind = 'line',
            legend="full",
            #estimator=None,
            facet_kws=dict(sharex=False),
            data=l[['price','communityname','buildingfloor']])

"""
2.3 新增板块的二手房在售均价和数量特征来对板块进行label encoding
    细分板块很多，而且不同板块之间价格有明显的差距，板块数据对预测分析是十分有利的；
    作为分类变量，如果one-hot处理，就会产生大量的稀疏列，因此我们考虑用另外一种方式来处理
    每个区的细分区块对房屋价格差距很大，比如西湖区的“学军”，“武林“板块和”五常“、”留下“板块价格就不在一个层次上
    基于此，通过观察我爱我家地图找房功能，我找到了下面的url可以得到不同板块的房屋在售数量和均价数据
    然后我利用每个板块的在售数量及均价数据来对表征这个板块
"""
import requests
import time
import json
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
import random
from tqdm import tqdm
from fake_useragent import UserAgent
import warnings
warnings.filterwarnings('ignore')
ua = UserAgent()
headers = {
            'accept': 'application/json, text/javascript, */*; q=0.01',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN,zh;q=0.9',
            'cache-control': 'no-cache',
            'cookie': '',
            'pragma': 'no-cache',
            'referer': 'https://m.5i5j.com/sh/zufang/index',
            'user-agent': ua.firefox,
            'x-requested-with': 'XMLHttpRequest',
        }
url ='https://hz.5i5j.com/map/ajax/location/sale?onMove=false&bounds={%22e%22:120.615301,%22w%22:119.562056,%22s%22:29.811872,%22n%22:30.735571}&boundsLevel=4&pageSize=20&page=0'
f = requests.get(url, headers = headers)
f1 = f.json()
f2 = f1['data']['res']['map']
name_sq,price_sq,total_sq = [],[],[]
# 将板块名称、房屋数量、房屋均价提取出来
for i in range(len(f2)):
    name_sq.append(f2[i]['name'])
    price_sq.append(f2[i]['price'])
    total_sq.append(f2[i]['total'])
info_sq = pd.DataFrame({'sqname':name_sq,'price_sq':price_sq,'total_sq':total_sq})
# info_sq中含有重复数据，这里利用取重复数据均值来替代填充，因为多个字段存在可能很大程度上由于手误导致
info_sq = info_sq.groupby('sqname',as_index = False)['price_sq','total_sq'].mean()

# 将以上提取的三个特征全部合并到原始样本当中
m1 = m.merge(info_sq, how = 'left', on =['sqname'])
# 填充之后发现total_sq列中存在部分缺失值的情况，"沿江"和"海宁"的数据缺失是由于前面的url中未提取出来导致
# 这里的解决方案是查询我爱我家官网获取，其中
# 海宁，price_sq：16500，total_sq：172；沿江，price_sq：26500，total_sq：636
import numpy as np
print(m1['sqname'].loc[np.isnan(m1.total_sq)].value_counts())
print(m1['sqname'].loc[np.isnan(m1.price_sq)].value_counts())
m1['price_sq'].loc[m1.sqname == '海宁'] = 16500
m1['total_sq'].loc[m1.sqname == '海宁'] = 172
m1['price_sq'].loc[m1.sqname == '沿江'] = 26500
m1['total_sq'].loc[m1.sqname == '沿江'] = 636

print(m1['sqname'].loc[np.isnan(m1.total_sq)].value_counts())
print(m1['sqname'].loc[np.isnan(m1.price_sq)].value_counts())

"""
3.构建模型特征
  利用回归模型预测房屋租赁价格水平
  可选特征有：livingroom-->卧室数量--不一定对（可能多余）
            buildingfloor-->当前楼层--不一定对（可能多余）
            heading-->朝向
            qyname -->区域名称
            subwaylines --> 地铁线路
            subwaystations -->地铁线路1
            toilet -->卫生间数量
            jtcx -->交通出行特点
            zbpt -->周边配套特点
            houseallfloor --> 房源总楼层数
            buildage --> 构造年份2（int）
            decoratelevelid --> 装修级别（int）
            fyld --> 房源亮点
            bedroom --> 卧室数量
            communityname --> 小区名称
            area --> 面积
            floorType --> 楼层特点
            location --> 地理位置经纬度
"""
feature = ['price','livingroom','buildingfloor','heading','qyname','subwaylines','subwaystations',
          'toilet','jtcx','zbpt','houseallfloor','buildage',
          'decoratelevelid','fyld','bedroom','communityname','area',
          'floorType','location','price_sq','total_sq']
data = m1[feature]
# 首先查看不同特征的缺失值情况
#print(data.isnull().sum())
# buildage_cn、buildyear、buildage都是指建造年份且三者缺失比例一致说明三者均缺失
# decoratelevel、decoratelevelid都是指装修级别，且二者缺失比例一致，说明二者均缺失
# 直接删除掉含有缺失值的行，共计减少261条数据
data = data.dropna()
# 方位数据处理
data.heading.value_counts()
data.heading.loc[(data.heading =='南')|(data.heading =='北')] = '南北'
data.heading.loc[(data.heading =='东')|(data.heading =='西')] = '东西'
print(data.heading.value_counts())

# 区域数据处理
# 杭州共有9个区，考虑到分类变量one-hot编码后，又会增加8个稀疏变量，这里改用2018年各区GDP财政数据和就业人员数作为替换变量
# GDP财政数据来自杭州市统计局2018年10月公布的统计年鉴
# url：http://tjj.hangzhou.gov.cn/content-getOuterNewsDetail.action?newsMainSearch.id=77fdf15544d84e429bccc0b43ece2708
# gdp单位：亿元；employers单位：人
# 由于钱塘新区年前刚成立，这里使用较低的常数值代替
gdp = {'西湖': 1121.1093,'余杭': 2034.4726,'江干': 624.0457,
       '上城': 1011.4403,'滨江': 1124.3357,'下城':860.5212,
       '拱墅':537.5017,'萧山': 2007.2357,'下沙': 652.3077,
       '海宁':866.32,'钱塘':99.00}
employers = {'西湖': 651987,'余杭': 473357,'江干': 474059,
             '上城': 242166,'滨江': 358926,'下城':295246,
             '拱墅':322032,'萧山': 829686,'下沙': 204846,
             '海宁':655573,'钱塘':99.00}

data['employers'] = data.qyname
data['gdp'] = data.qyname
data['employers'] = data['employers'].map(employers).astype(float)
data['gdp'] = data['gdp'].map(gdp).astype(float)

# subwaylines特征处理
# 发现subwaylines特征中存储的地铁线路数据，如果有周边有多条，对应list中也有多个线路值
data['subway'] = 0
data['subway'].loc[data['subwaylines'].str.contains('号线')]=1
# 根据现有数据开始对样本建模
feature_new = ['livingroom','buildingfloor','heading','toilet','houseallfloor','buildage',
               'decoratelevelid','bedroom','area','employers','gdp','subway','price','location','price_sq','total_sq']
trian_sample = data[feature_new]
# 先对heading字段one-hot处理，同时columns字段表示要处理的特征，drop_first表示剔除处理后的一个稀疏变量
trian_sample =pd.get_dummies(trian_sample, 
                             columns = ['heading'], 
                             drop_first = True, 
                             dtype = np.float)


# 发现toilet和bedroom两个字段中含有“多”文本，下面开始处理这部分异常数据
trian_sample[trian_sample['bedroom'].str.contains('多')].shape
# 联想到卧室数量和房屋面积客厅数量有关系,利用一个多维度图形来观察其中的复杂关系
plt.figure(figsize=(10, 7))
sns.boxplot(y='area', x='bedroom', hue ='livingroom',
            data=trian_sample.loc[(trian_sample['bedroom']!='多')&(trian_sample.livingroom >0)])
# livingroom=1的时候，平均面积在150以上的，bedingroom只有可能是6
trian_sample['bedroom'].loc[(trian_sample['livingroom'] ==1)&(trian_sample['bedroom'] =='多')] = '6'
# livingroom=2的时候，平均面积在130左右，bedingroom最有可能是3
# livingroom=2的时候，平均面积在150左右，bedingroom最有可能是4
# livingroom=2的时候，平均面积在200左右，bedingroom最有可能是5
# livingroom=2的时候，平均面积明显超过200，bedingroom最有可能是6
trian_sample['bedroom'].loc[(trian_sample['livingroom'] ==2)&(trian_sample['area']==126)] = '3'
trian_sample['bedroom'].loc[(trian_sample['livingroom'] ==2)&(trian_sample['area'] ==156)] = '4'
trian_sample['bedroom'].loc[(trian_sample['livingroom'] ==2)&(trian_sample['area'] >=180)] = '5'
trian_sample['bedroom'].loc[(trian_sample['livingroom'] ==2)&(trian_sample['area'] >=275)] = '6'
# livingroom=3的时候，面积在150左右，bedingroom最可能是3
trian_sample['bedroom'].loc[(trian_sample['livingroom'] ==3)&(trian_sample['area']<=160)] = '3'
# livingroom=3的时候，面积在200左右，bedingroom最可能是5
trian_sample['bedroom'].loc[(trian_sample['livingroom'] ==3)&(trian_sample['area']>160)] = '5'
# livingroom=3的时候，面积明显超过200，bedingroom最可能是4
trian_sample['bedroom'].loc[(trian_sample['livingroom'] ==3)&(trian_sample['area']>240)] = '4'


#trian_sample[trian_sample['toilet'].str.contains('多')]
# 考虑到卫生间数量和客厅及卧室数量有关，因此通过查看卧室、客厅、卫生间三者的分布箱线图：
plt.figure(figsize=(10, 7))
sns.boxplot(y='area', x='toilet', hue ='livingroom',
            data=trian_sample.loc[(trian_sample['toilet']!='多')&(trian_sample.livingroom >0)&(trian_sample['toilet']!='0')])

# trian_sample[trian_sample['toilet'].str.contains('多')]
trian_sample['toilet'].loc[(trian_sample['livingroom'] ==1)&(trian_sample['area'] >= 150)] = '3'
trian_sample['toilet'].loc[(trian_sample['livingroom'] ==2)&(trian_sample['area']>=103)] = '1'
trian_sample['toilet'].loc[(trian_sample['livingroom'] ==3)&(trian_sample['area']>=270)] = '3'
trian_sample['toilet'].loc[(trian_sample['livingroom'] ==4)&(trian_sample['area']>=300)] = '3'


# 借助API来计算房源地距离中心点的驾车距离和耗时数据
# 总结一下，这里会新增3个数据，距离地铁站距离，距离市中心驾车距离、距离市中心驾车耗时
# 先重排dataframe的index，这步很重要，不然会循环出错，因为前面有剔除样本操作，index不是连续的，for循环的时候要报错了
# 注意这可能是一个经常遇到的坑！！！！！
trian_sample.reset_index(drop = True, inplace = True)

# 调用百度地图借口返回地铁站相关数据
# 设置函数，方便调用数据
import requests
import re
def baidumap_distance(x):
    """x：纬度和经度，例如[129.3131,30.11]，类型：list"""
    distance = []
    ak = '注意：这里请填入你在百度地图官网申请的ak！！！'
    url = 'http://api.map.baidu.com/place/v2/search?query=地铁&location='
    other = '&radius=100000&scope=2&output=json&ak='
    a1 = re.findall(r' (\S+?)\]',x)
    a2 = re.findall(r'\[(\S+?)\,',x)
    url_ = url + a1[0] + ',' + a2[0] + other + ak
    headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'accept-encoding': 'gzip, deflate',
            'accept-language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'cache-control': 'no-cache',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:67.0) Gecko/20100101 Firefox/67.0',
            'host': 'api.map.baidu.com'}
    f = requests.get(url_, headers= headers)
    f = f.json()
    f = f['results']
    if f:       #这里对f是否为空做判断，如果是，填充99999；如果否，提取各个地铁站点的最小距离
        for i in range(len(f)):
            distance.append(f[i]['detail_info']['distance'])
        return(min(distance))
    else:
        return(99999)
print(u'函数构建完成！')

# 对用百度接口数据
# trian_sample['dist'] = trian_sample['location'].apply(baidumap_distance)
# 使用列表表达式最难受的是不知道进度情况，这里改成循环的方式
print(u'开始调用百度接口')
from tqdm import tqdm
dist = []
for i in tqdm(range(trian_sample.shape[0])):
    p = baidumap_distance(trian_sample['location'][i])
    dist.append(p)
print(u'百度接口调取完毕！')
# 合并入样本中
trian_sample['dist'] = dist

# 杭州大厦-武林广场作为杭州比较繁华的地带，因此我们提取房屋距离该地的距离和通勤时间数据
def dist_time_features(a,b = '120.167795,30.278638'):
    """a：str，代表起始位置坐标串
       b：str，代表终点位置坐标串，默认地点设置为杭州大厦经纬度坐标
    """
    a1 = re.findall(r' (\S+?)\]', a)
    a2 = re.findall(r'\[(\S+?)\,', a)
    b1 = b.split(',')[1]
    b2 = b.split(',')[0]
    headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'accept-encoding': 'gzip, deflate',
            'accept-language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'cache-control': 'no-cache',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:67.0) Gecko/20100101 Firefox/67.0',
            'host': 'api.map.baidu.com'}
    ak = '注意：这里请填入你在百度地图官网申请的ak！！！'
    url = 'http://api.map.baidu.com/routematrix/v2/driving?output=json&tactics=11&origins='
    m1 = '&destinations='
    m2 = '&ak='
    url_ = url + a1[0] + ',' + a2[0] + m1 + b1 + ',' + b2 + m2 + ak
    f = requests.get(url_, headers= headers)
    f = f.json()
    if f['result']:
        f1 = f['result'][0]['distance']['value']
        f2 = f['result'][0]['duration']['value']
    else:
        f1 = 99999
        f2 = 99999
    return([f1,f2])

print(u'开始调用百度接口')
from tqdm import tqdm
dist_dura = []
for i in tqdm(range(trian_sample.shape[0])):
    p = dist_time_features(trian_sample['location'][i])
    dist_dura.append(p)
print(u'百度接口调取完毕！')

# 将数据合并入trian_sample
print(dist_dura[0:5])
h1 = list(map(lambda x:x[0],dist_dura))
h2 = list(map(lambda x:x[1],dist_dura))
trian_sample['dist_hzds'] = h1
trian_sample['time_hzds'] = h2

# 在建模之前，利用均方误差来评价模型拟合预测的结果，先构建一个评价函数
# RMSE代表的是预测值和真实值差值的样本标准差。和MAE相比，RMSE对大误差样本有更大的惩罚。
#  不过RMSE有一个缺点就是对离群点敏感，这样会导致RMSE结果非常大。
# 这里用RMLSE(Root Mean Squared Logarithmic Error，RMSLE)
# RMSLE之余RMSE的变化在于它对y_actual和y_pred分别求取了对数:log(y_pred+1),log(y_actual+1)
def RMLSE(y_actual, y_pred):
    """ y_actual：代表实际的y值，array；
        y_pred：代表预测的y值，array"""
    r = (np.log(y_actual+1) - np.log(y_pred+1))**2
    r = r.sum()
    r = r/len(y_actual)
    r = r**0.5
    return(r)


#trian_sample = trian_sample.drop(['location'],axis =1).astype(float)
x = trian_sample.drop(['price'],axis =1).values
y = trian_sample.price.values
train_x, test_x, train_y, test_y = train_test_split(x, y,random_state=42, test_size=0.1, shuffle=True)

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
# 引入模型
# xgb
xgb = XGBRegressor(max_depth =10, learning_rate=0.1, n_estimators=50,
                   booster='gbtree')
xgb.fit(train_x, train_y)
print("XGBRegressor RMLSE is %f" % RMLSE(test_y,xgb.predict(test_x)))
# randomforest
rf = RandomForestRegressor(n_estimators = 50, max_depth = 10, random_state = 100)
rf.fit(train_x, train_y)
print("RandomForestRegressor RMLSE is %f" % RMLSE(test_y,rf.predict(test_x)))
# gbrt
gbrt = GradientBoostingRegressor(n_estimators = 50, learning_rate=0.1, max_depth = 10, random_state = 100)
gbrt.fit(train_x, train_y)
print("GradientBoostingRegressor RMLSE is %f" % RMLSE(test_y,gbrt.predict(test_x)))
# stacking方法
# 这里是用最简单的stacking方法，将多模型预测结果平均
algori_list = [xgb,rf,gbrt]
test_y_pred = np.column_stack([algori.predict(test_x) for algori in algori_list])
test_y_pred = test_y_pred.mean(axis =1)
print("Stacking model(xgb|rf|gbrt) rmlse is %f" % RMLSE(test_y,test_y_pred))


train_t.columns = ['客厅数','建筑楼层数','卫生间数','房屋所在楼层','建筑年份','装修级别','卧室数量','面积',
                   '所在区从业人数','所在区gdp','周边是否有地铁','价格','所在板块房屋均价','所在板块房屋数量',
                   '朝向_1','朝向_2','朝向_3','朝向_4','朝向_5','距离地铁站最近距离','距离杭州大厦距离','距离杭州大厦时间']

# 此时我们做一个相关性热力图
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
corr = train_t.corr()
f, ax= plt.subplots(figsize = (10, 6))
sns.heatmap(corr,cmap='RdBu', linewidths = 0.05, ax = ax)


# 特征在3个模型中的重要性得分
feature_name = ['客厅数','建筑楼层数','卫生间数','房屋所在楼层','建筑年份','装修级别','卧室数量','面积',
                   '所在区从业人数','所在区gdp','周边是否有地铁','所在板块房屋均价','所在板块房屋数量',
                   '朝向_1','朝向_2','朝向_3','朝向_4','朝向_5','距离地铁站最近距离','距离杭州大厦距离','距离杭州大厦时间']
rf = xgb.feature_importances_
xgb = xgb.feature_importances_
gbrt = gbrt.feature_importances_
feature_imp = pd.DataFrame({"feature":feature_name,"rf":rf,"xgb":xgb,"gbrt":gbrt})
print(feature_imp)
