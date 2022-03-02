Alimama 提供的，主要有三个文件：

### user_profile
覆盖了 `raw_sample`里的106万用户的基本信息，主要有以下字段：

```
userid

micro group id

final_gender_code: 1 for male, 2 for female

age_level

p_value_level: consumption_grade, 1 low, 2, mid, 3 high

shopping_level: shopping depath, 1: shallow user, 2: moderate user, 3: depth user

occupation: is the college student 

new_user

city_level:


```
### ad_feature

```
adgroup_id

cate_id

campaign_id

brand

customer_id

```
这个广告里信息好少，应该还有创意相关的东西: creative_id

### raw_sample
```
user

timestamp

pid

noclk
```

可见就是三张表，用户自身的画像，广告自身，以及用户浏览记录、点击记录
