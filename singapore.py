import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from scipy import stats

resale1990_df=pd.read_csv("singapore_resale_price_predict\ResaleFlatPricesBasedonApprovalDate19901999.csv")
resale2000_df=pd.read_csv("singapore_resale_price_predict\ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv")
resale2012_df=pd.read_csv("singapore_resale_price_predict\ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv")
resale2015_df=pd.read_csv("singapore_resale_price_predict\ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv")
resale2017_df=pd.read_csv("singapore_resale_price_predict\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")

#resale1990-1999 data 
resale1990=pd.read_csv("singapore_resale_price_predict\ResaleFlatPricesBasedonApprovalDate19901999.csv")
#preprocessing
resale1990['year']=resale1990['month'].apply(lambda j: int(j.split('-')[0]))
resale1990.drop(['month','block','street_name'],axis=1,inplace=True)
resale1990['remaining_lease_year']=99-(resale1990['year']-resale1990['lease_commence_date'])
encoded=LabelEncoder()
for i in resale1990.select_dtypes(include=['object']).columns:
    resale1990[i]=encoded.fit_transform(resale1990[i])

# building model
x=resale1990.drop('resale_price',axis=1)
y=resale1990['resale_price']

# scaler=RobustScaler()
# for i in x.columns:
#     x[i]=scaler.fit_transform(x[[i]])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
# y_train_transformed, _ = stats.boxcox(y_train + 1)  
# y_test_transformed = stats.boxcox(y_test + 1)[0]
model=LinearRegression().fit(x_train,y_train)

#resale2000-2012 data
resale2000=pd.read_csv("singapore_resale_price_predict\ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv")
resale2000['year']=resale2000['month'].apply(lambda j: int(j.split('-')[0]))
resale2000.drop(['month','block','street_name'],axis=1,inplace=True)
resale2000['remaining_lease_year']=99-(resale2000['year']-resale2000['lease_commence_date'])

encoded1=LabelEncoder()
for i in resale2000.select_dtypes(include=['object']).columns:
    resale2000[i]=encoded1.fit_transform(resale2000[i])

x1=resale2000.drop('resale_price',axis=1)
y1=resale2000['resale_price']

# scaler1=RobustScaler()
# for i in x1.columns:
#     x1[i]=scaler1.fit_transform(x1[[i]])

x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2)

# y1_train_transformed, _ = stats.boxcox(y1_train + 1) 
# y1_test_transformed = stats.boxcox(y1_test + 1)[0]

model1=LinearRegression().fit(x1_train,y1_train)

#resale2012-2015 data
resale2012=pd.read_csv("singapore_resale_price_predict\ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv")
resale2012['year']=resale2012['month'].apply(lambda j: int(j.split('-')[0]))
resale2012.drop(['month','block','street_name'],axis=1,inplace=True)
resale2012['remaining_lease_year']=99-(resale2012['year']-resale2012['lease_commence_date'])

encoded2=LabelEncoder()
for i in resale2012.select_dtypes(include=['object']).columns:
    resale2012[i]=encoded.fit_transform(resale2012[i])

x2=resale2012.drop('resale_price',axis=1)
y2=resale2012['resale_price']

# scaler2=RobustScaler()
# for i in x2.columns:
#     x2[i]=scaler2.fit_transform(x2[[i]])

x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2)

# y2_train_transformed, _ = stats.boxcox(y2_train + 1)  
# y2_test_transformed = stats.boxcox(y2_test + 1)[0]

model2=LinearRegression().fit(x2_train,y2_train)

#resale2015-2016 data

resale2015=pd.read_csv("singapore_resale_price_predict\ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv")
resale2015['year']=resale2015['month'].apply(lambda j: int(j.split('-')[0]))
resale2015.drop(['month','block','street_name'],axis=1,inplace=True)
encoded3=LabelEncoder()
for i in resale2015.select_dtypes(include=['object']).columns:
    resale2015[i]=encoded3.fit_transform(resale2015[i])

x3=resale2015.drop('resale_price',axis=1)
y3=resale2015['resale_price']

# scaler3=RobustScaler()
# for i in x3.columns:
#     x3[i]=scaler3.fit_transform(x3[[i]])

x3_train,x3_test,y3_train,y3_test=train_test_split(x3,y3,test_size=0.2)

# y3_train_transformed, _ = stats.boxcox(y3_train + 1) 
# y3_test_transformed = stats.boxcox(y3_test + 1)[0]

model3=LinearRegression().fit(x3_train,y3_train)

#resale2017-present data

resale2017=pd.read_csv("singapore_resale_price_predict\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")
resale2017['remaining_lease']=resale2017['remaining_lease'].apply(lambda j:int(j.split(' years')[0]))
resale2017['year']=resale2017['month'].apply(lambda j: int(j.split('-')[0]))
resale2017.drop(['month','block','street_name'],axis=1,inplace=True)
encoded=LabelEncoder()
for i in resale2017.select_dtypes(include=['object']).columns:
    resale2017[i]=encoded.fit_transform(resale2017[i])

x4=resale2017.drop('resale_price',axis=1)
y4=resale2017['resale_price']

# scaler4=RobustScaler()
# for i in x4.columns:
#     x4[i]=scaler4.fit_transform(x4[[i]])

x4_train,x4_test,y4_train,y4_test=train_test_split(x4,y4,test_size=0.2)

# y4_train_transformed, _ = stats.boxcox(y4_train + 1)  
# y4_test_transformed = stats.boxcox(y4_test + 1)[0]

model4=LinearRegression().fit(x4_train,y4_train)

def resale1990_pred(town,flat_type,storey_range,floor_area_sqm,flat_model,lease_commence_date,year):
    town_type={
        'ANG MO KIO':0, 
        'BEDOK':1, 
        'BISHAN':2, 'BUKIT BATOK':3, 'BUKIT MERAH':4, 'BUKIT PANJANG':5,
       'BUKIT TIMAH':6, 'CENTRAL AREA':7, 'CHOA CHU KANG':8, 'CLEMENTI':9,
       'GEYLANG':10, 'HOUGANG':11, 'JURONG EAST':12, 'JURONG WEST':13,
       'KALLANG/WHAMPOA':14, 'MARINE PARADE':15, 'QUEENSTOWN':16, 'SENGKANG':17,
       'SERANGOON':18, 'TAMPINES':19, 'TOA PAYOH':20, 'WOODLANDS':21, 'YISHUN':22,
       'LIM CHU KANG':23, 'SEMBAWANG':24, 'PASIR RIS':25
    }
    Town=town_type.get(town)

    flat_type_names={
        '1 ROOM':0, '2 ROOM':1,'3 ROOM':2, '4 ROOM':3, '5 ROOM':4,  'EXECUTIVE':5,
       'MULTI GENERATION':6

    }

    flatType=flat_type_names.get(flat_type)

    storey_range_types={
         '01 TO 03':0, '04 TO 06':1, '07 TO 09':2,'10 TO 12':3, '13 TO 15':4, '16 TO 18':5,
       '19 TO 21':6, '22 TO 24':7, '25 TO 27':8
    }

    storeyRange=storey_range_types.get(storey_range)

    flat_model_names={
        '2-ROOM':0, 'APARTMENT':1,'IMPROVED':2,'IMPROVED-MAISONETTE':3,'MAISONETTE':4,'MODEL A':5,'MODEL A-MAISONETTE':6, 'MULTI GENERATION':7,
        'NEW GENERATION':8,'PREMIUM APARTMENT':9,'SIMPLIFIED':10,'STANDARD':11, 'TERRACE':12
    }

    flatModel=flat_model_names.get(flat_model)
    remaining_lease_year=99-(year - lease_commence_date)
    

    price=np.array([Town,flatType,storeyRange,floor_area_sqm,flatModel,lease_commence_date,year,remaining_lease_year]).reshape(1, -1)
    # x_sample=pd.DataFrame(price, columns=['town','flat_type','storey_range','floor_area_sqm','flat_model','lease_commence_date','year','remaining_lease_year'])
    # scaler=RobustScaler()
    # for i in x_sample.columns:
    #     x_sample[i]=scaler.fit_transform(x_sample[[i]]) 
    price_pred=model.predict(price)

    return price_pred

def resale2000_pred(town,flat_type,storey_range,floor_area_sqm,flat_model,lease_commence_date,year):
    town_type={
        'ANG MO KIO':0, 'BEDOK':1, 'BISHAN':2, 'BUKIT BATOK':3, 'BUKIT MERAH':4,
       'BUKIT PANJANG':5, 'BUKIT TIMAH':6, 'CENTRAL AREA':7, 'CHOA CHU KANG':8,
       'CLEMENTI':9, 'GEYLANG':10, 'HOUGANG':11, 'JURONG EAST':12, 'JURONG WEST':13,
       'KALLANG/WHAMPOA':14, 'MARINE PARADE':15, 'PASIR RIS':16, 'PUNGGOL':17, 'QUEENSTOWN':18, 'SEMBAWANG':19,
       'SENGKANG':20, 'SERANGOON':21, 'TAMPINES':22, 'TOA PAYOH':23, 'WOODLANDS':24,
       'YISHUN':25
    }
    town=town_type.get(town)

    flat_type_names={
        '1 ROOM':0,'2 ROOM':1,'3 ROOM':2, '4 ROOM':3, '5 ROOM':4, 'EXECUTIVE':5, 
       'MULTI-GENERATION':6

    }

    flat_type=flat_type_names.get(flat_type)

    storey_range_types={
         '01 TO 03':0,'04 TO 06':1,'07 TO 09':2, '10 TO 12':3, '13 TO 15':4,
       '16 TO 18':5, '19 TO 21':6, '22 TO 24':7, '25 TO 27':8, '28 TO 30':9,
       '31 TO 33':10, '34 TO 36':11, '37 TO 39':12,'40 TO 42':13 
    }

    storey_range=storey_range_types.get(storey_range)

    flat_model_names={
        '2-room':0,'Adjoined flat':1,'Apartment':2,'Improved':3,'Improved-Maisonette':4, 'Maisonette':5,'Model A':6,
        'Model A-Maisonette':7,'Model A2':8,'Multi Generation':9,
        'New Generation':10, 'Premium Apartment':11, 'Premium Maisonette':12, 'Simplified':13,'Standard':14, 'Terrace':15
    }

    flat_model=flat_model_names.get(flat_model)

    remaining_lease_year=99-(year-lease_commence_date)
    price=np.array([town,flat_type,storey_range,floor_area_sqm,flat_model,lease_commence_date,year,remaining_lease_year]).reshape(1, -1)
    price_pred=model1.predict(price)

    return price_pred

def resale2012_pred(town,flat_type,storey_range,floor_area_sqm,flat_model,lease_commence_date,year):
    town_type={
        'ANG MO KIO':0, 'BEDOK':1, 'BISHAN':2, 'BUKIT BATOK':3, 'BUKIT MERAH':4,
       'BUKIT PANJANG':5, 'BUKIT TIMAH':6, 'CENTRAL AREA':7, 'CHOA CHU KANG':8,
       'CLEMENTI':9, 'GEYLANG':10, 'HOUGANG':11, 'JURONG EAST':12, 'JURONG WEST':13,
       'KALLANG/WHAMPOA':14, 'MARINE PARADE':15, 'PASIR RIS':16, 'PUNGGOL':17,
       'QUEENSTOWN':18, 'SEMBAWANG':19, 'SENGKANG':20, 'SERANGOON':21, 'TAMPINES':22,
       'TOA PAYOH':23, 'WOODLANDS':24, 'YISHUN':25
    }
    town=town_type.get(town)

    flat_type_names={
        '1 ROOM':0,'2 ROOM':1,'3 ROOM':2, '4 ROOM':3, '5 ROOM':4, 'EXECUTIVE':5, 
       'MULTI-GENERATION':6

    }

    flat_type=flat_type_names.get(flat_type)

    storey_range_types={
        '01 TO 03':0, '01 TO 05':1,  '04 TO 06':2,'06 TO 10':3,'07 TO 09':4,'10 TO 12':5,'11 TO 15':6,'13 TO 15':7, 
        '16 TO 18':8,'16 TO 20':9,'19 TO 21':10, '21 TO 25':11,'22 TO 24':12,
        '25 TO 27':13,'26 TO 30':14,'28 TO 30':15,'31 TO 33':16,'31 TO 35':17,'34 TO 36':18,'36 TO 40':19,'37 TO 39':20,'40 TO 42':21 
    }

    storey_range=storey_range_types.get(storey_range)

    flat_model_names={
        'Adjoined flat':0,'Apartment':1,'DBSS':2,'Improved':3,'Improved-Maisonette':4, 'Maisonette':5,'Model A':6,
        'Model A-Maisonette':7,'Model A2':8,'Multi Generation':9, 
        'New Generation':10, 'Premium Apartment':11,'Premium Maisonette':12,'Simplified':13,'Standard':14, 'Terrace':15,'Type S1':16
    }

    flat_model=flat_model_names.get(flat_model)

    remaining_lease_year=99-(year-lease_commence_date)
    price=np.array([town,flat_type,storey_range,floor_area_sqm,flat_model,lease_commence_date,year,remaining_lease_year]).reshape(1, -1)
    price_pred=model2.predict(price)

    return price_pred

def resale2015_pred(town,flat_type,storey_range,floor_area_sqm,flat_model,lease_commence_date,year):
    town_type={
        'ANG MO KIO':0, 'BEDOK':1, 'BISHAN':2, 'BUKIT BATOK':3, 'BUKIT MERAH':4,
       'BUKIT PANJANG':5, 'BUKIT TIMAH':6, 'CENTRAL AREA':7, 'CHOA CHU KANG':8,
       'CLEMENTI':9, 'GEYLANG':10, 'HOUGANG':11, 'JURONG EAST':12, 'JURONG WEST':13,
       'KALLANG/WHAMPOA':14, 'MARINE PARADE':15, 'PASIR RIS':16, 'PUNGGOL':17,
       'QUEENSTOWN':18, 'SEMBAWANG':19, 'SENGKANG':20, 'SERANGOON':21, 'TAMPINES':22,
       'TOA PAYOH':23, 'WOODLANDS':24, 'YISHUN':25
    }
    town=town_type.get(town)

    flat_type_names={
        '1 ROOM':0,'2 ROOM':1,'3 ROOM':2, '4 ROOM':3, '5 ROOM':4, 'EXECUTIVE':5, 
       'MULTI-GENERATION':6

    }

    flat_type=flat_type_names.get(flat_type)

    storey_range_types={
        '01 TO 03':0,'04 TO 06':1,'07 TO 09':2,'10 TO 12':3,  '13 TO 15':4, '16 TO 18':5, 
       '19 TO 21':6,  '22 TO 24':7, '25 TO 27':8, '28 TO 30':9,'31 TO 33':10,
       '34 TO 36':11,'37 TO 39':12, '40 TO 42':13, '43 TO 45':14,'46 TO 48':15, '49 TO 51':16
    }

    storey_range=storey_range_types.get(storey_range)

    flat_model_names={
        '2-room':0,'Adjoined flat':1,'Apartment':2,'DBSS':3,'Improved':4,'Improved-Maisonette':5, 'Maisonette':6,'Model A':7,'Model A2':8,
        'Model A-Maisonette':9,'Multi Generation':10,'New Generation':11,'Premium Apartment':12,'Premium Apartment Loft':13,
        'Premium Maisonette':14, 'Simplified':15, 'Standard':16,'Terrace':17,'Type S1':18, 'Type S2':19 
    }

    flat_model=flat_model_names.get(flat_model)

    remaining_lease_year=99-(year-lease_commence_date)
    price=np.array([town,flat_type,storey_range,floor_area_sqm,flat_model,lease_commence_date,year,remaining_lease_year]).reshape(1, -1)
    price_pred=model3.predict(price)

    return price_pred

def resale2017_pred(town,flat_type,storey_range,floor_area_sqm,flat_model,lease_commence_date,year):
    town_type={
        'ANG MO KIO':0, 'BEDOK':1, 'BISHAN':2, 'BUKIT BATOK':3, 'BUKIT MERAH':4,
       'BUKIT PANJANG':5, 'BUKIT TIMAH':6, 'CENTRAL AREA':7, 'CHOA CHU KANG':8,
       'CLEMENTI':9, 'GEYLANG':10, 'HOUGANG':11, 'JURONG EAST':12, 'JURONG WEST':13,
       'KALLANG/WHAMPOA':14, 'MARINE PARADE':15, 'PASIR RIS':16, 'PUNGGOL':17,
       'QUEENSTOWN':18, 'SEMBAWANG':19, 'SENGKANG':20, 'SERANGOON':21, 'TAMPINES':22,
       'TOA PAYOH':23, 'WOODLANDS':24, 'YISHUN':25
    }
    town=town_type.get(town)

    flat_type_names={
        '1 ROOM':0,'2 ROOM':1,'3 ROOM':2, '4 ROOM':3, '5 ROOM':4, 'EXECUTIVE':5, 
       'MULTI-GENERATION':6

    }

    flat_type=flat_type_names.get(flat_type)

    storey_range_types={
        '01 TO 03':0,'04 TO 06':1,'07 TO 09':2,'10 TO 12':3,  '13 TO 15':4, '16 TO 18':5, 
       '19 TO 21':6,  '22 TO 24':7, '25 TO 27':8, '28 TO 30':9,'31 TO 33':10,
       '34 TO 36':11,'37 TO 39':12, '40 TO 42':13, '43 TO 45':14,'46 TO 48':15, '49 TO 51':16
    }

    storey_range=storey_range_types.get(storey_range)

    flat_model_names={
        '2-room':0,'3Gen':1,'Adjoined flat':2,'Apartment':3,'DBSS':4,'Improved':5,'Improved-Maisonette':6,'Maisonette':7,
        'Model A':8, 'Model A-Maisonette':9,'Model A2':10,'Multi Generation':11,'New Generation':12, 'Premium Apartment':13,
        'Premium Apartment Loft':14,'Premium Maisonette':15,'Simplified':16, 'Standard':17,'Terrace':18,'Type S1':19, 'Type S2':20 
    }

    flat_model=flat_model_names.get(flat_model)

    remaining_lease_year=99-(year-lease_commence_date)
    price=np.array([town,flat_type,storey_range,floor_area_sqm,flat_model,lease_commence_date,year,remaining_lease_year]).reshape(1, -1)
    price_pred=model4.predict(price)

    return price_pred






    

