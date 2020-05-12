import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import bartlett
from scipy.stats import levene
from statsmodels.tsa.stattools import acf
from scipy.stats import jarque_bera
from scipy.stats import normaltest
import warnings
warnings.filterwarnings('ignore')

weather_df = pd.read_csv("weatherHistory.csv")

#####################################################################################################
#Varsayım 2: Hata terimi ortalamada sıfır olmalıdır

# Y = weather_df['Sicaklik']
# X = weather_df[['Nem','RuzgarHizi',"Ruzgar","Basinc"]]
#
# lrm = linear_model.LinearRegression()
# lrm.fit(X, Y)
#
# print('Değişkenler: \n', lrm.coef_)
# print('Sabit değer (bias): \n', lrm.intercept_)
#
# tahmin = lrm.predict(X)
# hatalar = Y - tahmin
# print("Ortalama hata : {:.15f}".format(np.mean(hatalar)))
# print("Hatalarımız 0dır. Hedefimiz doğru yolda.")
#
# ##############################################################################
# #Varsayım 3: homoscedasticity
#
# plt.figure(figsize=(9,6), dpi=90)
# plt.scatter(tahmin, hatalar)
# plt.xlabel('Tahmin Edilen')
# plt.ylabel('Artık (Residual)')
# plt.axhline(y=0)
# plt.title('Artık x Tahmin')
# plt.show()
#
# bart_stats = bartlett(tahmin, hatalar)
# lev_stats = levene(tahmin, hatalar)
#
# print("Bartlett test değeri : {0:3g} ve p değeri : {1:.21f}".format(bart_stats[0], bart_stats[1]))
# print("Levene test değeri   : {0:3g} ve p değeri : {1:.21f}".format(lev_stats[0], lev_stats[1]))
# print("p değerlerimiz 0.05ten küçük olduğu için hatalarımız heteroscedastiktir. \n ")
#
# ######################################################################################
# #Varsayım 4: düşük çoklu doğrusallık/low multicollinearity
#
# df = weather_df[['Nem','RuzgarHizi',"Ruzgar","Basinc"]]
#
# print(df.corr())
# print("Değerlerimiz -1 ile 1 arasında olduğu için *mükemmel çoklu doğrusallıktalar* \n")
#
# #########################################################################################
# #Varsayım 5: hata terimleri birbiriyle ilişkisiz olmalıdır
# #
# plt.figure(figsize=(9,6))
# plt.plot(hatalar)
# plt.show()
#
# acf_data = acf(hatalar)
#
# plt.figure(figsize=(9,6))
# plt.plot(acf_data[1:])
# plt.show()
#
# #otokorelasyon 0.68 ile 0.98 arasında değişiyor. Bootcampte -0,06 ile 0,05 arasına çok az deniyor.
# #Bizimkisi hala az mı? çok olması için ne kadar gerekiyor?
#
#
# ########################################################################################
# #Varsayım 6: özellikler hatalarla korele olmamalıdır
#
# jb_stats = jarque_bera(hatalar)
# norm_stats = normaltest(hatalar)
#
# print("Jarque-Bera test değeri : {0} ve p değeri : {1}".format(jb_stats[0], jb_stats[1]))
# print("Normal test değeri      : {0}  ve p değeri : {1:.30f}".format(norm_stats[0], norm_stats[1]))
#
# print("P değerlerimiz 0.05ten küçük çıktığı için hatalarımız normal dağılmamıştır")


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#PART2 --- Ev satışları bölümü

# house = pd.read_csv("train.csv")
#
# house['yeni_mi'] = np.where(house['YearBuilt']>=2005, 1, 0)
#
# Y = house["SalePrice"]
# X = house[["yeni_mi","FullBath","GarageCars","WoodDeckSF","OverallQual","LotArea"]]
#
# lrm = linear_model.LinearRegression()
# lrm.fit(X, Y)
#
# print('Değişkenler: \n', lrm.coef_)
# print('Sabit değer (bias): \n', lrm.intercept_)
#
# tahmin = lrm.predict(X)
# hatalar = Y - tahmin
#
# print("Modelin ortalama hatası : {:.15f}".format(np.mean(hatalar)))
# print("Hatamız 0'dır. ")

# #########################################################################################################
# # #Varsayım 3: homoscedasticity
#
# plt.figure(figsize=(9,6), dpi=90)
# plt.scatter(tahmin, hatalar)
# plt.xlabel('Tahmin Edilen')
# plt.ylabel('Artık (Residual)')
# plt.axhline(y=0)
# plt.title('Artık x Tahmin')
# plt.show()
#
# bart_stats = bartlett(tahmin, hatalar)
# lev_stats = levene(tahmin, hatalar)
#
# print("\n\nBartlett test değeri : {0:3g} ve p değeri : {1:.21f}".format(bart_stats[0], bart_stats[1]))
# print("Levene test değeri   : {0:3g} ve p değeri : {1:.21f}".format(lev_stats[0], lev_stats[1]))
# print("p değerlerimiz 0.05ten küçük olduğu için hatalarımız heteroscedastiktir. \n ")
#
#
# # ######################################################################################
# # #Varsayım 4: düşük çoklu doğrusallık/low multicollinearity
#
# df = house[["yeni_mi","FullBath","GarageCars","WoodDeckSF","OverallQual","LotArea"]]
#
# print(df.corr())
# print("Değerlerimiz -1 ile 1 arasında olduğu için *mükemmel çoklu doğrusallıktalar* \n")
#
# # #########################################################################################
# # #Varsayım 5: hata terimleri birbiriyle ilişkisiz olmalıdır
# # #
# plt.figure(figsize=(9,6))
# plt.plot(hatalar)
# plt.show()
#
# acf_data = acf(hatalar)
#
# plt.figure(figsize=(9,6))
# plt.plot(acf_data[1:])
# plt.show()
#
# print("Sonuçlar istediğimiz değerler arasındadır.")
#
# # ########################################################################################
# #Varsayım 6: özellikler hatalarla korele olmamalıdır
#
# jb_stats = jarque_bera(hatalar)
# norm_stats = normaltest(hatalar)
#
# print("Jarque-Bera test değeri : {0} ve p değeri : {1}".format(jb_stats[0], jb_stats[1]))
# print("Normal test değeri      : {0}  ve p değeri : {1:.30f}".format(norm_stats[0], norm_stats[1]))
#
# print("P değerlerimiz 0.05ten küçük çıktığı için hatalarımız normal dağılmamıştır")




