import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
#Burada ihtiyacmız olan ve önceden indirmiş olduğumuz kütüphaneleri "import" diyerek
#kodumuza çağırmış oluyoruz

veriSeti = pd.read_csv('Salary_Data.csv')
#Bu satırı yazmadan önce spyderdan çalışma konumunu .csv dosyamızın olduğu yer olarak seçiyoruz
#Salary_Data.csv yi koyacak yeni bir dosya açabilirsiniz ve sağ üst köşeden o dosya yolunu seçebilirsiniz
#Bu satırda yaptığımız şey Salary_Data.csv adlı odsyamızı pandas(pd) ile okuyup veriSeti adlı değişkene atamaktır

X = veriSeti.iloc[:, :-1]
Y = veriSeti.iloc[:, 1]

#Şimdi artık veri setimiz de kodumuzda olduğuna göre veri setimizin "Yıl" ve "Maaş" Sütunlarını ayırabiliriz
#ki hepsinin üstünde ayrı ayrı çalışalım.

#veriSeti.iloc[] yaparak matrisimizi yani csv mizi keseceğiz
#[: , :-1] köşeli parantezde ilk parametremiz satır ve ikinci parametremizde sütun.
#[:, :-1] ilk parametrede sadece ":" yazarak tüm satırları alıyoruz demiş olduk
#ikinci parametredeki ":-1" ise 2. parametre sütunlar olduğu için sondan birinci sütun hariç tüm sütunları al demek
#Yani X imiz yıl verisi olacak
#[:,1] deki 1 ise sadece 1. indexli sütunu al demek. Yani maaş sütunumuz

from sklearn.model_selection import train_test_split as tts
X_train,X_test,Y_train,Y_test = tts(X,Y,test_size = 1/3,random_state = 0)
#şimdi scikitlearn de bulunan test ve train ayırım fonksiyonunu kullanacağız 
#şimdi datamızı eğitim ve test verisi olarak ikiye ayıracağız. 30 verimiz var 10u test olsun 20si ilede eğitelim
#X_train,X_test... bunları train test splite veriyoruz. tts ilk parametre olarak dizileri alır bizim dizilerimiz de X VE Y
#test boyutu dediğimiz gibi 30da 10 yani 1/3, random_state ise modelimizde değerlerde rasgtgelelik olup olmaması biz rastgeleye göre değerlerin 
#değişmesini istemiyoruz o yüzden random_state = 0

from sklearn.linear_model import LinearRegression as lr
regressor = lr()
regressor.fit(X_train,Y_train)
#scikitlearn içiden ve linear_model den LinearRegressionu çağırıyoruz
#ve bunu regressor adlı bir değişkene atıyoruz. yani artık regressor lineer regresyon özelliği taşıyor.
#ondan sonrasında lineer regresyonu kendi datamıza fit ediyoruz. Yani X_train datası ve Y_train datası arasında lineer regresyon ile bir koreelasyon bulmasını sağlıyoruz


Y_pred = regressor.predict(X_test)
#Sonrasında burada X_test üzerinden tahmin yürütmesini sağlıyoruz. X_test çünkü ayırdığımız yıl verisi üzerinde ne kadar maaş tahmini yaptığını göreceğiz

#artık modelimizi eğittik ve Variable explorer (sağda konsolun üstünde) dan da gördüğümüz gibi tahmin verilerimiz aldık ve gayet iyi bir oran

plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = "blue")
plt.title("Yıla Göre Maaş")
plt.xlabel('Tecrübe Yılı')
plt.ylabel('Maaş')
#Buradada verimizi görselleştiriyoruz. matplotlib kullanacağımız kütüphane. Scatter = dağılım grafiği kullanacağız.
#dağılım grafiğinin alcağı veriler yıl ve maaş yani X_train ve Y_train.nokta rengimizide kırmızı yaptık
#plotlada regresyon çizgimizi yani tahmin çizgimizi grafik üzerinde çizeceğiz.
#plotumuz trainden yaptığımız predict ile olacak.


plt.scatter(X_test,Y_test, color ="red")
plt.plot(X_train,regressor.predict(X_train) , color = "blue")
plt.title("Yıla göre maaş (Test set)")
plt.xlabel("Tecrübe yılı")
plt.ylabel("Maaş")
plt.show

#Buradada test seti üzerinden deniyoruz grafiği. Yani yeni gelen verileri ne kadar doğru tahmin edebiliyor görüyoruz.






