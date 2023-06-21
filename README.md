# Real Time Facial Emotion Detection 
# Truth From Facial Expressions With Deep Learning Program That Detects Timely Emotion Analysis And Design
<br>
**Proje 2**      **Meryem Özlem AYDOĞAN**
<br>
~Rapor içeriğinden:
<br>
**1.2 Dünyadaki ve Ülkemizdeki Benzer Örnekler**
<br>
Gerçek zamanlı duygu tespiti, son yıllarda yapay zeka ve görüntü işleme alanında büyük ilgi 
görmüştür. Bu sebeple dünyada ve ülkemizde birçok benzer örneği vardır. Araştırma sonucu 
ulaşılabilen benzer nitelikteki projeleri incelediğinde, gerçek zamanlı evrişimli sinir ağları 
mimarisi kullanılarak, yüz tanıma, cinsiyet sınıflandırması ve duygu sınıflandırması tek bir 
adımda eş zamanlı olarak gerçekleştirilen modeller farkındalık oluşturduğu saptanmıştır. Yüz 
ifadeleri üzerinden yaş tahmini yapan uygulamalar geliştirilmiş ve bu uygulamalar için büyük 
veri setleri kullanılmıştır.
MIT’nin geliştirdiği “Affdex”, Microsoft’un "Azure Cognitive Services" ve "Face API’sı, 
Amazon “Rekognition”, “SkyBiometry”, Emotion AI" , “Face++”, “Anthropic” gibi 
platformların gerçek zamanlı duygu tespiti projeleri üzerinde çalıştığı bilinmektedir. Bu 
projeler, geniş veri kümelerini kullanarak duygu analizi modelleri oluşturarak duygusal 
tepkileri sınıflandırmak için derin öğrenme tekniklerini kullanmaktadır.
Literatürde duygu analizi alanında yapılan çalışmalarda en etkili sonuçlarının alındığı makine 
öğrenimi yöntemine Pang ve Lee öncülük etmiştir. En iyi başarı sonucuna %78.9 ulaşılmıştır. 
Pang ve Lee’nin yaptıkları projede, farklı olarak metin sınıflandırıcılığı işlenmiştir. Dandıl ve 
Özdemir, 2019 yılında yaptıkları çalışmada klasik evrişimsel sinir ağı AlexNet ile gerçek 
zamanlı video karelerinden yüz ifadeleri temelinde bir duygu tanıma sistemi önermiştir. 
Chen ve Ark, 2015 yılında yaptıkları çalışmada resim görüntülerinden duygu sınıflandırması
yapmak için kullandıkları otomatik öğrenen evrişimsel sinir ağı (ESA-CNN) ile elle 
hazırlanmış özelliklere dayanan geleneksel yöntemlerden daha iyi performans almışlardır.
Yapılan çalışmalar ESA’nın duygu tanımada özellik çıkarmak için kullanılabileceğini 
göstermektedir. Evrişim yapısı üzerindeki giriş katmanı, eğitilecek modelin verileri girilerek 
istenilen yapıya dönüştürülen katmandır. 
Duygu analiz yöntemleri, makine öğrenme tabanlı yöntemler ve sözlük tabanlı yöntemler olarak 
sınıflandırılmaktadır. Bu yöntemler içerisinde, literatürde yapılan çalışmalar göz önüne 
alındığında makine öğrenimi tabanlı yöntemlerin maksimum doğruluk verdiği görülmüştür. 
Yapılmış olunan bu projeyi diğer uygulamalardan ayıran bazı özellikler vardır. Detaylı 
araştırmalar sonucu elde edilmiş olan başarı oranlarının belirli bir sınır üzerine çıkamıyor oluşu
ve çoğu projenin bu doğruluk sınırına yaklaşamamaları dikkat çekici bulunmuştur. Bu soruna 
çözüm sunabilmek amacıyla model oluşturma kısmına ek katmanlar eklenmiş ve model eğitim 
için yaygın olarak kullanılan kombinasyonlar son araştırmalarda test edilerek belirli optimal 
eşikler belirlenmeye çalışılmıştır. Bunlar sonucunda ise değerlendirme ve başarı oranının
artması sağlanmıştır. Diğer projelere kıyasla ek olarak, anlık cevap değeri döndürebilen bir 
yazılım geliştirilmiş ve ifadelerin okunmasında kolaylık sağlanmıştır. Bu sayede model hızlı bir 
şekilde denenerek, yeniden yapılandırılmış ve geliştirilmesine olanak sunulmuştur.
<br>
<br>
**2. PROJE İÇERİĞİ VE KAPSAMI** 
<br>
Son zamanlarda yüz tanıma ve algılama sistemleri birçok ticari, askeri, güvenlik, sosyal ve 
psikolojik alanlardaki uygulamalarda sıkça kullanılmaktadır. Yapılan analizler; insan 
yüzlerinin hareketlerinin tanımlanmasını ve yorumlanmasını içermektedir. İnsanlar tarafından 
bile zor analiz edilebilen duygusal ifadeler bilgisayar ortamında test edilip belirlenmesinin 
kolaylık sağlayacağının düşünülmesi derin öğrenme alanına popülerlik kazandırmıştır. Bu 
bağlamda bilgisayarlı görü alanına da değinilmelidir. Bilgisayarlı görü, günümüzde yüz ve 
duygu sınıflandırma alanlarında yaygın olarak kullanılmaktadır. Yüz tanıma, görüntü veya 
videolardan elde edilen verilerdeki kişilerin otomatik olarak tanımlanması veya doğrulanması
işlemidir. Yüz tanıma işlemlerinin dört temel aşaması vardır. Bu işlemler sırasıyla yüz
algılama, normalleştirme, öznitelik çıkarma ve sınıflandırmadır. Normalleştirme ve 
sınıflandırma algoritmaları yüz tanımada ne kadar başarılı olursa olsun, eğer özellik çıkarma 
aşaması başarılı olmazsa o sistem istenilen başarıyı yakalayamamaktadır.
Gerçek zamanlı duygu tespit projesi, görüntü işleme ve yapay zeka algoritmalarını içeren bir 
dizi adımdan oluşmaktadır. İlk adım olarak, anlık görüntüdeki insan yüzlerini tespit etmek 
için yüz tanıma algoritmaları kullanılmıştır. Yüz veya yüzler tespit edildikten sonra, duygu 
durumunu belirlemek için evrişimsel sinir ağları kullanılarak geliştirilen duygu analiz modeli 
kullanılmıştır. Bu model, derin öğrenme teknikleriyle eğitilmiş ve farklı duygu kategorilerine 
ayrılan veri kümesiyle beslenmiştir. Son olarak, belirlenen duygu durumları, anlık analizler 
yapmak ve sonuçları kullanıcıya sunmak için kullanılmıştır.
Proje kapsamı boyunca yapılan eklemeler sonucu yenilenen modelin başarı oranı ve 
kullanılabilirliği doğru orantılı şekilde artış gösterdiği çizilen grafikler yardımıyla 
gözlenmiştir. Öznitelik çıkarımı için derin öğrenme tekniklerinden biri olan ve yapay sinir 
ağları içeren bir yaklaşım olan Evrişimli Sinir Ağları (ESA-CNN) kullanılarak yeni bir model 
geliştirilmiştir. 
Model eğitimi için yaygın olarak kullanılan kombinasyonlar son çalışmalarda test edilmiş ve 
sınıflandırma algoritmalarının gösterdikleri başarım sonuçlarına etkisi incelenmiştir. 
Değerlendirmeler yapılarak, güçlü performans gösteren sınıflandırma algoritması ve gerçek 
zamanlı evrişimli sinir ağları mimarisi kullanılarak, yüz görsellerinden duygu sınıflandırması 
işlemi eş zamanlı olarak gerçekleştiren proje ortaya çıkarılmıştır.
<br>
<br>

**3.2 Altyapı, Donanım ve Yazılım Özellikleri**
Derin öğrenmenin yüksek başarımı için kaliteli bir veri setine ihtiyaç duyulmaktadır. Bu 
sebeple, eğitim ve test performansının yüksek olarak ölçüldüğü veri setlerine bakılmalı ve 
uygun olan set seçilmelidir. Projede, face expression recognition veri seti kullanılarak derin 
öğrenme ile duygu tanımaya yönelik bir proje geliştirilmiştir. Face expression recognition veri 
seti duygu tespit projesindeki ihtiyacı karşılamaktadır. Face expression recognition veri 
setinde toplam 35887 görüntü bulunmaktadır. Görüntülerin 28821 tanesi eğitim, 7066 tanesi 
ise Public ve Private testler için ayrılmıştır. Public testler model bitirildikten sonraki başarım 
oranını test etmek için kullanılırken, Private testler ise veri setindeki görsellerden bir kısmını 
"PrivateTest" olarak ayırır ve daha sonra test etmek için kullanılır. Proje için kullanılan 
görsellerin teknik detayları bu veri seti sayesinde incelenebilmektedir. Böylece kolonlarda 
kullanılan veri setindeki örneklerin kaç gruba ayrıldığı görülebilir ve set içindeki alanlara veri 
görselleştirilmesi uygulanabilmektedir. Kullanılan veri seti 35887 satır ve 3 kolondan 
oluşmaktadır. Bu veri setinde yedi duyguyu tespit etmeye yönelik resimler bulunmaktadır. Bu 
duygular kızgın (3993 tane), iğrenme (436 tane), korku (4103 tane), mutlu (7164 tane), üzgün 
(4938 tane), şaşırma (3205 tane), nötr (4982 tane). Görsellerin yapısı fonksiyonlar yardımı ile 
48x48 boyutunda ve gri tonlarında olacak şekilde düzenlenmiştir. Model içeriğindeki 
görsellerle ayrı ayrı eğitim gerçekleştirilerek geliştirilen yeni model test edilmiştir. Model ile 
gerçekleştirilen çalışmada, her bir veri setinde yedi farklı duygu sınıfı (korku, öfke, iğrenme, 
mutluluk, nötr, üzüntü, şaşırma) ele alınmıştır.

<br>
<br>

**4.2 Sistemin-Yazılımın Kullanım Alanları**
Yüz ifadeleri, insanların günlük iletişiminde içsel duygularını ifade etmenin en açık 
göstergelerinden biridir. Bir kişinin fiziksel veya ruhsal durumu, yüz ifadeleri analiz edilerek 
tespit edilebilir. Bu nedenle yüz ifadesi tanıma; otopilot, insan-bilgisayar etkileşimi, tıbbi tedavi 
ve yüz ifadeleriyle ilgili diğer alanlarda büyük öneme sahiptir. Bu alanın giderek daha önemli 
bir araştırma konusu haline gelmekte olduğu yapılan araştırmalar sonucunda belirlenmiştir. Yüz
ifadesinden duygu tanıma, otizm ve şizofreninin belirlenmesi gibi psikolojik bilimsel, 
değerlendirmeler, uykulu bir sürücünün saptanması, Alzheimer hastalığı veya şizofreninin 
erken aşamalarında anormalliklerin belirlenmesi, tıbbi sorgular, medikal araştırmalar ve suç 
tahmin sistemleri gibi çeşitli uygulamalar için kullanılmaktadır. Ayrıca yüz ifadeleri eğitim 
alanı uygulamalarında da kullanılabilmektedir. Otomatik duygu tanımanın kullanımı; dijital 
reklam, pazarlama analizi, çevrimiçi oyunlar, müşterilerin geri bildirim değerlendirmesi, ticari, 
askeri, güvenlik, sosyal uygulamalarda, sağlık hizmetleri gibi çeşitli akıllı sistemlerde (e-sağlık, 
öğrenme, turizm için öneri, akıllı şehir, akıllı konuşma vb.) büyük bir potansiyele sahiptir. Ek 
olarak, bir kullanıcının ürün veya hizmetlerle etkileşim sırasında duygusal tepkilerini analiz 
etmek için kullanılabilir. Reklam kampanyalarının etkinliğini değerlendirmek ve hedef kitleye 
uygun pazarlama stratejileri geliştirmek için kullanılabilir. Eğitim alanında, öğrencilerin 
duygusal durumlarını izlemek ve eğitim sürecini optimize etmek için kullanılabilir.
Makine öğrenmesinde, duygu tespitine yönelik çeşitli yüz ifadesi tanıma algoritmaları 
önerilmiştir. Ancak görüntülerdeki karmaşıklık, çeşitlilik, üst üste gelme, aydınlatma 
problemleri ve yüz ifadesi tanımadaki diğer zorluklar nedeniyle, pratik uygulamalardaki tanıma 
doğruluğu halen tatmin edici sonuçlar vermemektedir. Son yıllarda sinir ağlarında olan 
gelişmeler derin öğrenme etkili modelleri ortaya çıkarmıştır ve çok sayıda derin öğrenme 
mimarisinin gelişimine yol açmıştır. Bu alanlardan biri olan yüz ifadeleri tanıma, bilgisayarlı 
görme ve yapay zekâ alanında önemli bir role sahiptir. Derin öğrenme, duygu tanıma 
problemleri için sınıflandırma verimliliği konusunda gerçek bir umut vadetmiştir. Özellikle 
derin evrişimli ağlar görüntülerin, videoların ve sesin işlenmesinde çığır açmıştır.
Görüntülerdeki nesnelerin ve bölgelerin algılanması, bölümlendirilmesi ve tanınması için 
büyük bir başarı uygulanması sağlanmaktadır.
Bunlara örnek olarak trafik işareti tanıma, biyolojik görüntülerin bölümlere ayrılması ve doğal 
görüntülerde yüzlerin, metinlerin, yayaların ve insan vücutlarının tespiti gibi etiketlenmiş 
verilerin fazla olduğu uygulamalar, otonom robotlar veya otonom araçlar verilebilir. CNN'in 
yakın zamandaki en büyük başarısı yüz tanıma uygulamasıdır. Kişinin jest ve mimiklerinden
yüzünün incelenmesiyle ne tür duyguya sahip olduğu kolayca anlaşılabilmektedir. Merak ile 
veri kullanımı aktifleştirilmektedir. Kullanıcıların yararına olan ve her türlü sektörde kullanım 
alanı bulabilen önemli bir çalışmadır. Tüm bu faydaları düşünüldüğünde duygu analizi hem 
günümüzün hem de geleceğin çalışma alanları olduğu ve gün geçtikçe de vazgeçilmez bir 
teknoloji olacağı ön görülmektedir.

<br>
<br>
**4.3 Potansiyel Hedef Kullanıcılar**
İnsan yüzleriyle gerçekleştirilen duygu analiz projeleri; ticaret, askeri saha, üretim, sağlık ve 
hizmet sektörleri için ilgi çekici olmaktadır. Artık firmalar kendileri ile ilgili neler 
düşünüldüğünü bilmek ve bu düşüncelere göre hareket etmek istemektedirler. Bu sayede derin 
öğrenme destekli sistemler ekonomi, pazarlama, politika, vb. araştırma alanlarında 
kullanılabilecek zengin veri kaynağı durumuna getirilmiştir. Bu durum, Duygu Analizi 
konusunun gün geçtikçe önem kazanmasına yardımcı olmuştur. Yüz ifadelerinden duygu 
tanımanın ileri aşamasında, metin madenciliği kavramı öne çıkmaktadır.
Derin öğrenme destekli duygu tanıma projesi oldukça yoğun bir hedef kullanıcı kitlesine 
sahiptir. Bu kitlede bahsedilen birçok sektör vardır. Derin öğrenme destekli duygu tanıma 
projesinin kamera sistemlerine entegre edilmesi sayesinde zengin veri kaynağı elde 
edilmektedir. Bu veri kaynağının işlenmesi ile çok çeşitli bilgiler edinilmektedir. Bir okul 
sistemine bağlanmasıyla eğitim hakkındaki bilgiler edilirken, askeri bir sisteme entegre 
edilmesi suç analizlerinin duygu tespit sayesinde gerçekleştirilmesini kolaylaştırmaktadır. Bir 
ticaret şirketi, pazarlama yaptığı mağazaya sistemi entegre ederse, müşterilerin farklı ürünlere 
karşı oluşturduğu duygu tutumlarını analiz ederek ulaşmış olacaktır. Markaların stratejik, hızlı 
ve daha bilinçli pazarlama ve ürün geliştirme kararları vermelerine yardımcı olur. Böylece 
müşterilerine ve iç muhasebesine karşı faydacı bir tutum sergilemesi sağlanmış olacaktır. Proje 
bir hastane kamera sistemine bağlanırsa, belirli hastalıkların önceden tespitini saptayacak ve 
yeni gelişim gösteren bulguların belirlenmesini kolaylaştıracaktır. Okul sistemine olan faydacı 
etkisi ise şu şekilde açıklanabilmektedir, öğrencilerin derslere göre gösterdiği duygu durumları 
kayıt altına alınmaktadır. Bu sayede yaş gruplarının derslere karşı tutumları ve konuların zorluk 
seviyeleri gibi bilgiler öğrenilebilmektedir. Ek olarak, yapılmış olan bir yemeğin uluslararası 
insanlar topluluğunun damak zevkine hitap edip, etmemesi konusunda da kullanılabilir. 
Toplumun ruh durumunun tespit edilmesi, film değerlendirmeleri, pazar-fiyat denge analizleri, 
Bilimsel ve medikal araştırmalar, Suç analizi, güvenlik, istihbarat, vb. gibi hem genel hem de 
özel anlamda birçok alanda kullanılabilmektedir Projenin potansiyel ve hedef kullanıcıları 
arasında, eğitimciler, psikologlar, sağlık profesyonelleri ve iletişim uzmanları yer almaktadır. 
Makine Etkileşimi ve Duygu Analizi, Pazarlama ve reklamcılık stratejilerinde, Müşteri 
Deneyimi analizinde, Güvenlik ve İzleme, Sağlık Sektörü ve sosyal bilimlerde çeşitli alt 
alanların uygulamalarında kullanılıp geliştirilebilir. Tüm olası değerlendirmeler saymakla 
maalesef bitmemektedir. Her alanda kullanılma potansiyeli olduğu ön görülen bu sistemlerin 
geleceğe yönelik sadece bir tahmin olarak kalmadığı çünkü günümüzde bu sistemden 
yararlanıldığı da bilinmektedir.

<br>
<br>

![Ekran Görüntüsü (102)](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/f560dbaa-2adb-4e6d-8541-1549dc06d64b)
![Ekran Görüntüsü (103)](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/b22230dd-e68c-4ba4-b43c-efc1a7a48660)
![Ekran Görüntüsü (104)](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/d885fb4f-0ef9-4439-9b6f-a485408b4ab2)
![Ekran görüntüsü 2023-06-11 163034](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/4787e6f7-1c60-4fea-ace5-f6d5e4d04ebb)

![Ekran Görüntüsü (106)](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/0e1a1911-2a6c-4e11-8672-b925a7f60399)
![Ekran görüntüsü 2023-06-11 170327](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/bcb8dac7-2ddc-4f8c-b789-d985dc287257)
![Ekran görüntüsü 2023-06-11 170351](https://github.com/meryemozlem/real_time_facial_emotion_detection/assets/82104183/8492315d-48a1-479f-977e-477ad0860de6)



