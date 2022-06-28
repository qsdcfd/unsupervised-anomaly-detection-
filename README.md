# 


## 주관 데이콘 

<br>

![image](https://user-images.githubusercontent.com/86671456/176123398-94da4445-6ef5-4115-a7d8-3e70fb956a86.png)


## Abstract

| 분석명 |목적|결과|
|:-----:|----------|-----|
|신용카드 사기 거래 탐지| 비지도학습기반으로 신용카드 사기 거리 탐지 AI모델 만들기|미정|

|  소스 데이터 |     데이터 입수 난이도    |      분석방법     |데이터 출처|
|:------------------:| -----|:---------------:|-----------|
|CSV|하 |Unsupervised Learniing, Clustering   |데이콘|
|  분석 적용 난이도  |     분석적용 난이도 사유    |      분석주기     | 분석결과 검증 Owner|
|상| 비지도학습을 활요한 프로젝트 경험이 미약함|Daily|Dacon |



<br>

### Machine Learning Project 

---
**진행: 애자일 방법론**

|  프로젝트 순서 |     Point    | 세부 내용 |  
|:------------------:| -----|------|
|문제 정의|해결할 점, 찾아내야할 점 |신용카드 거래데이터를 활용하여 사기 거래를 찾는 비지도학습|
|데이터 수집|공개 데이터, 자체 수집, 제공된 데이터 |데이콘data, |   
|데이터 전처리|문제에 따라서 처리해야할 방향 설정 ||
|Feature Engineering|모델 선정 혹은 평가 지표에 큰 영향|항구별로 오는 유통 제품 파악하여 labeling and Encoding|
|연관 데이터 추가|추가 수집 |제주도민의 유통 제품, 쿠팡의 제주도 물류 허브를 구축하기 위한 시도  |
|알고리즘 선택| 기본적, 현대적|Catboost, DNN, Blending|   
|모델 학습|모델을 통해서 얻고 싶은 것 |기존 데이터의 양이 적어서 국토연구원 데이터를 추가하여 항구별 유통 파악 및 지역별 운송장 건수 파악|
|모델 평가|확률 | 상위 9.5%|
|모델 성능 향상|성능 지표, 하이퍼파라미터, 데이터 리터러시 재수정 |하이퍼파라미터 조정 및 추가 Ensembling, Top5%   |

<br>

### Basic information

**공식기간: 2022.07.01 ~ 2022.08.19**


- 인원:김재근, 이수현, [이세현](https://github.com/qsdcfd). 이채영, 한창헌
- 직책: 
- 데이터: Dacon 데이터(train, val, test, submission)
- 주 역할:
- 보조 역할: 
- 추가 역할:
- 협업장소: Github, GoogleMeet
- 소통: Slack, Notion,Git project, Google OS
- 저장소: Github, Google Drive
- 개발환경: Visual studio code, Juypter Notebook, colab
- 언어 :python 3.8.x
- 라이브러리:Numpy,Pandas, Scikit-learn 1.1.x, lazypredict, Pycaret,Keras, Tensorflow, Pytorch
- 시각화 라이브러리: Seaborn, Matplotlib, Plot,Plotly, Tensorboard
- 시각화 도구: Tableau, GA
- 웹 크롤링: Slunimue

<br>

#### 파일 설명


- docs: 문서화 작업
- conf: 환경설정 관련
- build: 데이터 집산
- Definition: 프로젝트의 전반적인 문제 정의 및 내용 설정
- Data: 전처리 파일 및 모델링을 위한 파일
- models: 여러 모델들의 집합
- src :scripts
