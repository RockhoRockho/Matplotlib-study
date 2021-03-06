# matplotlib-studynote

--

## Day 1 (2021-08-31)

- Matplotlib 특징 학습
- 라인 플롯 학습(`plt.Figure()`, `plt.Axes()`)
- 라인 스타일 학습
- 색상 스타일 학습
- 플롯 축 학습
- 플롯 레이블 학습
- 폰트관리자 학습
- 플롯 범례(legend) 학습
- 다중 플롯 학습(`plt.subplot()`, `plt.subplots()`, `plt.GridSpec`, `grid`)
- 텍스트와 주석 학습(`annotate()`)
- 눈금 맞춤 학습(Customizing Ticks)
- 스타일 학습
- 플롯 종류 학습
- 막대 플롯 학습(`bar()`, `barh()`)

--

## Day 2 (2021-09-01)

- 스템플롯 학습(`stem()`)
- 박스플롯 학습(`boxplot()`)
- 산점도 학습(`scatter()`, `cmap()`)
- x와 y의 일관성 차트(coherence) 학습
- 오차 막대 학습(`errorbar()`)
- 2차원 유사 플롯 학습(`pcolor`/`pcolormesh`)
- 히스토그램, 구간화, 밀도 학습(`hist()`, `bin()`, `hexbin()`)
- 밀도와 등고선 플롯 학습(`contour()`, `imshow()`)
- 스트림 플롯 학습(`streamplot())
- 화살표 2차원 필드 학습(`quiver()`)
- 파이 차트 학습(`pie()`)
- 레이다 차트 학습
- 생키 다이어그램 학습(`Sankey()`)
- 3차원 플로팅 학습(`mplot3d()`)

---

##  Day 3 (2021-10-07)

- FontProperties를 사용하는 방법 
 - 1. 그래프의 폰트가 필요한 항목마다 지정한다. 
  - `fm.FontProperties(fname=font, size=15)`
 - 2. matplotlib 라이브러리의 rcParams[]로 전역 글꼴로 설정한다. 
  - `plt.rcParams['font.family'] = 'NanumGothic'`
  - `plt.rc('font', family=font_name)`
 - 3. rcParams를 matplotlib 설정 파일에 직접 넣어준다.
  - font.family: NanumGothicCoding

- anscombe 그래프출력 
 - figure()
 - add_subplot(행, 열, 위치)
 - plot() 데이터전달
 - title() 제목추가
 - suptitle() 전체제목추가
 - show() 그래프출력

---

##  Day 4 (2021-10-07)

- 워드클라우드
 - generate() : 함수로 단어별 출현 빈도수를 비율로 계산
 - STOPWORDS : 불용어
 - 함수 인수 앞 `*` : 가변인자
 - 함수 인수 앞 `**` : 데이터 dict변형

- nltk의 헌법 데이터를 가져오기 실행
 - 명사형 단어로 한국지도에 단어로 워드클라우딩 

---

##  Day 5 (2021-10-15)

- folium 실시
 - shop_201806_01.csv파일을 바탕으로 구글서울지도안에 마커표시
 - 시도, 군구열 추가생성
 - 서울에 해당관련 건물(상권업종대분류명-학문/교육, 부동산, 컴퓨터를 상권업종중분류명으로 세분화하여 지도표시)
 - folium.map에  

---

##  Day 6 (2021-10-18)

전국도시공원 표준데이터 이용
- 데이터 가져오기
- 데이터 정보확인
- 데이터 전처리
- 데이터 시각화
 - ggplot
 - seaborn(scatterplot) 시각화
 - coord_flip() x,y축 변경으로 세로축바를 가로축바로 변경

---

##  Day 7 (2021-10-20)

따릉이 데이터 이용
- 데이터 가져오기
- 데이터 정보확인
- 데이터 전처리(역이름, 위도, 경도, LCD자전거, QR자전거, 새싹자전거, 총자전거 대수(만듦))
- 서울시 지도에 따릉이역 마커표시, 마커주석에 데이터넣기, 원표시

공무원 법인카드 내역서 데이터 이용

- 데이터 가져오기
- 데이터 정보확인
- 데이터 전처리(datetime을 이용하여 연, 월, 연월, 일, 시, 분, 요일 데이터 만듦)

##  Day 8 (2021-10-25)

공무원 법인카드 워드클라우드
- 매장명 열을 만들어 데이터 전처리후 제작

스타벅스 사이트
- 전국 스타벅스 매장(시도/구군별) 파악
