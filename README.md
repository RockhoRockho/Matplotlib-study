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
