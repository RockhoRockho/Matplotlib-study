#!/usr/bin/env python
# coding: utf-8

# # Matplotlib
# 
# ---
# 
# 

# ## Matplotlib 특징
# 
# * 파이썬의 대표적인 과학 계산용 그래프 라이브러리
# * 선 그래프, 히스토그램, 산점도 등의 고품질 그래프 제공
# * 저수준 api를 사용한 다양한 시각화 기능 제공
# * 다양한 운영체제와 그래픽 백엔드에서 동작
# 

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
mpl.__version__


# ## 라인 플롯(Line Plot)
# 
# * 플롯(plot)은 그림(figure)와 축(axes)으로 구성
# * `plt.Figure`: 축과 그래픽, 텍스트, 레이블을 표시하는 모든 객체를 포함하는 컨테이너
# * `plt.Axes`: 눈금과 레이블이 있는 테두리 박스로 시각화를 형성하는 플롯 요소 포함

# In[8]:


fig = plt.figure()
ax = plt.axes()


# In[6]:


fig=plt.figure()
plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1.0] * 5)


# In[12]:


x =np.arange(0, 10, 0.01)
fig = plt.figure()
plt.plot(x, np.sin(x))


# In[13]:


plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))


# In[20]:


plt.plot(np.random.randn(50).cumsum())


# ## 라인 스타일(Line Style)
# 

# In[21]:


plt.plot(np.random.randn(50).cumsum(), linestyle='solid')
plt.plot(np.random.randn(50).cumsum(), linestyle='dashed')
plt.plot(np.random.randn(50).cumsum(), linestyle='dashdot')
plt.plot(np.random.randn(50).cumsum(), linestyle='dotted')


# ## 색상 스타일(Color Style)
# 

# In[22]:


plt.plot(np.random.randn(50).cumsum(), color='g')
plt.plot(np.random.randn(50).cumsum(), color='#1243FF')
plt.plot(np.random.randn(50).cumsum(), color=(0.2, 0.4, 0.6))
plt.plot(np.random.randn(50).cumsum(), color='darkblue')


# In[23]:


plt.plot(np.random.randn(50).cumsum(), color='skyblue')
plt.plot(np.random.randn(50).cumsum(), color='dodgerblue')
plt.plot(np.random.randn(50).cumsum(), color='royalblue')
plt.plot(np.random.randn(50).cumsum(), color='navy')


# In[26]:


plt.plot(np.random.randn(50).cumsum(), 'b-')
plt.plot(np.random.randn(50).cumsum(), 'g--')
plt.plot(np.random.randn(50).cumsum(), 'c-.')
plt.plot(np.random.randn(50).cumsum(), 'm:')


# ## 플롯 축(Plot Axis)

# In[27]:


plt.plot(np.random.randn(50))


# In[28]:


plt.plot(np.random.randn(50))
plt.xlim(-1, 50)
plt.ylim(-5, 5)


# In[29]:


plt.plot(np.random.randn(50))
plt.axis([-1, 50, -5, 5])


# In[30]:


plt.plot(np.random.randn(50))
plt.axis('tight')


# In[31]:


plt.plot(np.random.randn(50))
plt.axis('equal')


# ## 플롯 레이블(Plot Label)

# In[32]:


plt.plot(np.random.randn(50))
plt.title('title')
plt.xlabel('x')
plt.ylabel('random.randn')


# In[36]:


plt.plot(np.random.randn(50), 'g:',  label='A',)
plt.plot(np.random.randn(50), label='B')
plt.plot(np.random.randn(50), label='C')
plt.title('title')
plt.xlabel('x')
plt.ylabel('random.randn')
plt.legend()


# ## 폰트 관리자(Font Manager)

# In[37]:


set(f.name for f in mpl.font_manager.fontManager.ttflist)


# In[39]:


font1 = {'family':'DejaVu Sans', 'size':24, 'color':'black'}
font2 = {'family':'Kalam', 'size':18, 'weight':'bold', 'color':'darkred'}
font3 = {'family':'CentSchbook Bt', 'size':16, 'weight':'light', 'color':'blue'}

plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
plt.title('title', fontdict=font1)
plt.xlabel('xlabel', fontdict=font2)
plt.ylabel('ylabel', fontdict=font3)


# ## 플롯 범례(Plot Legend)
# 

# In[47]:


fig, ax =plt.subplots()
ax.plot(np.random.randn(10), '-r', label='A')
ax.plot(np.random.randn(10), ':g', label='B')
ax.plot(np.random.randn(10), '--b', label='C')
ax.axis('equal')
ax.legend()


# In[44]:


ax.legend(loc='lower right')
fig


# In[49]:


ax.legend(loc='upper center', frameon=False, ncol=2)
fig


# In[54]:


ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
fig


# In[56]:


plt.figure(figsize=(8, 4))
x = np.linspace(0, 10, 1000)
y = np.cos(x[:, np.newaxis] * np.arange(0, 2, 0.2))
lines = plt.plot(x, y)
plt.legend(lines[:3], ['c1', 'c2', 'c3'], loc='upper right')


# In[57]:


plt.plot(x, y[:, 0], label='c1')
plt.plot(x, y[:, 1], label='c2')
plt.plot(x, y[:, 2], label='c3')
plt.plot(x, y[:, 3:])
plt.legend(framealpha=1, frameon=True, loc='upper right')


# In[61]:


x = np.linspace(0, 20, 100)
l = np.cos(x) - np.cos(x[:, np.newaxis])

plt.imshow(l)
plt.colorbar()


# In[62]:


plt.imshow(l, cmap='Blues')
plt.colorbar()


# In[63]:


plt.imshow(l, cmap='RdBu')
plt.colorbar()


# In[65]:


speckles = (np.random.random(l.shape) < 0.01)
l[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))

plt.imshow(l, cmap='RdBu')
plt.colorbar(extend='both')
plt.clim(-1, 1)


# In[66]:


plt.imshow(l, cmap=plt.cm.get_cmap('Blues', 5))
plt.colorbar()
plt.clim(-1, 1)


# ## 다중 플롯(Multiple Subplots)

# In[67]:


ax1 = plt.axes()
ax2 = plt.axes([0.65, 0.5, 0.2, 0.3])


# In[69]:


for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.text(0.5, 0.5, str((3, 3, i)), ha= 'center')


# In[70]:


fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.text(0.5, 0.5, str((3, 3, i)), ha = 'center')


# In[71]:


fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')


# In[73]:


for i in range(3):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)), ha='center')
fig


# In[74]:


grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.4)

plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2])


# In[76]:


plt.figure(figsize=(5, 6))

x = range(1, 21)
columns = [np.random.randn(20) * i for i in range(1, 7)]

i = 0
for c in columns:
    i += 1
    
    plt.subplot(3, 2, i)
    plt.plot(x, c, marker='o', linewidth=1, label=c)
    plt.xlim(-1, 21)
    plt.ylim(c.min()-1, c.max()+1)


# ## 텍스트와 주석(Text and Annotation)

# In[78]:


fig, ax = plt.subplots()
ax.axis([0, 10, 0, 10])
ax.text(3, 6, '. transData(3, 6)', transform=ax.transData)
ax.text(0.2, 0.4, '. transAxes(0.2, 0.4)', transform=ax.transAxes)
ax.text(0.2, 0.2, '. transFigure(0.2, 0.2)', transform=fig.transFigure)


# In[79]:


ax.set_xlim(-6, 10)
ax.set_ylim(-6, 10)
fig


# In[82]:


x = np.arange(1, 40)
y = x + 1.1
plt.scatter(x, y, marker='.')
plt.axis('equal')
plt.annotate('interesting point', xy=(4, 5), xytext=(20, 10),
            arrowprops=dict(shrink=0.05))


# In[84]:


x1 = np.random.normal(30, 3, 100)
x2 = np.random.normal(20, 3, 100)
x3 = np.random.normal(10, 3, 100)

plt.plot(x1, label='p1')
plt.plot(x2, label='p2')
plt.plot(x3, label='p3')

plt.legend(bbox_to_anchor=(0, 1.02, 1., .102), loc=0, ncol=3,
          mode='extend', borderaxespad=0.)
plt.annotate('important value', (50, 20), xytext=(5, 40), arrowprops=dict(arrowstyle='->'))
plt.annotate('important value', (40, 30), xytext=(50, 40), arrowprops=dict(arrowstyle='->'))


# ## 눈금 맞춤(Customizing Ticks)
# 

# In[85]:


ax = plt.axes()
ax.plot(np.random.randn(100).cumsum())

ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())


# In[87]:


fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)


# In[88]:


for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(4))
    axi.yaxis.set_major_locator(plt.MaxNLocator(4))
fig


# In[89]:


x = np.linspace(-np.pi, np.pi, 1000, endpoint=True)
y = np.sin(x)
plt.plot(x, y)

ax = plt.gca()
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


# In[95]:


x = np.linspace(1, 10)
y = [10**el for el in x]
z = [2*el for el in x]

fig =plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x, y, '-y')
ax1.set_yscale('log')
ax1.set_title('Logarithmic plot of $ {10}^{x} $')
ax1.set_ylabel(r'$ {y} = {10}^{x} $')
plt.grid(b=True, which = 'both', axis = 'both')


ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x, y, '--r')
ax2.set_yscale('linear')
ax2.set_title('Linear plot of $ {10}^{x} $')
ax2.set_ylabel(r'$ {y} = {10}^{x} $')
plt.grid(b=True, which = 'both', axis = 'both')

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x, z, '-.g')
ax3.set_yscale('log')
ax3.set_title('Logarithmic plot of $ {2}^{x} $')
ax3.set_ylabel(r'$ {y} = {2}^{x} $')
plt.grid(b=True, which = 'both', axis = 'both')

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x, z, ':b')
ax4.set_yscale('linear')
ax4.set_title('Linear plot of $ {2}^{x} $')
ax4.set_ylabel(r'$ {y} = {2}^{x} $')
plt.grid(b=True, which = 'both', axis = 'both')


# ## 스타일(Style)

# In[94]:


fig = plt.figure(figsize=(10, 10))
x = range(1, 11)
columns = [np.random.randn(10) * i for i in range(1, 26)]

for n, v in enumerate(plt.style.available[1:]):
    plt.style.use(v)
    plt.subplot(5, 5, n+1)
    plt.title(v)
    
    for c in columns:
        plt.plot(x, c, marker='', color='royalblue', linewidth=1, alpha=0.4)
        plt.subplots_adjust(hspace=0.5, wspace=0.4)


# In[96]:


plt.style.use(['seaborn-notebook'])


# ## 플롯 종류
# 
# | 함수 | 설명 |
# |------|------|
# | `bar` | 막대 플롯(bar plot) 생성 |
# | `barbs` | barbs의 2차원 필드 그리기 |
# | `boxplot` | 상자 및 수염 플롯 생성 |
# | `cohere` | x와 y의 일관성 시각화 그리기 |
# | `contour` | 플롯 등고선 |
# | `errorbar` | 오류 막대 그래프 |
# | `hexbin` | 육각형 binning 플롯 생성 |
# | `hist` | 히스토그램 플롯 |
# | `imshow` | 축에 이미지 표시 |
# | `pcolor` | 2차원 배열의 유사 플롯 생성 |
# | `pcolormesh` | 사각 망사 그래프 |
# | `pie` | 파이 차트 플롯 |
# | `plot` | 플롯 라인 또는 마커 |
# | `quiver` | 화살표의 2차원 필드 생성 |
# | `sankey` | Sankey 흐름도 생성 |
# | `scatter` | x대 y의 산점도 생성 |
# | `stem` | 줄기 그래프 생성 |
# | `streamplot` | 벡터 흐름의 스트림 라인 그리기 |

# ## 막대 플롯(Barplot)

# In[98]:


height = [np.random.randn() * i for i in range(1, 6)]
names = ['A', 'B', 'C', 'D', 'E']
y_pos = np.arange(len(names))
plt.bar(y_pos, height)
plt.xticks(y_pos, names, fontweight='bold')
plt.xlabel('group')


# In[99]:


height = [np.random.randn() * i for i in range(1, 6)]
names = ['A', 'B', 'C', 'D', 'E']
y_pos = np.arange(len(names))
plt.barh(y_pos, height)
plt.yticks(y_pos, names, fontweight='bold')
plt.xlabel('group')


# In[102]:


bars1 = [12, 28, 1, 8 ,22]
bars2 = [28, 7, 16, 4, 10]
bars3 = [25, 3, 23, 25, 17]
bars = np.add(bars1, bars2).tolist()

r = [0, 1, 2, 3, 4]
names = ['A', 'B', 'C', 'D', 'E']

plt.bar(r, bars1, color='royalblue', edgecolor='white')
plt.bar(r, bars2, bottom=bars1, color='skyblue', edgecolor='white')
plt.bar(r, bars3, bottom=bars2, color='lightblue', edgecolor='white')

plt.xlabel('group', fontweight='bold')
plt.xticks(r, names, fontweight='bold')


# In[106]:


bar_width=0.25

bars1 = np.random.randint(1, 30, 5)
bars2 = np.random.randint(1, 30, 5)
bars3 = np.random.randint(1, 30, 5)
bars = np.add(bars1, bars2).tolist()

r1 = np.arange(len(bars1))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
names = ['A', 'B', 'C', 'D', 'E']

plt.bar(r1, bars1, color='royalblue', width=bar_width, edgecolor='white', label='r1')
plt.bar(r2, bars2, color='skyblue', width=bar_width, edgecolor='white', label='r2')
plt.bar(r3, bars3, color='lightblue', width=bar_width, edgecolor='white', label='r3')

plt.xlabel('group', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])
plt.legend()


# In[107]:


x = [0, 5, 10, 15, 30, 40, 50, 60, 100]
v = [0, -5, -10, -15, -30, -40, -50, -60, -100]
n = len(v)
y = np.ones(n)
u = np.zeros(n)

plt.barbs(x, y, u, v, length=9)
plt.xticks(x)
plt.ylim(0.98, 1.05)


# ## 스템 플롯(Stem Plot)

# In[ ]:





# ## 박스플롯(Box Plot)

# In[ ]:





# ## 산점도(Scatter Plot)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# * `cmap`: color map: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

# In[ ]:





# In[ ]:





# ## x와 y의 일관성 차트(coherence)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 오차 막대(Error Bar)
# 
# * 예상 측정 오차를 나타내는 데 사용
# * 오차 막대는 값의 불확실성을 나타냄

# In[ ]:





# In[ ]:





# ## 2차원 유사 플롯(pcolor/pcolormesh)

# In[ ]:





# In[ ]:





# ## 히스토그램, 구간화, 밀도(Histograms, Binnings, and Density)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 밀도와 등고선 플롯(Density and Contour Plots)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 스트림 플롯(Stream Plot)

# In[ ]:





# In[ ]:





# ## 화살표 2차원 필드(quiver)

# In[ ]:





# ## 파이 차트(Pie Chart)
# 
# * 원그래프, 파이차트는 전체에 대한 각 부분의 비율을 부채꼴 모양으로 나타낸 그래프
# * 각 부채꼴의 중심각이 전체에서 차지하는 비율을 나타내며, 비율을 한눈에 볼 수 있다는 장점
# * 전체적인 비율을 쉽게 파악할 수 있어서 언론사에서 통계 수치를 공개할 때 자주 활용

# In[ ]:





# In[ ]:





# ## 레이다 차트(Radar Chart)
# 
# * 어떤 측정 목표에 대한 평가항목이 여러 개일 때 항목 수에 따라 원을 같은 간격으로 나누고, 중심으로부터 일정 간격으로 동심으로 척도를 재는 칸을 나누어 각 평가항목의 정량화된 점수에 따라 그 위치에 점을 찍고 평가항목간 점을 이어 선으로 만들어 항목 간 균형을 한눈에 볼 수 있도록 해주는 도표
# * 여러 측정 목표를 함께 겹쳐 놓아 비교하기에도 편리
# 각 항목 간 비율뿐만 아니라 균형과 경향을 직관적으로 알 수 있어 편리

# In[ ]:





# ## 생키 다이어그램(Sankey Diagram)
# 
# * 흐름(Flow) 다이어그램의 한 종류로서 그 화살표의 너비로 흐름의 양을 비율적으로 보여줌

# In[ ]:





# ## 3차원 플로팅(Three-Dimensional Plotting)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 참고문헌
# 
# * Matplotlib, https://matplotlib.org/
# * Igor Milovanovi, "Python Data Visualization Cookbook", Packt
# * Jake VanderPlas, "Python Data Science Handbook", O'Reilly
# * Wes Mckinney, "Python for Data Analysis", O'Reilly
