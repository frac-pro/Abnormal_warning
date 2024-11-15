
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_wells_data(log, height=1800, width=1000):
   """
   绘制单口井的多个参数曲线图

   参数:
   log: DataFrame, 包含要绘制的井数据
   height: int, 图形高度
   width: int, 图形宽度

   返回:
   fig: plotly图形对象
   """
   columns = log.columns
   
   # 创建子图
   fig = make_subplots(
       rows=len(columns), 
       cols=1, 
       subplot_titles=columns,
       vertical_spacing=0.05  # 调整子图间距
   )

   # 为每个参数添加曲线
   for i, column in enumerate(columns):
       fig.add_trace(
           go.Scatter(
               x=log.index, 
               y=log[column], 
               mode="lines", 
               name=column,
               showlegend=True
           ),
           row=i + 1, 
           col=1
       )
       # 更新每个子图的Y轴标题
       fig.update_yaxes(title_text=column, row=i+1, col=1)
       
   # 更新布局
   fig.update_layout(
       height=height,
       width=width,
       showlegend=True,
       hovermode="x unified",  # 统一x轴显示
       title_text="Well Data Analysis"  # 添加总标题
   )

   # 更新x轴标题
   fig.update_xaxes(title_text="Data Points", row=len(columns), col=1)

   return fig