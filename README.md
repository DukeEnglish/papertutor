# paper_tutor

照抄
1. 需要能够获取到paper的title等信息
2. 需要能够在前端页面上展示出来
3. 需要能体现

# 结构
- app/：包含了应用程序的代码和资源文件。
  - static/：用于存放静态文件，例如CSS样式表、JavaScript文件和图像文件。
    - css/：存放CSS样式表文件。
    - js/：存放JavaScript文件。
    - images/：存放图像文件。
  - templates/：存放HTML模板文件。
    - base.html：基础模板，定义了页面的共同结构和布局。
    - index.html：主页模板，展示主页的内容。
    - papers.html：论文列表模板，展示论文的内容。
- data/：用于存放论文数据或其他需要的数据文件。
- main.py：主要的Flask应用程序文件，包含路由和应用程序的逻辑。