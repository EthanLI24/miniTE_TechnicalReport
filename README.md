# miniTE Technical Report

miniTE 技术报告：原生 NVFP4 强化学习的稳定训练与高效实现。

## 阅读报告

- [English report (default)](public/index.html)
- [中文报告](public/miniTE_tech_report_zh.html)

下载仓库后，用浏览器打开 `public/index.html`，或在仓库目录启动本地预览：

```sh
python3 -m http.server 8000 --directory public
```

然后访问 `http://localhost:8000`。页面支持中英文切换、交互图表、SVG/CSV 导出和打印。

## 文件说明

`public/` 是完整的网站发布目录。报告正文、图片、交互图表代码和图表数据已经内嵌在 HTML 中；数学公式使用 MathJax 3.2.2 CDN，需要联网渲染。

此仓库发布报告网页。原始 Notion 导出、实验执行草稿和未发布论文源文件保留在研究工作区，原附件引用在网页中保留为文字。正文中的实验表格、图表数据和公开文献链接完整保留。

## Cloudflare 部署

仓库附带静态资源配置，发布目录为 `public/`，项目名称为 `minite-technical-report`。登录 Cloudflare 后运行：

```sh
npx wrangler login
npx wrangler deploy
```

也可以将仓库连接到 Cloudflare Workers Builds：构建命令留空，部署命令为 `npx wrangler deploy`，项目名称与配置文件中的名称保持一致。合并到 `main` 后自动更新网站。

首次部署会分配 `workers.dev` 地址，之后可以添加独立域名。本次 Git 提交本身不会启用公网网站。
