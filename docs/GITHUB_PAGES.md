# GitHub 项目主页部署

已准备两种首页：GitHub 仓库直接显示的英文 `README.md`，以及 `site/` 中的独立学术项目页。中文工程说明保留在 `README.zh-CN.md`。

## 本地预览

在仓库根目录执行：

```bash
python3 -m http.server 8785 --bind 127.0.0.1 --directory site
```

浏览 `http://127.0.0.1:8785/`。网站为纯静态 HTML/CSS/JS，无需 npm、数据库、账号或第三方字体服务。包含三数据集切换、论文图放大、命令复制、结果 CSV 及英文使用指南。

## 发布到 GitHub Pages

代码仓库：https://github.com/Alexander-wu/FutureWorlds。GitHub Pages 已启用 GitHub Actions，主页地址：https://alexander-wu.github.io/FutureWorlds/。后续推送 main 的 site/ 修改将自动更新。

1. 把经过确认的 `FutureWorlds/` 内容放入目标代码仓库。建议独立项目仓库，避免覆盖个人主页。
2. 在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
3. 在 **Actions → Deploy project website → Run workflow** 运行随仓库提供的部署任务；后续 main 分支的 `site/` 改动会触发更新。
4. 以成功部署任务返回的真实 `page_url` 为准，将其填入仓库 About 的 Website 和 README 链接。不要在部署前把猜测的地址当作已上线地址。

工作流仅上传 `site/`，不会把权重、实验数据、内部路径或整个工作区作为网页发布。所有资源使用相对路径，可部署在独立仓库子路径。当前配置使用 GitHub 官方 Pages 工作流版本，参考 [GitHub 文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

## 发布前内容核对

- 作者、单位、论文链接和 BibTeX 尚未确认，因此当前页面没有伪造这些字段。
- Hugging Face 仓库发布后，再把真实的模型链接加入页面；目前标记 Publication pending。
- 主页指标来自正文 Table 1，见 `site/data/results.json` 中的来源哈希。论文更新后同步表格、JSON、CSV、首页指标及 README。
- 图像为已有论文素材的原样副本，来源与 SHA 记录在 `site/data/assets.json`。
- 当前独立站不是闭环机器人演示。静态案例和指标没有被描述为真实机器人成功率。

## 网站文件

| 文件 | 用途 |
|---|---|
| `site/index.html` | 主项目页 |
| `site/guides.html` | 英文复现 / 数据 / 训练指南 |
| `site/style.css` | 共享响应式样式 |
| `site/app.js` | 表格切换、命令复制、图片查看器 |
| `site/assets/` | 论文方法图、案例、效率图 |
| `site/data/` | 可下载结果及素材来源 |
| `.github/workflows/pages.yml` | GitHub Pages 部署 |

本地源码修改后需重新生成 `SOURCE_MANIFEST.json`，再运行 `python scripts/verify_sources.py`；源码发布包应排除 `.venv`、`.git`、本地路径配置和缓存。

## Animated cases

The Demos section includes synchronized GT / SFT / MemSPO videos for RT-1, BridgeV2, and RoboCasa. The three MP4 files total approximately 1.2 MB and loop with native pause, seek, and fullscreen controls. Playback respects reduced-motion preferences. All 32 source frames remain in order, without interpolation. Qualitative checkpoints use 500 / 200 / 400 MemSPO updates respectively; hashes and provenance are recorded in `site/data/rollouts.json`.
