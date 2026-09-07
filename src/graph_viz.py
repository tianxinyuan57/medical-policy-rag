"""法条引用图谱可视化 —— 生成交互式 ECharts 力导向图

嵌入 Streamlit 用 components.html()，也可以独立导出成 HTML 文件。
"""

import json
import os
import functools


# ECharts 库：优先用本地内联（Streamlit 的 srcdoc iframe 加载不了外部脚本），
# 本地文件缺失时回退到 CDN。
ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"
_ECHARTS_LOCAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "echarts.min.js",
)


@functools.lru_cache(maxsize=1)
def _echarts_script_tag() -> str:
    """返回可用的 ECharts <script> 标签。

    Streamlit 的 components.html 用 srcdoc iframe 渲染，
    外部 script src 在其中加载不到（无 base URL / CSP 限制），
    所以必须把库内联进去。
    """
    if os.path.exists(_ECHARTS_LOCAL):
        with open(_ECHARTS_LOCAL, encoding="utf-8") as f:
            lib = f.read()
        return f"<script>{lib}</script>"
    return f'<script src="{ECHARTS_CDN}"></script>'

# 四类法规的配色（深色背景下的高对比霓虹色）
CATEGORY_COLORS = [
    "#60a5fa",  # 法律       — 蓝
    "#a78bfa",  # 行政法规    — 紫
    "#34d399",  # 部门规章    — 绿
    "#fbbf24",  # 规范性文件  — 琥珀
]

# 三种引用关系的边配色
EDGE_COLORS = {
    "based_on": "#f472b6",   # 依据 — 粉（最强关系）
    "refer_to": "#38bdf8",   # 参照 — 天蓝
    "mention": "#475569",    # 提及 — 灰
}


def build_graph_html(
    graph_data: dict,
    height: int = 680,
    highlight: list[str] | None = None,
    dark: bool = True,
) -> str:
    """生成引用图谱的交互式 HTML。

    Args:
        graph_data: citation_graph.to_vis_json() 的输出
        height: 画布高度（px）
        highlight: 需要高亮的节点名列表（用于「当前检索命中」）
        dark: 深色主题
    """
    nodes = graph_data["nodes"]
    links = graph_data["links"]
    categories = graph_data["categories"]
    highlight = highlight or []

    # ---- 节点：加上颜色、标签、高亮态 ----
    vis_nodes = []
    for n in nodes:
        cat = n["category"]
        is_hl = n["name"] in highlight
        vis_nodes.append({
            "id": n["name"],
            "name": n["name"],
            "value": n["value"],
            "category": cat,
            "symbolSize": n["symbolSize"],
            "inDegree": n.get("inDegree", 0),
            "itemStyle": {
                "color": "#fb7185" if is_hl else CATEGORY_COLORS[cat],
                "borderColor": "#fff" if is_hl else "rgba(255,255,255,0.25)",
                "borderWidth": 3 if is_hl else 1,
                "shadowBlur": 22 if is_hl else 8,
                "shadowColor": "#fb7185" if is_hl else CATEGORY_COLORS[cat],
            },
            "label": {
                # 只有大节点或高亮节点默认显示标签，避免糊成一团
                "show": n["symbolSize"] > 26 or is_hl,
                "fontWeight": "bold" if is_hl else "normal",
                "color": "#fff" if is_hl else ("#cbd5e1" if dark else "#334155"),
            },
        })

    # ---- 边：按关系类型着色，宽度随权重 ----
    vis_links = []
    for l in links:
        t = l.get("type", "mention")
        w = l.get("value", 1)
        vis_links.append({
            "source": l["source"],
            "target": l["target"],
            "value": w,
            "relType": t,
            "articles": l.get("articles", []),
            "lineStyle": {
                "color": EDGE_COLORS.get(t, EDGE_COLORS["mention"]),
                "width": min(1 + w * 0.55, 6),
                "opacity": 0.75 if t == "based_on" else 0.5,
                "curveness": 0.18,
            },
        })

    bg = "#0f172a" if dark else "#ffffff"
    text_color = "#e2e8f0" if dark else "#1e293b"

    payload = json.dumps({
        "nodes": vis_nodes,
        "links": vis_links,
        "categories": [
            {"name": c["name"], "itemStyle": {"color": CATEGORY_COLORS[i]}}
            for i, c in enumerate(categories)
        ],
    }, ensure_ascii=False)

    rel_labels = json.dumps({
        "based_on": "依据（下位法依据上位法）",
        "refer_to": "参照（条款交叉引用）",
        "mention": "提及",
    }, ensure_ascii=False)

    return f"""
<div id="graph-wrap" style="position:relative;width:100%;height:{height}px;
     background:{bg};border-radius:14px;overflow:hidden;">
  <div id="graph" style="width:100%;height:100%;"></div>

  <!-- 图例：引用关系类型 -->
  <div style="position:absolute;left:14px;bottom:12px;
       background:rgba(15,23,42,.82);border:1px solid rgba(255,255,255,.12);
       border-radius:10px;padding:9px 13px;font-size:11.5px;
       color:{text_color};line-height:1.9;backdrop-filter:blur(6px);">
    <div style="font-weight:600;margin-bottom:3px;opacity:.75;">引用关系</div>
    <div><span style="display:inline-block;width:22px;height:3px;
         background:{EDGE_COLORS['based_on']};vertical-align:middle;
         margin-right:7px;border-radius:2px;"></span>依据</div>
    <div><span style="display:inline-block;width:22px;height:3px;
         background:{EDGE_COLORS['refer_to']};vertical-align:middle;
         margin-right:7px;border-radius:2px;"></span>参照</div>
    <div><span style="display:inline-block;width:22px;height:3px;
         background:{EDGE_COLORS['mention']};vertical-align:middle;
         margin-right:7px;border-radius:2px;"></span>提及</div>
  </div>

  <!-- 操作提示 -->
  <div style="position:absolute;right:14px;bottom:12px;
       background:rgba(15,23,42,.82);border:1px solid rgba(255,255,255,.12);
       border-radius:10px;padding:8px 13px;font-size:11px;
       color:{text_color};opacity:.72;backdrop-filter:blur(6px);">
    悬停查看关联 · 拖拽调整布局 · 滚轮缩放
  </div>
</div>

{_echarts_script_tag()}
<script>
(function() {{
  var el = document.getElementById('graph');
  if (!el) return;
  if (typeof echarts === 'undefined') {{
    el.innerHTML = '<div style="display:flex;align-items:center;'
      + 'justify-content:center;height:100%;color:#94a3b8;font-size:13px;">'
      + 'ECharts 未能加载，请检查 assets/echarts.min.js</div>';
    return;
  }}

  var chart = echarts.init(el, null, {{renderer: 'canvas'}});
  var data = {payload};
  var relLabels = {rel_labels};

  chart.setOption({{
    backgroundColor: 'transparent',
    tooltip: {{
      backgroundColor: 'rgba(15,23,42,.95)',
      borderColor: 'rgba(255,255,255,.15)',
      textStyle: {{color: '#e2e8f0', fontSize: 12}},
      extraCssText: 'border-radius:10px;padding:11px 14px;max-width:320px;'
                    + 'box-shadow:0 8px 28px rgba(0,0,0,.5);',
      formatter: function(p) {{
        if (p.dataType === 'node') {{
          return '<b style="font-size:13px;">《' + p.data.name + '》</b><br/>'
               + '<span style="opacity:.7">类型：</span>'
               + data.categories[p.data.category].name + '<br/>'
               + '<span style="opacity:.7">被引用：</span>'
               + p.data.inDegree + ' 次<br/>'
               + '<span style="opacity:.7">关联强度：</span>' + p.data.value;
        }}
        var arts = p.data.articles && p.data.articles.length
                 ? '<br/><span style="opacity:.7">涉及：</span>'
                   + p.data.articles.join('、') : '';
        return '《' + p.data.source + '》<br/>'
             + '<span style="color:#94a3b8;">↓ '
             + (relLabels[p.data.relType] || p.data.relType) + '</span><br/>'
             + '《' + p.data.target + '》<br/>'
             + '<span style="opacity:.7">引用次数：</span>' + p.data.value + arts;
      }}
    }},
    legend: [{{
      data: data.categories.map(function(c) {{ return c.name; }}),
      top: 12, left: 'center',
      textStyle: {{color: '{text_color}', fontSize: 12}},
      itemGap: 18, itemWidth: 12, itemHeight: 12,
      icon: 'circle',
      inactiveColor: '#475569'
    }}],
    animationDuration: 1400,
    animationEasingUpdate: 'quinticInOut',
    series: [{{
      type: 'graph',
      layout: 'force',
      data: data.nodes,
      links: data.links,
      categories: data.categories,
      roam: true,
      draggable: true,
      focusNodeAdjacency: true,
      label: {{
        position: 'right',
        formatter: function(p) {{
          var n = p.data.name;
          return n.length > 13 ? n.slice(0, 12) + '…' : n;
        }},
        fontSize: 11.5
      }},
      labelLayout: {{hideOverlap: true}},
      edgeSymbol: ['none', 'arrow'],
      edgeSymbolSize: [0, 7],
      force: {{
        repulsion: 380,
        gravity: 0.09,
        edgeLength: [70, 190],
        friction: 0.14,
        layoutAnimation: true
      }},
      emphasis: {{
        focus: 'adjacency',
        scale: 1.12,
        label: {{show: true, fontWeight: 'bold', fontSize: 13, color: '#fff'}},
        lineStyle: {{opacity: 0.95, width: 4}},
        itemStyle: {{shadowBlur: 28}}
      }},
      blur: {{
        itemStyle: {{opacity: 0.14}},
        lineStyle: {{opacity: 0.05}},
        label: {{opacity: 0.1}}
      }}
    }}]
  }});

  window.addEventListener('resize', function() {{ chart.resize(); }});
}})();
</script>
"""


def build_ego_graph_html(
    graph,
    center: str,
    height: int = 380,
    dark: bool = True,
) -> str:
    """生成单个法规的「自我中心网络」小图（只显示它和直接邻居）。

    用在问答结果里，展示当前命中法规的引用关系。
    """
    from collections import defaultdict

    edges = graph.neighbors(center, "both")
    if not edges:
        return ""

    active = {center}
    for e in edges:
        active.add(e["source"])
        active.add(e["target"])

    deg = defaultdict(int)
    in_deg = defaultdict(int)
    for e in edges:
        deg[e["source"]] += e["weight"]
        deg[e["target"]] += e["weight"]
        in_deg[e["target"]] += e["weight"]

    def category_of(name: str) -> int:
        if name.endswith("法"):
            return 0
        if "条例" in name:
            return 1
        if any(k in name for k in ("办法", "规定", "规范", "规程")):
            return 2
        return 3

    sub = {
        "nodes": [
            {
                "name": n,
                "value": deg[n],
                "symbolSize": 46 if n == center else min(18 + deg[n] * 2.6, 40),
                "category": category_of(n),
                "inDegree": in_deg[n],
            }
            for n in sorted(active)
        ],
        "links": [
            {
                "source": e["source"],
                "target": e["target"],
                "value": e["weight"],
                "type": e["type"],
                "articles": e["articles"],
            }
            for e in edges
        ],
        "categories": [
            {"name": "法律"}, {"name": "行政法规"},
            {"name": "部门规章"}, {"name": "规范性文件"},
        ],
    }

    return build_graph_html(sub, height=height, highlight=[center], dark=dark)


def export_standalone_html(graph_data: dict, out_path: str, title: str = "法条引用图谱"):
    """导出成可独立打开的 HTML 文件。"""
    body = build_graph_html(graph_data, height=780)
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  body {{margin:0;background:#0f172a;color:#e2e8f0;
         font-family:'Noto Sans SC',-apple-system,BlinkMacSystemFont,sans-serif;}}
  .hd {{padding:22px 26px 12px;}}
  .hd h1 {{margin:0 0 6px;font-size:22px;
           background:linear-gradient(135deg,#60a5fa,#a78bfa);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
  .hd p {{margin:0;color:#64748b;font-size:13px;}}
  .wrap {{padding:0 26px 26px;}}
</style></head>
<body>
  <div class="hd">
    <h1>🕸️ 法条引用图谱</h1>
    <p>{len(graph_data['nodes'])} 部法规 · {len(graph_data['links'])} 条引用关系 ·
       节点大小 = 关联强度，颜色 = 法规位阶</p>
  </div>
  <div class="wrap">{body}</div>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from citation_graph import get_graph

    g = get_graph()
    data = g.to_vis_json()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(project_root, "tests", "citation_graph.html")
    export_standalone_html(data, out)
    print(f"✅ 已导出独立 HTML: {out}")
    print(f"   {len(data['nodes'])} 节点 / {len(data['links'])} 边")
