"""法条引用图谱（Legal Citation Graph）

====================================================================
为什么需要引用图谱？
====================================================================

法律文档不是孤立的文本，而是一张**相互引用的网络**：

    《医疗纠纷预防和处理条例》第 16 条  ——「患者可查阅复制病历」
              │ 引用
              ↓
    《医疗机构病历管理规定》            ——「病历具体包括哪些材料」
              │ 引用
              ↓
    《病历书写基本规范》                ——「每类病历怎么写」

用户问「患者能查哪些病历」，正确答案需要串起这三部法规。
但纯语义检索只会返回字面/语义最相似的片段，
可能全部命中《医疗纠纷条例》，漏掉真正规定"病历范围"的下位法规。

引用图谱的解法：
    检索命中某条 → 沿引用边扩展 → 把强关联的上下位法条一起召回

这就是 GraphRAG 的核心思想：**用结构化关系补充向量检索的盲区**。

====================================================================
三类引用关系
====================================================================

1. **依据关系（based_on）**：「根据《XX法》，制定本办法」
   → 下位法 依据 上位法。表明法律位阶。

2. **参照关系（refer_to）**：「依照《XX法》第X条的规定处罚」
   → 具体条款的交叉引用。最有检索价值。

3. **修订关系（amends）**：「本办法自施行之日起，《XX》同时废止」
   → 时效性关系。

====================================================================
工程要点
====================================================================

· 名称归一化：《中华人民共和国药品管理法》≡《药品管理法》
· 噪音过滤：《医疗机构执业许可证》是证照、《条例》是自指代词，都不是引用
· PDF 断行修复：《处方 管理办法》中间的空格要去掉
· 条款级定位：「《XX法》第八十条」要能抽出条款号
"""

import os
import re
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR


# =====================================================
# 正则模式
# =====================================================

# 书名号引用：《XXX》
RE_BOOK = re.compile(r'《([^》]{2,50})》')

# 引用后紧跟的条款号：《XX法》第八十条 / 《XX法》第8条
RE_ARTICLE_AFTER = re.compile(
    r'^\s*第([一二三四五六七八九十百千零\d]+)条'
)

# 依据关系的触发词（出现在引用前）
BASED_ON_TRIGGERS = ["根据", "依据", "遵照", "按照.{0,4}的规定制定", "制定本"]

# 参照关系的触发词
REFER_TRIGGERS = ["依照", "按照", "适用", "参照", "违反", "符合"]

# 排除：证照、目录、表单类（不是法规）
RE_EXCLUDE = re.compile(
    r'(证$|证书$|执照$|许可证$|目录$|名录$|清单$|表$|申报表$|'
    r'报告$|规划$|通知书$|意见书$|批准书$|登记注册书$|证明$)'
)

# 自指代词（指本文件，不构成外部引用）
SELF_REF = {"条例", "规定", "办法", "细则", "实施细则", "暂行规定",
            "技术规范", "分组方案", "规范", "通知", "意见", "本办法",
            "本规定", "本条例", "暂行办法", "管理办法", "指导意见"}


def normalize_name(name: str) -> str:
    """法规名称归一化。

    《中华人民共和国药品管理法（2019 年修订）》 → 药品管理法
    《处方 管理办法》（PDF 断行）              → 处方管理办法
    """
    n = name.strip()
    n = re.sub(r'\s+', '', n)                    # 去掉所有空白（修复 PDF 断行）
    n = n.replace('中华人民共和国', '')
    n = re.sub(r'[（(][^）)]*[）)]', '', n)      # 去掉括号内容（修订年份等）
    return n.strip()


# =====================================================
# 引用抽取
# =====================================================

class CitationExtractor:
    """从法规文本中抽取引用关系。"""

    def __init__(self, kb_names: list[str]):
        """
        Args:
            kb_names: 知识库中的法规名列表（文件名去后缀）
        """
        self.kb_names = kb_names
        # 归一化名 → 原始文件名
        self.norm_index = {normalize_name(n): n for n in kb_names}

    def _match_kb(self, ref_name: str) -> str | None:
        """把引用名匹配到知识库里的法规。返回原始文件名，无匹配返回 None。"""
        rn = normalize_name(ref_name)

        if not rn or rn in SELF_REF or len(rn) <= 4:
            return None
        if RE_EXCLUDE.search(rn):
            return None

        # 1. 精确匹配
        if rn in self.norm_index:
            return self.norm_index[rn]

        # 2. 双向包含匹配（取最长匹配，避免"药品管理法"匹配到"药品管理法实施条例"）
        candidates = []
        for kn, kf in self.norm_index.items():
            if rn == kn:
                return kf
            if rn in kn or kn in rn:
                candidates.append((len(kn), kn, kf))

        if candidates:
            # 优先选长度最接近的
            candidates.sort(key=lambda x: abs(x[0] - len(rn)))
            return candidates[0][2]

        return None

    def _classify(self, context_before: str) -> str:
        """根据引用前的上下文判断引用类型。"""
        tail = context_before[-25:]  # 只看紧邻的文字
        for trig in BASED_ON_TRIGGERS:
            if re.search(trig, tail):
                return "based_on"
        for trig in REFER_TRIGGERS:
            if trig in tail:
                return "refer_to"
        return "mention"

    def extract(self, source_name: str, text: str) -> list[dict]:
        """从一份法规文本中抽取所有引用。

        Returns:
            [{"source":…, "target":…, "type":…, "article":…, "context":…}, …]
        """
        edges = []
        for m in RE_BOOK.finditer(text):
            ref_raw = m.group(1)
            target = self._match_kb(ref_raw)

            if target is None or target == source_name:
                continue  # 无匹配 或 自引用

            # 引用类型
            ctx_before = text[max(0, m.start() - 30):m.start()]
            rel_type = self._classify(ctx_before)

            # 引用后是否带条款号
            after = text[m.end():m.end() + 15]
            am = RE_ARTICLE_AFTER.match(after)
            article = f"第{am.group(1)}条" if am else None

            # 上下文片段（用于展示）
            ctx = text[max(0, m.start() - 40):min(len(text), m.end() + 40)]
            ctx = re.sub(r'\s+', ' ', ctx).strip()

            edges.append({
                "source": source_name,
                "target": target,
                "type": rel_type,
                "article": article,
                "context": ctx,
            })
        return edges


# =====================================================
# 引用图谱
# =====================================================

class CitationGraph:
    """法条引用图谱：建图 + 查询 + 图扩展检索。"""

    def __init__(self, data_dir: str = None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = data_dir or os.path.join(project_root, DATA_DIR)

        self.nodes: list[str] = []
        self.edges: list[dict] = []
        # 邻接表
        self.out_edges: dict[str, list[dict]] = defaultdict(list)  # 我引用了谁
        self.in_edges: dict[str, list[dict]] = defaultdict(list)   # 谁引用了我

        self._build()

    def _build(self):
        """扫描 data/ 目录，抽取引用关系建图。"""
        files = sorted(
            f for f in os.listdir(self.data_path) if f.endswith(".txt")
        )
        self.nodes = [f[:-4] for f in files]

        extractor = CitationExtractor(self.nodes)

        # 聚合：同一对 (source,target) 的多次引用合并为一条边，记权重
        agg: dict[tuple, dict] = {}

        for f in files:
            name = f[:-4]
            with open(os.path.join(self.data_path, f), encoding="utf-8") as fp:
                text = fp.read()

            for e in extractor.extract(name, text):
                key = (e["source"], e["target"])
                if key not in agg:
                    agg[key] = {
                        "source": e["source"],
                        "target": e["target"],
                        "type": e["type"],
                        "weight": 0,
                        "articles": [],
                        "contexts": [],
                    }
                a = agg[key]
                a["weight"] += 1
                # 依据关系优先级最高
                if e["type"] == "based_on":
                    a["type"] = "based_on"
                elif e["type"] == "refer_to" and a["type"] == "mention":
                    a["type"] = "refer_to"
                if e["article"] and e["article"] not in a["articles"]:
                    a["articles"].append(e["article"])
                if len(a["contexts"]) < 3:
                    a["contexts"].append(e["context"])

        self.edges = list(agg.values())
        for e in self.edges:
            self.out_edges[e["source"]].append(e)
            self.in_edges[e["target"]].append(e)

    # ---------- 查询接口 ----------

    def neighbors(self, name: str, direction: str = "both") -> list[dict]:
        """获取某部法规的引用邻居。

        Args:
            name: 法规名
            direction: "out"=它引用谁, "in"=谁引用它, "both"=双向
        """
        if direction == "out":
            return list(self.out_edges.get(name, []))
        if direction == "in":
            return list(self.in_edges.get(name, []))
        return list(self.out_edges.get(name, [])) + list(self.in_edges.get(name, []))

    def expand(self, names: list[str], max_add: int = 3,
               min_weight: int = 1) -> list[tuple[str, str, int]]:
        """图扩展：给定一组法规，返回强关联的邻居法规。

        这是 GraphRAG 的核心 —— 用引用关系补充检索召回。

        Args:
            names: 检索命中的法规名列表
            max_add: 最多扩展几部
            min_weight: 边权重下限（引用次数）

        Returns:
            [(法规名, 关联原因, 权重), …] 按权重降序
        """
        seen = set(names)
        scored: dict[str, tuple[str, int]] = {}

        for n in names:
            for e in self.neighbors(n, "both"):
                if e["weight"] < min_weight:
                    continue
                other = e["target"] if e["source"] == n else e["source"]
                if other in seen:
                    continue

                arrow = "引用" if e["source"] == n else "被引用于"
                reason = f"《{n}》{arrow}《{other}》"
                if e["articles"]:
                    reason += f"（{', '.join(e['articles'][:2])}）"

                # 同一个法规可能被多条边指向，取权重最大的
                if other not in scored or e["weight"] > scored[other][1]:
                    scored[other] = (reason, e["weight"])

        ranked = sorted(scored.items(), key=lambda kv: kv[1][1], reverse=True)
        return [(name, r, w) for name, (r, w) in ranked[:max_add]]

    # ---------- 统计 ----------

    def stats(self) -> dict:
        """图谱统计信息。"""
        in_deg = {n: len(self.in_edges.get(n, [])) for n in self.nodes}
        out_deg = {n: len(self.out_edges.get(n, [])) for n in self.nodes}
        connected = [n for n in self.nodes
                     if in_deg[n] > 0 or out_deg[n] > 0]

        type_count = defaultdict(int)
        for e in self.edges:
            type_count[e["type"]] += 1

        return {
            "node_count": len(self.nodes),
            "connected_count": len(connected),
            "edge_count": len(self.edges),
            "total_citations": sum(e["weight"] for e in self.edges),
            "type_distribution": dict(type_count),
            "most_cited": sorted(
                [(n, in_deg[n]) for n in self.nodes if in_deg[n] > 0],
                key=lambda x: x[1], reverse=True
            )[:10],
            "most_citing": sorted(
                [(n, out_deg[n]) for n in self.nodes if out_deg[n] > 0],
                key=lambda x: x[1], reverse=True
            )[:10],
        }

    # ---------- 导出（给前端可视化用）----------

    def to_vis_json(self, min_weight: int = 1) -> dict:
        """导出为可视化用的 JSON（ECharts 力导向图格式）。"""
        edges = [e for e in self.edges if e["weight"] >= min_weight]

        # 只保留有连接的节点
        active = set()
        for e in edges:
            active.add(e["source"])
            active.add(e["target"])

        # 节点度数（决定大小）
        deg = defaultdict(int)
        in_deg = defaultdict(int)
        for e in edges:
            deg[e["source"]] += e["weight"]
            deg[e["target"]] += e["weight"]
            in_deg[e["target"]] += e["weight"]

        # 按法规类型分类（决定颜色）
        def category_of(name: str) -> int:
            if name.endswith("法"):
                return 0  # 法律
            if "条例" in name:
                return 1  # 行政法规
            if any(k in name for k in ("办法", "规定", "规范", "规程")):
                return 2  # 部门规章
            return 3      # 规范性文件

        nodes = [
            {
                "name": n,
                "value": deg[n],
                "symbolSize": min(12 + deg[n] * 2.2, 62),
                "category": category_of(n),
                "inDegree": in_deg[n],
            }
            for n in sorted(active)
        ]

        links = [
            {
                "source": e["source"],
                "target": e["target"],
                "value": e["weight"],
                "type": e["type"],
                "articles": e["articles"],
            }
            for e in edges
        ]

        return {
            "nodes": nodes,
            "links": links,
            "categories": [
                {"name": "法律"},
                {"name": "行政法规"},
                {"name": "部门规章"},
                {"name": "规范性文件"},
            ],
        }

    def save_vis_json(self, path: str = None, min_weight: int = 1):
        """把可视化数据存到文件。"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = path or os.path.join(project_root, "tests", "citation_graph.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_vis_json(min_weight), f, ensure_ascii=False, indent=2)
        return path


# =====================================================
# 单例
# =====================================================

_graph_cache: CitationGraph | None = None


def get_graph() -> CitationGraph:
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = CitationGraph()
    return _graph_cache


# =====================================================
# 命令行
# =====================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="法条引用图谱")
    parser.add_argument("--stats", action="store_true", help="显示图谱统计")
    parser.add_argument("--node", type=str, help="查看某部法规的引用关系")
    parser.add_argument("--expand", type=str, help="测试图扩展（逗号分隔多个法规名）")
    parser.add_argument("--export", action="store_true", help="导出可视化 JSON")
    args = parser.parse_args()

    print("=" * 72)
    print("🕸️  法条引用图谱")
    print("=" * 72)
    print("正在扫描 data/ 抽取引用关系...")

    g = get_graph()
    s = g.stats()

    print(f"✅ 建图完成\n")
    print(f"  节点: {s['node_count']} 部法规（其中 {s['connected_count']} 部有引用关系）")
    print(f"  边:   {s['edge_count']} 条唯一引用边")
    print(f"  引用: {s['total_citations']} 处")
    print(f"  类型: {s['type_distribution']}")

    if args.stats or not any([args.node, args.expand, args.export]):
        print(f"\n{'─'*72}")
        print("  📥 被引用最多（法律位阶高 / 基础性强）")
        print(f"{'─'*72}")
        for n, d in s["most_cited"]:
            bar = "█" * d
            print(f"  {d:2d} {bar:<12} 《{n}》")

        print(f"\n{'─'*72}")
        print("  📤 引用他人最多（依赖性强 / 综合性文件）")
        print(f"{'─'*72}")
        for n, d in s["most_citing"]:
            bar = "█" * d
            print(f"  {d:2d} {bar:<12} 《{n}》")

    if args.node:
        print(f"\n{'─'*72}")
        print(f"  🔗 《{args.node}》的引用关系")
        print(f"{'─'*72}")
        out = g.neighbors(args.node, "out")
        inn = g.neighbors(args.node, "in")
        print(f"\n  ↗ 它引用了 {len(out)} 部：")
        for e in sorted(out, key=lambda x: -x["weight"]):
            arts = f" {e['articles']}" if e["articles"] else ""
            print(f"     [{e['type']:<9}] ×{e['weight']}  《{e['target']}》{arts}")
        print(f"\n  ↙ {len(inn)} 部引用了它：")
        for e in sorted(inn, key=lambda x: -x["weight"]):
            arts = f" {e['articles']}" if e["articles"] else ""
            print(f"     [{e['type']:<9}] ×{e['weight']}  《{e['source']}》{arts}")

    if args.expand:
        names = [n.strip() for n in args.expand.split(",")]
        print(f"\n{'─'*72}")
        print(f"  🌐 图扩展测试：{names}")
        print(f"{'─'*72}")
        for name, reason, w in g.expand(names, max_add=5):
            print(f"  +《{name}》")
            print(f"    理由: {reason}  (权重 {w})")

    if args.export:
        p = g.save_vis_json()
        print(f"\n📄 可视化数据已导出: {p}")

    print()
