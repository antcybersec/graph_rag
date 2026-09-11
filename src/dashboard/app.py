"""Streamlit dashboard comparing the three RAG pipelines on the shared eval set.

Run from anywhere:  streamlit run src/dashboard/app.py

Reads data/results/benchmark_results.jsonl (resolved from this file, not the
cwd). All data shaping lives in src/dashboard/data.py; this module is layout,
color, and charts only.

Color: exactly three categorical slots (blue / orange / aqua), one per pipeline,
held constant across every chart so a reader who learns "GraphRAG is orange"
stays right. Both the light and dark steps were validated for CVD separation and
contrast against Streamlit's real surfaces before being hardcoded here; on the
light surface aqua sits just under 3:1, so every bar chart carries visible value
labels and every section has a table view, which is the required relief.

Chrome (not data color), second pass: the first pass took TigerGraph's product UI
and a Dribbble "benchmark dashboard" shot literally -- orange/blue brand color used
as UI chrome (top rule, section-header bars), rounded cards, alternating category
bands. Reviewed against a real (bad) example of hackathon-submission design --
sanctiontrace.netlify.app, 2026-09-11: dark void background, gradient headline text,
five different neon-colored unlabeled stat callouts ("+126%", "720x faster than
human" with no visible baseline), glassmorphism cards -- the generic "AI SaaS landing
page" template LLM assistants default to when asked for something that "looks
impressive," optimized for a 10-second glance rather than for someone actually
checking a claim. That's the wrong reference class for a data-analysis tool.

Current design commitments instead: (1) one considered dark theme, not a light/dark
pair each half-tuned -- see `.streamlit/config.toml`. (2) Chrome (borders, dividers,
kickers, the UI accent color) stays strictly neutral; color is spent ONLY on the
three pipeline series, so a color on this page always means "which pipeline," never
"this is a brand element" -- the first pass violated this by making TigerGraph orange
both the UI accent AND GraphRAG's data color. (3) No gradients, no glow, no
unlabeled/uncontextualized stat callouts -- every number on this page sits next to
what it measures and how many rows it's over. (4) A tabular/mono numeral face for
data (KPI values, table figures) borrowed from Linear/Vercel Analytics/Stripe-style
tools, not from marketing sites -- real information hierarchy through type weight and
size, not identical boxed widgets. (5) The Dribbble category-band technique
(band_categories() below) is kept because it's a genuine reading aid, toned to a
neutral tint since our bars aren't good/bad-coded the way that reference's green/pink
bands were.
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run src/dashboard/app.py` (which puts src/dashboard, not the
# repo root, on sys.path) to resolve the `src.dashboard.data` import.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dashboard import data as bench  # noqa: E402

st.set_page_config(
    page_title="RAG pipeline comparison",
    page_icon="",
    layout="wide",
)

FONT = '"Inter", system-ui, -apple-system, "Segoe UI", sans-serif'
MONO = '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace'

# Chrome only -- see the module docstring. Two fonts, each with one job: Inter for
# everything read as prose, IBM Plex Mono only for numerals (KPI values, table
# figures) where fixed-width digits actually earn their keep -- not decorative
# tracked-uppercase labels, which is the tic this is deliberately avoiding.
CSS_OVERRIDES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

/* Neutral cards: a hairline border + a fill one step lighter than the page
   background (elevation by contrast, not shadow/glow -- shadows barely read on a
   dark surface anyway). No color here; color is reserved for pipeline identity. */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px !important;
    border-color: rgba(231,233,236,0.10) !important;
    background: rgba(231,233,236,0.03);
}

/* Kicker: a small, quiet uppercase label -- used exactly once, above the page
   title, to say what this page is before the title says what it's about. */
.kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    color: #7d8590;
    margin-bottom: 0.3rem;
}

/* Section headers: plain weight, a hairline rule underneath -- a document's section
   break, not a card or a colored tab. */
.section-head {
    font-size: 1.15rem;
    font-weight: 600;
    margin: 0.3rem 0 0.7rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(231,233,236,0.10);
}

/* Custom KPI block replacing st.metric for the pipeline summary tiles: a small
   muted label, then the number itself large and in the mono numeral face, then a
   muted detail line -- real size/weight hierarchy instead of three identical
   Streamlit metric widgets. */
.kpi-label { font-size: 0.72rem; letter-spacing: 0.06em; color: #7d8590; text-transform: uppercase; }
.kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 2.1rem; font-weight: 600; line-height: 1.3; margin: 0.15rem 0; font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: 0.8rem; color: #a3a9b3; }

/* Same mono/tabular treatment on Streamlit's own st.metric (used in the per-question
   drill-down) so the numeral system is one system, not "custom cards get the nice
   font and everything else gets Streamlit's default." */
div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }
</style>
"""
st.markdown(CSS_OVERRIDES, unsafe_allow_html=True)


def section_header(title: str) -> None:
    """Replaces a bare st.subheader with the hairline-rule header style above."""
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)

# Categorical slots 1-3 + chart ink, stepped per mode. Do not reorder: the slot
# order is the CVD-safety mechanism, and the hue must follow the pipeline.
THEMES = {
    "light": {
        "surface": "#ffffff",
        "series": {"RAG": "#2a78d6", "GraphRAG": "#eb6834", "Agentic GraphRAG": "#1baf7a"},
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "sequential": ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
    },
    "dark": {
        # Kept in sync with .streamlit/config.toml's backgroundColor -- ring_markers()
        # strokes markers in this exact color so overlapping points separate cleanly.
        "surface": "#0b0d10",
        "series": {"RAG": "#3987e5", "GraphRAG": "#d95926", "Agentic GraphRAG": "#199e70"},
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "sequential": ["#0d366b", "#1c5cab", "#2a78d6", "#3987e5", "#86b6ef", "#cde2fb"],
    },
}


def active_theme() -> dict:
    """Streamlit's current mode, so the chart steps match the surface behind them."""
    mode = None
    try:
        ctx_theme = getattr(st, "context", None)
        mode = getattr(getattr(ctx_theme, "theme", None), "type", None)
    except Exception:
        mode = None
    if mode not in THEMES:
        try:
            mode = st.get_option("theme.base")
        except Exception:
            mode = None
    return THEMES.get(mode or "light", THEMES["light"])


THEME = active_theme()
SERIES = THEME["series"]


@st.cache_data(show_spinner="Loading benchmark results...")
def load(path_str: str, _mtime: float):
    """Cached load. `_mtime` is part of the cache key so a re-run of the
    benchmark invalidates it without a manual clear."""
    return bench.load_results(Path(path_str))


def style(fig: go.Figure, *, height: int = 380, showlegend: bool = True) -> go.Figure:
    """Recessive chrome, transparent ground, thin marks -- the data is the only
    loud thing. Transparent backgrounds let Streamlit's own surface show through
    in either mode."""
    fig.update_layout(
        template="none",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=THEME["text_secondary"]),
        margin=dict(l=8, r=8, t=56, b=8),
        bargap=0.34,
        bargroupgap=0.06,
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            title_text="",
            font=dict(color=THEME["text_secondary"]),
        ),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        title=dict(font=dict(size=15, color=THEME["text_primary"]), x=0, xanchor="left"),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=THEME["axis"],
        linewidth=1,
        ticks="outside",
        tickcolor=THEME["axis"],
        color=THEME["muted"],
        title_font=dict(color=THEME["text_secondary"], size=12),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=THEME["grid"],
        gridwidth=1,
        griddash="solid",
        zeroline=False,
        showline=False,
        color=THEME["muted"],
        title_font=dict(color=THEME["text_secondary"], size=12),
    )
    fig.update_traces(
        marker_cornerradius=4,
        selector=dict(type="bar"),
    )
    return fig


def label_bars(fig: go.Figure, template: str) -> go.Figure:
    """Value on every column cap. Normally labeling every mark is chartjunk, but
    these charts have 3-15 bars and the light-mode aqua slot needs the relief."""
    fig.update_traces(
        texttemplate=template,
        textposition="outside",
        textfont=dict(color=THEME["text_secondary"], size=11, family=FONT),
        cliponaxis=False,
        selector=dict(type="bar"),
    )
    return fig


def band_categories(fig: go.Figure, n_categories: int) -> go.Figure:
    """Faint alternating vertical bands, one per x-axis category cluster -- the
    visual-grouping technique from the Dribbble "Benchmarks" reference (see module
    docstring), toned down to a neutral tint since our bars aren't good/bad-coded
    the way that shot's green/pink bands were. Purely a reading aid: it separates
    each qtype's 3-bar cluster from its neighbor without adding a fourth color."""
    for i in range(n_categories):
        if i % 2 == 1:
            fig.add_vrect(
                x0=i - 0.5, x1=i + 0.5,
                fillcolor=THEME["muted"], opacity=0.08, line_width=0, layer="below",
            )
    return fig


def ring_markers(fig: go.Figure, size: int = 15) -> go.Figure:
    """>=8px markers with a 2px surface ring so overlapping points stay legible."""
    fig.update_traces(
        marker=dict(size=size, line=dict(width=2, color=THEME["surface"])),
        selector=dict(mode="markers"),
    )
    fig.update_traces(
        marker=dict(size=size, line=dict(width=2, color=THEME["surface"])),
        textfont=dict(color=THEME["text_secondary"], size=12, family=FONT),
        selector=dict(mode="markers+text"),
    )
    return fig


def swatch(label: str) -> str:
    """A colored dot beside ink-colored text -- identity never rides the text."""
    color = SERIES.get(label, THEME["muted"])
    return (
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{color};margin-right:8px;vertical-align:middle"></span>'
        f'<span style="font-weight:600;vertical-align:middle">{label}</span>'
    )


def chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


# ---------------------------------------------------------------- load & intro

path = bench.results_path()
if not path.exists():
    st.error(f"No benchmark results found at `{path}`. Run `python -m src.eval.run_benchmark` first.")
    st.stop()

df, stats = load(str(path), path.stat().st_mtime)
if df.empty:
    st.error(f"`{path}` has no valid (non-error) rows.")
    st.stop()

ORDER = bench.pipeline_order(df)
QTYPES = bench.qtype_order(df)

st.markdown('<div class="kicker">GRAPH_RAG BENCHMARK</div>', unsafe_allow_html=True)
st.title("RAG vs GraphRAG vs Agentic GraphRAG")
st.markdown(
    """
Three retrieval-augmented pipelines answer the same eval set of
**{n_q} questions**, so the only thing that varies is how each one finds and uses
evidence. **RAG** does plain vector retrieval over document chunks; **GraphRAG**
retrieves over an entity graph built from those documents; **Agentic GraphRAG**
lets an agent plan and issue several graph queries before answering. Every answer
is scored 1-5 by an LLM judge on accuracy and completeness (against the gold
answer) and groundedness (against the context the pipeline actually retrieved),
alongside retrieval quality (doc precision/recall against gold document ids) and
cost (latency, tokens, LLM calls). Judge scores are model judgments, not ground
truth, so read the per-question drill-down at the bottom before trusting any
single gap.
""".format(n_q=stats.get("unique_questions", df["qid"].nunique()))
)

excluded = stats.get("excluded_rows", 0)
if excluded:
    st.caption(
        f"Showing {stats['valid_rows']} scored rows "
        f"({' / '.join(f'{k} {v}' for k, v in stats.get('rows_per_pipeline', {}).items())}). "
        f"Excluded {excluded} error/retry rows from {stats['total_lines']} lines in the results file "
        f"({stats['error_rows']} errors, {stats['malformed_rows']} incomplete, "
        f"{stats['superseded_duplicates']} superseded duplicates)."
    )

# ------------------------------------------------------------------- filters

with st.sidebar:
    st.subheader("Filters")
    picked_qtypes = st.multiselect(
        "Question type",
        options=QTYPES,
        default=QTYPES,
        format_func=bench.qtype_label,
        help="Scopes every chart and the drill-down below.",
    )
    latency_field = st.radio(
        "Latency measure",
        options=["latency_sec", "total_latency_sec"],
        format_func=lambda f: bench.METRIC_LABELS[f],
        help="Pipeline latency excludes the judge call; end-to-end includes it.",
    )
    st.divider()
    st.caption(f"Source: `{Path(stats['path']).relative_to(bench.project_root())}`")

if not picked_qtypes:
    st.warning("Select at least one question type.")
    st.stop()

view = df[df["qtype"].isin(picked_qtypes)]
if view.empty:
    st.warning("No rows match the current filter.")
    st.stop()

filtered = len(picked_qtypes) < len(QTYPES)
scope = f"{', '.join(bench.qtype_label(q) for q in picked_qtypes)} questions" if filtered else "all questions"
latency_label = bench.METRIC_LABELS[latency_field]

# ------------------------------------------------------------------- summary

section_header("Summary")
if filtered:
    st.caption(f"Filtered to {scope} - {view['qid'].nunique()} questions.")

tiles = st.columns(len(ORDER))
means = view.groupby("pipeline_label", observed=True)
for col, name in zip(tiles, ORDER):
    if name not in means.groups:
        continue
    rows = means.get_group(name)
    with col:
        with st.container(border=True):
            st.markdown(swatch(name), unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="kpi-label">Mean judge score (1-5)</div>
                <div class="kpi-value">{rows['judge_mean'].mean():.2f}</div>
                <div class="kpi-sub">{rows[latency_field].mean():.1f}s {latency_label.split('(')[0].strip().lower()}
                    &middot; {rows['total_tokens'].mean():,.0f} tok &middot; {rows['num_llm_calls'].mean():.1f} calls/q</div>
                """,
                unsafe_allow_html=True,
            )

summary = bench.pipeline_summary(view)
st.dataframe(
    summary,
    hide_index=True,
    width="stretch",
    column_config={
        "Pipeline": st.column_config.TextColumn(width="medium"),
        "questions": st.column_config.NumberColumn("Questions", format="%d"),
        "Accuracy": st.column_config.NumberColumn(format="%.2f"),
        "Completeness": st.column_config.NumberColumn(format="%.2f"),
        "Groundedness": st.column_config.NumberColumn(format="%.2f"),
        "Doc precision": st.column_config.NumberColumn(format="%.3f"),
        "Doc recall": st.column_config.NumberColumn(format="%.3f"),
        "Pipeline latency (s)": st.column_config.NumberColumn(format="%.1f"),
        "End-to-end latency (s)": st.column_config.NumberColumn(format="%.1f"),
        "Total tokens": st.column_config.NumberColumn(format="%.0f"),
        "LLM calls": st.column_config.NumberColumn(format="%.2f"),
    },
)
st.caption("Every number is a mean over the questions in scope. Latency and tokens are per question.")

# -------------------------------------------------------------- judge scores

section_header("Answer quality")

scores = bench.long_judge_scores(view)
overall = scores.groupby(["dimension", "pipeline_label"], observed=True)["score"].mean().reset_index()

fig = px.bar(
    overall,
    x="dimension",
    y="score",
    color="pipeline_label",
    barmode="group",
    color_discrete_map=SERIES,
    category_orders={"pipeline_label": ORDER},
    labels={"dimension": "", "score": "Mean score (1-5)", "pipeline_label": ""},
    title=f"Mean judge score by dimension - {scope}",
)
fig.update_yaxes(range=[0, 5.6], dtick=1)
chart(style(label_bars(fig, "%{y:.2f}")), "judge-overall")

with st.expander("Score distributions (box) and the table view"):
    box = px.box(
        scores,
        x="dimension",
        y="score",
        color="pipeline_label",
        color_discrete_map=SERIES,
        category_orders={"pipeline_label": ORDER},
        labels={"dimension": "", "score": "Score (1-5)", "pipeline_label": ""},
        title="Per-question score distribution",
        points=False,
    )
    box.update_traces(line=dict(width=2))
    box.update_yaxes(range=[0.5, 5.5], dtick=1)
    chart(style(box), "judge-box")
    st.dataframe(
        overall.pivot(index="pipeline_label", columns="dimension", values="score")
        .round(2)
        .reset_index()
        .rename(columns={"pipeline_label": "Pipeline"}),
        hide_index=True,
        width="stretch",
    )

# ------------------------------------------------------ quality by qtype

section_header("Quality by question type")
st.markdown(
    "Where the aggregate hides the story: the pipelines do not win or lose uniformly, "
    "they trade places by question type."
)

dim_choice = st.selectbox(
    "Dimension",
    options=["judge_mean", *bench.JUDGE_DIMS],
    format_func=lambda d: bench.METRIC_LABELS[d],
)

by_qtype = bench.mean_by(view, [dim_choice], ["qtype", "pipeline_label"])
by_qtype["qtype_label"] = by_qtype["qtype"].astype(str).map(bench.qtype_label)

# METRIC_LABELS["judge_mean"] is already "Mean judge score" -- prefixing another
# "Mean " onto it produced "Mean mean judge score (1-5)" on the axis; only the
# per-dimension labels ("Accuracy" etc.) need the prefix added.
axis_label = bench.METRIC_LABELS[dim_choice] if dim_choice == "judge_mean" else f"Mean {bench.METRIC_LABELS[dim_choice].lower()}"

fig = px.bar(
    by_qtype,
    x="qtype_label",
    y=dim_choice,
    color="pipeline_label",
    barmode="group",
    color_discrete_map=SERIES,
    category_orders={
        "pipeline_label": ORDER,
        "qtype_label": [bench.qtype_label(q) for q in picked_qtypes],
    },
    labels={
        "qtype_label": "",
        dim_choice: f"{axis_label} (1-5)",
        "pipeline_label": "",
    },
    title=f"{bench.METRIC_LABELS[dim_choice]} by question type",
)
fig.update_yaxes(range=[0, 5.6], dtick=1)
fig = band_categories(style(label_bars(fig, "%{y:.1f}"), height=420), len(picked_qtypes))
chart(fig, "qtype-grouped")

with st.expander("All three dimensions, one panel per question type"):
    facet_src = bench.mean_by(view, bench.JUDGE_DIMS, ["qtype", "pipeline_label"]).melt(
        id_vars=["qtype", "pipeline_label"],
        value_vars=bench.JUDGE_DIMS,
        var_name="dimension",
        value_name="score",
    )
    facet_src["dimension"] = facet_src["dimension"].map(lambda d: bench.METRIC_LABELS[d][:4])
    facet_src["qtype_label"] = facet_src["qtype"].astype(str).map(bench.qtype_label)
    small = px.bar(
        facet_src,
        x="dimension",
        y="score",
        color="pipeline_label",
        barmode="group",
        facet_col="qtype_label",
        color_discrete_map=SERIES,
        category_orders={
            "pipeline_label": ORDER,
            "qtype_label": [bench.qtype_label(q) for q in picked_qtypes],
            "dimension": [bench.METRIC_LABELS[d][:4] for d in bench.JUDGE_DIMS],
        },
        labels={"dimension": "", "score": "Mean score (1-5)", "pipeline_label": ""},
        title="Accuracy / Completeness / Groundedness, faceted by question type",
    )
    small.update_yaxes(range=[0, 5.2], dtick=1)
    small.for_each_annotation(
        lambda a: a.update(
            text=a.text.split("=")[-1],
            font=dict(size=12, color=THEME["text_primary"], family=FONT),
        )
    )
    chart(style(small, height=360), "qtype-facets")
    st.dataframe(
        by_qtype.pivot(index="qtype_label", columns="pipeline_label", values=dim_choice)
        .round(2)
        .reset_index()
        .rename(columns={"qtype_label": "Question type"}),
        hide_index=True,
        width="stretch",
    )

# ------------------------------------------------------------ cost & tradeoff

section_header("Cost")
st.markdown(
    "Quality is not free: the same three colors, now on latency and tokens. "
    "The scatters put quality against cost so the tradeoff is one glance, not two charts."
)

left, right = st.columns(2)
with left:
    fig = px.box(
        view,
        x="pipeline_label",
        y=latency_field,
        color="pipeline_label",
        color_discrete_map=SERIES,
        category_orders={"pipeline_label": ORDER},
        labels={"pipeline_label": "", latency_field: latency_label},
        title=f"{latency_label} per question",
        points=False,
    )
    fig.update_traces(line=dict(width=2))
    chart(style(fig, showlegend=False), "latency-box")
with right:
    fig = px.box(
        view,
        x="pipeline_label",
        y="total_tokens",
        color="pipeline_label",
        color_discrete_map=SERIES,
        category_orders={"pipeline_label": ORDER},
        labels={"pipeline_label": "", "total_tokens": "Total tokens"},
        title="Tokens per question",
        points=False,
    )
    fig.update_traces(line=dict(width=2))
    chart(style(fig, showlegend=False), "tokens-box")

tradeoff = bench.mean_by(
    view, ["judge_mean", latency_field, "total_tokens", "num_llm_calls"], ["pipeline_label"]
)

left, right = st.columns(2)
for col, x_field, title in (
    (left, latency_field, f"Quality vs {latency_label.lower()}"),
    (right, "total_tokens", "Quality vs tokens"),
):
    with col:
        fig = px.scatter(
            tradeoff,
            x=x_field,
            y="judge_mean",
            color="pipeline_label",
            text="pipeline_label",
            color_discrete_map=SERIES,
            category_orders={"pipeline_label": ORDER},
            labels={
                x_field: bench.METRIC_LABELS[x_field] if x_field in bench.METRIC_LABELS else x_field,
                "judge_mean": "Mean judge score (1-5)",
                "pipeline_label": "",
            },
            title=f"{title} (up and to the left is better)",
        )
        fig.update_traces(textposition="top center")
        fig.update_yaxes(range=[0, 5.4], dtick=1)
        fig.update_xaxes(rangemode="tozero")
        chart(ring_markers(style(fig, showlegend=False)), f"tradeoff-{x_field}")

# ------------------------------------------------------------------ retrieval

section_header("Retrieval quality")
st.markdown(
    "Doc precision and recall compare each pipeline's retrieved document ids against "
    "the gold ids for that question, before the generator sees anything."
)

retrieval = bench.mean_by(view, ["doc_precision", "doc_recall"], ["pipeline_label"])

left, right = st.columns(2)
with left:
    melted = retrieval.melt(
        id_vars="pipeline_label", var_name="metric", value_name="value"
    )
    melted["metric"] = melted["metric"].map(bench.METRIC_LABELS)
    fig = px.bar(
        melted,
        x="metric",
        y="value",
        color="pipeline_label",
        barmode="group",
        color_discrete_map=SERIES,
        category_orders={"pipeline_label": ORDER},
        labels={"metric": "", "value": "Mean", "pipeline_label": ""},
        title="Mean doc precision and recall",
    )
    fig.update_yaxes(range=[0, 1.05], dtick=0.2)
    chart(style(label_bars(fig, "%{y:.2f}")), "retrieval-bars")
with right:
    fig = px.scatter(
        retrieval,
        x="doc_recall",
        y="doc_precision",
        color="pipeline_label",
        text="pipeline_label",
        color_discrete_map=SERIES,
        category_orders={"pipeline_label": ORDER},
        labels={
            "doc_recall": "Mean doc recall",
            "doc_precision": "Mean doc precision",
            "pipeline_label": "",
        },
        title="Precision vs recall (up and to the right is better)",
    )
    fig.update_traces(textposition="top center")
    fig.update_xaxes(range=[0, 1], dtick=0.2)
    fig.update_yaxes(range=[0, 1], dtick=0.2)
    chart(ring_markers(style(fig, showlegend=False)), "retrieval-scatter")

# ------------------------------------------------------------------ drilldown

section_header("Per-question drill-down")
st.markdown("Spot-check a single question across all three pipelines, side by side.")

qids = list(dict.fromkeys(view.sort_values(["qtype", "qid"])["qid"].tolist()))
questions = view.drop_duplicates("qid").set_index("qid")


def qid_label(qid: str) -> str:
    row = questions.loc[qid]
    text = str(row["question"])
    return f"{qid} - {bench.qtype_label(row['qtype'])} - {text[:70]}{'...' if len(text) > 70 else ''}"


picked_qid = st.selectbox("Question", options=qids, format_func=qid_label)

detail = view[view["qid"] == picked_qid].set_index("pipeline_label")
head = questions.loc[picked_qid]

st.markdown(f"**Question** ({bench.qtype_label(head['qtype'])})")
st.info(str(head["question"]))
st.markdown("**Reference answer**")
st.success(str(head["reference_answer"]) or "_(empty)_")

cols = st.columns(len(ORDER))
for col, name in zip(cols, ORDER):
    with col:
        st.markdown(swatch(name), unsafe_allow_html=True)
        if name not in detail.index:
            st.caption("No scored row for this question.")
            continue
        row = detail.loc[name]
        a, c, g = st.columns(3)
        a.metric("Acc", "-" if row["accuracy"] != row["accuracy"] else f"{row['accuracy']:.0f}")
        c.metric("Comp", "-" if row["completeness"] != row["completeness"] else f"{row['completeness']:.0f}")
        g.metric("Grnd", "-" if row["groundedness"] != row["groundedness"] else f"{row['groundedness']:.0f}")
        st.caption(
            f"P {row['doc_precision']:.2f} - R {row['doc_recall']:.2f} - "
            f"{row[latency_field]:.1f}s - {row['total_tokens']:,.0f} tok - "
            f"{row['num_llm_calls']:.0f} calls"
        )
        st.markdown("**Answer**")
        st.write(str(row["answer"]) or "_(empty)_")
        with st.expander("Judge reasoning"):
            st.write(str(row["judge_reasoning"]) or "_(none)_")

with st.expander(f"All rows in scope ({len(view)})"):
    table = view[
        [
            "qid",
            "qtype",
            "pipeline_label",
            "accuracy",
            "completeness",
            "groundedness",
            "doc_precision",
            "doc_recall",
            latency_field,
            "total_tokens",
            "num_llm_calls",
            "question",
            "answer",
            "reference_answer",
        ]
    ].rename(columns={"pipeline_label": "pipeline", **bench.METRIC_LABELS})
    st.dataframe(table, hide_index=True, width="stretch", height=420)
